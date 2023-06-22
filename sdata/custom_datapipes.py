# Custom datapipes, partly copied from release 0.6.0 to also support 0.5.1, shout out to pytorch and torchdata
import os
import tarfile
import time
import warnings
from io import BufferedIOBase, RawIOBase
import re
from typing import (
    Iterator,
    List,
    Tuple,
    Optional,
    cast,
    IO,
    Callable,
    Dict,
    Union,
)
import random
import gc
from collections import deque


import numpy as np
import torch
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter import TarArchiveLoader
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from torch.utils.data.datapipes.utils.common import StreamWrapper
import webdataset as wds



def _is_stream_handle(data):
    obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
    return isinstance(obj_to_check, (BufferedIOBase, RawIOBase))


def _shard_expand(s: str) -> List[str]:
    expansion = r"\{[0-9]+\.\.[0-9]+\}"
    m = re.search(expansion, s)
    if not m:
        return [s]
    prefix = s[: m.start()]
    rest = _shard_expand(s[m.end() :])
    rng = s[m.start() + 1 : m.end() - 1]
    lohi = rng.split("..")
    if len(lohi[0]) == len(lohi[1]) and lohi[0].startswith("0"):
        fmt = "{prefix}{i:0>{l}d}{r}"
    elif len(lohi[0]) <= len(lohi[1]):
        if lohi[0].startswith("0") and lohi[0] != "0":
            raise ValueError(
                "shard_expand: low bound must not start with 0 if low bound is shorter"
            )
        fmt = "{prefix}{i}{r}"
    else:
        raise ValueError("shard_expand: low bound must be shorter than high bound")
    lo, hi = (int(x) for x in lohi)
    if lo >= hi:
        raise ValueError(f"shard_expand: bad range in in shard spec {s}.")
    result = []
    for i in range(lo, hi + 1):
        for r in rest:
            expanded: str = fmt.format(prefix=prefix, i=i, r=r, l=len(lohi[1]))
            result.append(expanded)
    return result


class CustomShardExpanderIterDataPipe(IterDataPipe[str]):
    r"""
    Expands incoming shard strings into shards.

    Sharded data files are named using shell-like brace notation. For example,
    an ImageNet dataset sharded into 1200 shards and stored on a web server
    might be named `imagenet-{000000..001199}.tar`.

    Note that shard names can be expanded without any server transactions;
    this makes `shard_expand` reproducible and storage system independent
    (unlike :class `.FileLister` etc.).

    Args:
        source_datapipe: a DataPipe yielding a stream of  pairs

    Returns:
        a DataPipe yielding a stream of expanded pathnames.

    Example:
        from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(["ds-{00..05}.tar"])
        >>> expand_dp = source_dp.shard_expand()
        >>> list(expand_dp)
        ['ds-00.tar', 'ds-01.tar', 'ds-02.tar', 'ds-03.tar', 'ds-04.tar', 'ds-05.tar']
        >>> source_dp = IterableWrapper(["imgs_{00..05}.tar", "labels_{00..05}.tar"])
        >>> expand_dp = source_dp.shard_expand()
        >>> list(expand_dp)
        ['imgs_00.tar', 'imgs_01.tar', 'imgs_02.tar', 'labels_00.tar', 'labels_01.tar', 'labels_02.tar']
    """

    def __init__(self, source_datapipe: IterDataPipe[str]) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[str]:
        for path in self.source_datapipe:
            yield from _shard_expand(path)


class SeedSetter(IterDataPipe):
    """
    Resets the seed on call of __iter__ (invoked in the reset() method
    """

    def __init__(self, datapipe, debug=False):
        super().__init__()
        self.datapipe = datapipe
        self.is_init = False
        self.debug = False

    # # def reset(self):
    def reset(self):
        # this will be called whenever __iter__ is invoked again (this should be kept in mind for shuffling
        if not self.is_init:
            # we only wanna do this once
            self.is_init = True

            worker_info = torch.utils.data.get_worker_info()

            if worker_info:
                worker_id = worker_info.id
                newseed = np.random.get_state()[1][0] + worker_id
                if self.debug:
                    print(f"Worker #{worker_id} reseeding with {newseed}")
                np.random.seed(newseed)
                torch.random.manual_seed(newseed)
                random.seed(newseed)

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        # self.set_seed()
        # print(f'seed in worker init: {seed}')
        for data in self.datapipe:
            yield data


class SplitByWorker(IterDataPipe):
    """
    distributed data across workers to mimic behavior of shard splitting in webdataset
    """

    def __init__(self, datapipe, debug: bool = False):
        super().__init__()
        self.datapipe = datapipe
        # self.drop_last = drop_last
        self.worker_id = 0
        self.num_workers = 1
        self.debug = debug
        self.do_print = True

    def reset(self):
        # this will be called whenever __iter__ is invoked again (this should be kept in mind for shuffling
        worker_info = torch.utils.data.get_worker_info()
        if self.debug:
            print(f"worker {worker_info} configured")
        if worker_info:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for i, data in enumerate(self.datapipe):
            # # avoid hanging due to uneven number of shards per worker
            if i % self.num_workers == self.worker_id:
                if self.debug and self.do_print:
                    print(f"data worker {self.worker_id} got first shard {data}")
                    self.do_print = False
                yield data


class PrefixResampler(IterDataPipe):
    def __init__(
        self,
        datapipe: IterDataPipe[Tuple],
        prefixes: List[str],
        ps: List[float] = None,
        is_braceexpand: bool = False,
        buffersize_per_prefix: Union[int, Dict] = 100000,
        custom_seeding: bool=True,
        debug: bool = True,
        handler: Callable = wds.reraise_exception,
    ):
        super().__init__()
        self.source = datapipe
        if is_braceexpand:
            # only dirs
            prefixes = [os.path.split(p.split("{")[0])[0] for p in prefixes]

        assert len(set(prefixes)) == len(prefixes), "Prefixes should be unique"
        self.ps = {k: p for k, p in zip(prefixes, ps)}

        print(f"{self.__class__.__name__} got the following prefixes: {prefixes}")

        if isinstance(buffersize_per_prefix, int):
            buffersize_per_prefix = {pref: buffersize_per_prefix for pref in prefixes}

        assert len(buffersize_per_prefix) == len(
            prefixes
        ), f"Buffersize per prefix (len={len(buffersize_per_prefix)}) has to have the same length than prefixes (len={len(prefixes)})"
        self.url_buffer = {
            prf: deque(maxlen=buffersize_per_prefix[prf]) for prf in prefixes
        }
        self.warn_once = {prf: True for prf in self.url_buffer}

        sum_ = sum(list(self.ps.values()))
        self.ps = {k: self.ps[k] / sum_ for k in self.ps}

        print(
            f"Got the following (prob, prefix) pairs for {len(self.ps)} prefixes {[(k, p) for k, p in self.ps.items()]}"
        )

        self.handler = handler
        self.is_init = not custom_seeding
        self.debug = debug

        assert np.isclose(
            sum(self.ps.values()), 1.0
        ), "Probabilities must have the same length than prefix and must sum up to 1"

    def reset(self):
        if self.debug:

            worker_info = torch.utils.data.get_worker_info()

            if worker_info:
                worker_id = worker_info.id
                print(f"Worker #{worker_id} has seed {np.random.get_state()[1][0]}")

    def __iter__(self):
        keep_target = False
        target_prefix=None
        for url in self.source:
            try:
                assert isinstance(
                    url, (tuple, Tuple)
                ), f"source datapipe of {self.__class__.__name__} should yield tuples"
                key, content = url
                if not keep_target:
                    keep_target = True
                    target_prefix = np.random.choice(
                        list(self.ps), 1, p=list(self.ps.values())
                    ).item()
                current_prefix = list(filter(lambda x: key.startswith(x), self.ps))
                if not len(current_prefix) == 1:
                    raise ValueError(
                        f"the received prefix is non-unique and matches "
                        f"all of {current_prefix}, aborting"
                    )
                current_prefix = current_prefix[0]

                if (
                    len(self.url_buffer[current_prefix])
                    >= self.url_buffer[current_prefix].maxlen
                ):
                    maxsize = self.url_buffer[current_prefix].maxlen
                    if self.warn_once[current_prefix]:
                        self.warn_once[current_prefix] = False
                        warnings.warn(
                            f"buffer size for prefix {current_prefix} in {self.__class__.__name__} exceeds its max buffer size {maxsize},"
                            f"thus discarding this element. "
                            f"Is this intended?"
                        )
                else:
                    self.url_buffer[current_prefix].append(url)

                if current_prefix == target_prefix:
                    keep_target = False
                    # FIFO
                    out = self.url_buffer[target_prefix].popleft()[1]
                    yield out
            except Exception as e:
                if self.handler(e):
                    pass
                else:
                    raise e


class Dataset2SamplesConverter(IterDataPipe):
    def __init__(
        self, datapipe: IterDataPipe, handler: Callable = wds.reraise_exception
    ):
        super().__init__()
        self.datapipe = datapipe
        self.handler = handler

    def __iter__(self) -> Iterator[Dict]:
        try:
            for sample in self.datapipe:
                try:
                    # dict-style sample from tuple
                    key = os.path.split(sample[0][0])[-1].split(".")[0]
                    url = os.path.split(sample[0][0])[0]
                    out = {}
                    for s in sample:
                        key_ = (
                            s[0].split(key)[-1][1:]
                            if s[0].split(key)[-1].startswith(".")
                            else s[0].split(key)[-1]
                        )
                        data = s[1]

                        if _is_stream_handle(data):
                            ds = data
                            # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
                            data = b"".join(data)
                            ds.close()
                            del ds

                        out[key_] = data
                        del data
                    sample = out
                    del out
                    sample["__key__"] = key
                    sample["__url__"] = url

                    yield sample

                except Exception as exn:
                    if self.handler(exn):
                        gc.collect()
                        continue
                    else:
                        break

        except Exception as e:
            if self.handler(e):
                print(f"Catched exception in {self.__class__.__name__}: ", e)
            else:
                print(f"Catched exception in {self.__class__.__name__}: ", e)
                raise e


class TarArchiveLoaderAndCloser(TarArchiveLoader):
    def __init__(self, handler: Callable = wds.reraise_exception, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handler
        self.times = None
        self.profile = False

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            start = time.perf_counter()
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(
                    data_stream.file_obj, tarfile.TarFile
                ):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (
                        self.mode
                        if hasattr(data_stream, "seekable") and data_stream.seekable()
                        else self.mode.replace(":", "|")
                    )
                    # typing.cast is used here to silence mypy's type checker
                    tar = tarfile.open(
                        fileobj=cast(Optional[IO[bytes]], data_stream),
                        mode=reading_mode,
                    )
                    if self.profile:
                        self.open_times.append(time.perf_counter() - start)
                try:
                    for tarinfo in tar:
                        start = time.perf_counter()
                        if not tarinfo.isfile():
                            continue
                        extracted_fobj = tar.extractfile(tarinfo)
                        if extracted_fobj is None:
                            warnings.warn(
                                f"failed to extract file {tarinfo.name} from source tarfile {pathname}"
                            )
                            raise tarfile.ExtractError
                        inner_pathname = os.path.normpath(
                            os.path.join(pathname, tarinfo.name)
                        )
                        sw = StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)  # type: ignore[misc]

                        if self.profile:
                            self.extract_times.append(time.perf_counter() - start)
                        yield inner_pathname, sw
                        # sw.autoclose()
                        del sw
                    # close tarfile after it's been exceeded
                finally:
                    tar.close()
                    del tar
                    del tarinfo

                    if _is_stream_handle(data_stream):
                        data_stream.autoclose()
                    del data_stream
                    gc.collect()
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!"
                )
                if self.handler(e):
                    if hasattr(e, "args") and len(e.args) > 0:
                        e.args = (e.args[0] + " @ " + str(pathname),) + e.args[1:]
                else:
                    raise e




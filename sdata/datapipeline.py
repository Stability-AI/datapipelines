import importlib
import os
from typing import Callable, Optional, Union, List, Any, Dict
import re
from packaging import version
from operator import itemgetter
import functools
import time
import warnings
import threading


from omegaconf import ListConfig, DictConfig
import torchdata
import torch.distributed as dist

from torch.utils.data.datapipes.iter import IterableWrapper, FileOpener
from torchdata.datapipes.iter import IterKeyZipper
import webdataset as wds

from .custom_datapipes import (
    CustomShardExpanderIterDataPipe,
    SplitByWorker,
    PrefixResampler,
    TarArchiveLoaderAndCloser,
    SeedSetter,
    Dataset2SamplesConverter,
    _is_stream_handle,
)

class TimeoutError(Exception):
    pass


def timeout_wrapper(func):
    def wrapper(*args, **kwargs):
        if (
            "SDATA_MAX_EXC_TIME" not in os.environ
            or not os.environ["SDATA_MAX_EXC_TIME"]
        ):
            res = func(*args, **kwargs)
            del args
            del kwargs
            return res

        timeout = float(os.environ["SDATA_MAX_EXC_TIME"])

        result = [None]
        exception = [None]
        event = threading.Event()

        def wrapped_func():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
            finally:
                event.set()

        thread = threading.Thread(target=wrapped_func)
        thread.start()
        event.wait(timeout)

        if not event.is_set():
            raise TimeoutError(
                f"Function call timed out (longer than {timeout  } secs)."
            )

        thread.join()

        if exception[0] is not None:
            raise exception[0]

        del thread
        del exception
        del wrapped_func
        del event
        del args
        del kwargs

        return result[0]

    return wrapper


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    print(exn)
    warnings.warn(repr(exn))
    time.sleep(0.05)
    return True


def time_measure(name: str = "function"):
    def wrapper(fn):
        def measure_time(*args, **kwargs):
            start = time.perf_counter()
            r = fn(*args, **kwargs)
            end = time.perf_counter()
            if "SDATA_PROFILE" in os.environ and os.environ["SDATA_PROFILE"]:
                if r is None:
                    return r
                try:
                    if isinstance(r, Dict):
                        r[f"{name}-time"] = end - start
                    else:
                        args[1][f"{name}-time"] = end - start

                except Exception as e:
                    print(f"Exception raised when measuring time for {name}")
                    raise e

            del args
            del kwargs

            return r

        return measure_time

    return wrapper


def instantiate(config: Union[Dict, DictConfig]) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return create_obj(config["target"])(**config.get("params", dict()))


def make_callable(config):
    return functools.partial(
        create_obj(config["target"]), **config.get("params", dict())
    )


def create_obj(string: str, reload: bool = False, invalidate_cache: bool = True) -> Any:
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class KeyPassThroughDecoder(wds.Decoder):
    def __init__(self, *args, passthrough_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.passthrough_keys = passthrough_keys
        if self.passthrough_keys is None:
            self.passthrough_keys = []

    def decode1(self, key, data):
        # if data is a stream handle, we need to read all the content before decoding
        if _is_stream_handle(data):
            ds = data
            # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
            data = b"".join(data)
            ds.close()

        key = "." + key
        for f in self.handlers:
            result = f(key, data)
            if isinstance(result, wds.autodecode.Continue):
                key, data = result.key, result.data
                continue
            if result is not None:
                del data
                return result
        return data

    @timeout_wrapper
    @time_measure(name="KeyPassThroughDecoder")
    def decode(self, sample):
        """Decode an entire sample.

        :param sample: the sample, a dictionary of key value pairs
        """
        result = {}
        assert isinstance(sample, dict), sample
        for k, v in list(sample.items()):
            if k[0] == "_":
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                result[k] = v
                continue
            if self.only is not None and k not in self.only:
                result[k] = v
                continue
            assert v is not None
            if self.partial:
                if isinstance(v, bytes) or k in self.passthrough_keys:
                    result[k] = self.decode1(k, v)
                else:
                    result[k] = v
            else:
                assert (
                    isinstance(v, bytes) or k in self.passthrough_keys
                ), f"key: {k}; passthrough_keys: {self.passthrough_keys}"
                result[k] = self.decode1(k, v)
        return result


def tarfilter(x):
    ret = x.endswith(".tar")
    del x
    return ret


def grouper(x):
    key = x[0].split("/")[-1].split(".")[0]
    del x
    return key


def tuple_grouper(x):
    key = x[0][0].split("/")[-1].split(".")[0]
    del x
    return key


def merge_samples(s1, s2, meta_urls):
    s1_files = [os.path.splitext(s[0])[1] for s in s1]
    meta_key_list = [mk for mk in meta_urls if mk in s2[0][0]]
    if len(meta_key_list) == 0:
        raise ValueError(
            f"no matching meta key found for the following file(s): {os.path.splitext(s2[0][0])[0]}"
        )
    elif len(meta_key_list) > 1:
        raise ValueError(
            f"More than one matching meta key found for the following file(s): {os.path.splitext(s2[0][0])[0]}"
        )

    meta_key = meta_key_list[0]
    outs2 = [
        s
        if os.path.splitext(s[0])[1] not in s1_files
        else (os.path.splitext(s[0])[0] + meta_key + os.path.splitext(s[0])[1], s[1])
        for s in s2
    ]
    del s2
    return list(s1) + outs2


def merge_them(u1, u2):
    # concat lists: these lists should contain all tarfiles from the same prefix but
    # with different filenames
    return u1[1] + [
        u2,
    ]


def identity(x):
    return True


def map_to_tuple(x):
    return (
        os.path.join(os.path.split(x)[0], os.path.splitext(os.path.split(x)[1])[0]),
        [
            x,
        ],
    )


def filter_with_meta_set(x, meta_set):
    return itemgetter(0)(x) in meta_set


def get_ref_key(x, suffix):
    return os.path.splitext(x.replace("_" + suffix, ""))[0]


def list_files_in_datapipe(
    urls: Union[List, ListConfig],
    is_braceexpand: bool,
    tar_sampler: Callable = identity,
) -> torchdata.datapipes.iter.IterDataPipe:
    """

    :param datapipe:
    :param is_braceexpand:
    :return:
    """
    datapipe = IterableWrapper(urls)

    if version.parse(torchdata.__version__) >= version.parse("0.6.0"):
        if is_braceexpand:
            datapipe = CustomShardExpanderIterDataPipe(datapipe)
        else:
            datapipe = datapipe.list_files(recursive=True).filter(tarfilter)
    else:
        if is_braceexpand:
            datapipe = CustomShardExpanderIterDataPipe(datapipe)
        else:
            datapipe = datapipe.list_files(recursive=True).filter(tarfilter)

    datapipe = datapipe.filter(tar_sampler)

    return datapipe


class StableDataPipeline(wds.DataPipeline, wds.compat.FluidInterface):
    """
    Central class for reading data from tars on local fs and building samples based on consecutive files with the same keys
    """

    def __init__(
        self,
        urls: Union[List[str], str, ListConfig],
        meta_urls: Optional[Union[List[str], str]] = None,
        metadata_buffer_size: Union[int, None] = 10000,
        repeat: int = None,
        shardshuffle: int = 10000,
        sample_shuffle: int = 1,
        resample_prefixes: bool = False,
        prefix_probs: Optional[List[float]] = None,
        split_data_by_worker: bool = True,
        tar_sampler: Optional[Union[DictConfig, Dict, Callable]] = identity,
        handler: Union[Callable, DictConfig] = wds.reraise_exception,
        debug: bool = False,
        n_shards: int = 100000,
    ):
        """

        :param urls: folders to load the shards from, can be a list of different prefoxes for dataset mixing
        :param meta_urls: can be used for aligned metadata files stored as tars
        :param metadata_buffer_size:
        :param repeat: number of repetitions in the training data. Default is None which means looping perpetually.
        :param shardshuffle: Shuffle buffer size for shard shuffling. size 1 means no shufflin. Default is 10k.
        :param sample_shuffle: Shuffle buffer for sample-level-shuffling. Default is 1 which means no shuffling
        :param resample_prefixes: Whether to resample when different prefixes are in the entire dataset.
         This can be useful in combination with prefix probs when training on merged datasets of non-equal size.
        :param prefix_probs: list containing resampling probabilities for every prefix in `urls`
        :param split_data_by_worker: Whether to split shards across worker threads for num_workers > 0
        :param handler: handler for handling exceptions as in webdataset
        """
        super().__init__()

        if isinstance(urls, (List, ListConfig, list)):
            pass
        elif isinstance(urls, str):
            urls = [urls]
        else:
            raise TypeError(
                "urls need to be path to a S3 prefix or list of paths to more than one prefixes"
            )

        if isinstance(handler, (DictConfig, Dict)):
            handler = make_callable(handler)


        # get some information abt fs where shards live in and the way shards are specified
        is_braceexpand = any(["{" in u for u in urls])

        if is_braceexpand:
            brace_expansion = re.compile(r"\{[0-9]+\.\.[0-9]+\}")
            assert all(len(re.findall(brace_expansion, u)) == 1 for u in urls), (
                "Specifiying tars in listed prefixes should be consistent. "
                "It should be either braceexpand notation or just using some "
                "base prefix. If this still fails, you might have some urls with "
                "multiple or malformed braceexpands."
            )

        if isinstance(tar_sampler, (Dict, dict, DictConfig)):
            tar_sampler = instantiate(tar_sampler)

        main_datapipe = list_files_in_datapipe(
            urls,
            is_braceexpand=is_braceexpand,
            tar_sampler=tar_sampler,
        ).map(fn=map_to_tuple)

        if meta_urls:
            print(
                f"Zipping together {len(meta_urls)} meta datapipes with the following suffixes {meta_urls} "
                f"and adding this to the main datapipes "
            )

            if isinstance(meta_urls, str):
                meta_urls = [meta_urls]

            meta_urls_base = [os.path.split(m) for m in urls]
            # meta_urls = [[os.path.join(m[0], os.path.splitext(m[1])[0]+f"_{suffix}"+os.path.splitext(m[1])[1]) for m in meta_urls_base] for suffix in meta_urls]
            meta_files = [
                [os.path.join(m[0] + f"_{suffix}", m[1]) for m in meta_urls_base]
                for suffix in meta_urls
            ]

            for suffix, meta_url_collection in zip(meta_urls, meta_files):
                # this is the meta data which will be added to the man data
                meta_datapipe = list_files_in_datapipe(
                    meta_url_collection,
                    is_braceexpand=is_braceexpand,
                    tar_sampler=tar_sampler,
                )
                # filter out non-exisiting shards
                meta_set = set([get_ref_key(pth, suffix) for pth in meta_datapipe])
                main_datapipe = main_datapipe.filter(
                    functools.partial(filter_with_meta_set, meta_set=meta_set)
                )

                # cycle in side branch to avoid exhausting after iterating over the entire dataset
                meta_datapipe = meta_datapipe.cycle()
                # merging always based on filenames where the metadata shards are expected to have <main_shard_id>.tar,
                # e.g. for a main shard "0000.tar" and an optical flow metadatashard we'd have "0000.tar" for the metadata shard
                # and the resulting key would be /path/to/prefix/0000
                main_datapipe = IterKeyZipper(
                    main_datapipe,
                    ref_datapipe=meta_datapipe,
                    key_fn=itemgetter(0),
                    ref_key_fn=functools.partial(get_ref_key, suffix=suffix),
                    keep_key=True,
                    merge_fn=merge_them,
                    buffer_size=metadata_buffer_size,
                )
        # main_datapipe = main_datapipe

        # start shuffling accross shards for the first time to mix different datasets
        # (can be the same for all workers, just as an additional shuffled initialization)
        if shardshuffle > 1 and not resample_prefixes and len(urls) > 1:
            # back to datapipes. We further apply a map to remove the key, so that the result is the sames than
            # for the prefix subsampler
            main_datapipe = main_datapipe.shuffle(buffer_size=n_shards).map(
                fn=itemgetter(1)
            )
        elif resample_prefixes:
            main_datapipe = PrefixResampler(
                main_datapipe.shuffle(buffer_size=n_shards),
                ps=prefix_probs,
                prefixes=urls,
                is_braceexpand=is_braceexpand,
                custom_seeding=split_data_by_worker,
                debug=debug
            )
        else:
            main_datapipe = main_datapipe.map(itemgetter(1))

        if not resample_prefixes:
            shardshuffle = max(shardshuffle, 1)
            main_datapipe = main_datapipe.shuffle(buffer_size=shardshuffle)

        main_datapipe = main_datapipe.sharding_filter()

        # after this operation datapipes in the distinct processes contain different tars
        if dist.is_available() and dist.is_initialized():
            # after this operation datapipes in the distinct processes contain different tars

            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
            main_datapipe.apply_sharding(world_size, global_rank)
            print("#" * 100)
            print(f"distributing shards for worker with global rank {global_rank}")
            print("#" * 100)

        else:
            print(
                f"torch distributed not used, not applying sharding in {self.__class__.__name__}"
            )

        if split_data_by_worker:
            print("Distributing shards across the worker threads in every process")
            main_datapipe = SplitByWorker(
                datapipe=main_datapipe, debug=debug
            )
        else:
            main_datapipe = SeedSetter(main_datapipe, debug=debug)

        main_datapipe = main_datapipe.cycle(count=repeat)

        # unzip before loading, since here we can be sure that all shards are distributed and shuffled
        # aligned with their corresponding metadata shards
        meta_len = len(meta_urls) if meta_urls else 0
        main_datapipe, *meta_datapipes = main_datapipe.unzip(
            sequence_length=meta_len + 1
        )


        # regular fileopener
        main_datapipe = FileOpener(main_datapipe, mode="b")
        meta_datapipes = [FileOpener(m, mode="b") for m in meta_datapipes]

        # adapted TarLoader which closes open tarfile handles after exceeding them
        # main_datapipe = TarArchiveLoaderAndCloser(datapipe=main_datapipe).groupby(grouper)
        #
        main_datapipe = TarArchiveLoaderAndCloser(
            datapipe=main_datapipe, handler=handler
        ).groupby(grouper)
        meta_datapipes = [
            TarArchiveLoaderAndCloser(datapipe=m, handler=handler).groupby(grouper)
            for m in meta_datapipes
        ]

        # zip again, this time we're searching based on the same keys
        for meta_dp in meta_datapipes:
            # here we da
            main_datapipe = IterKeyZipper(
                main_datapipe,
                ref_datapipe=meta_dp,
                key_fn=tuple_grouper,
                merge_fn=functools.partial(merge_samples, meta_urls=meta_urls),
                buffer_size=metadata_buffer_size,
            )

        if sample_shuffle > 0:
            main_datapipe = main_datapipe.shuffle(buffer_size=sample_shuffle)

        main_datapipe = Dataset2SamplesConverter(main_datapipe, handler=handler)
        self.append(main_datapipe)
        # self.append(dataset2samples(handler=handler))

    def decode(
        self,
        *args,
        pre=None,
        post=None,
        only=None,
        partial=False,
        passthrough_keys=None,
        handler=wds.reraise_exception,
    ):
        handlers = [
            wds.autodecode.ImageHandler(x) if isinstance(x, str) else x for x in args
        ]
        decoder = KeyPassThroughDecoder(
            handlers,
            passthrough_keys=passthrough_keys,
            pre=pre,
            post=post,
            only=only,
            partial=partial,
        )
        return self.map(decoder, handler=handler)

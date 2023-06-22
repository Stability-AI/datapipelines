import torch
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig, ListConfig
from typing import Dict, List, Optional, Union, Callable
import webdataset as wds


from .filters import AbstractFilter
from .mappers import AbstractMapper
from .datapipeline import StableDataPipeline, instantiate, make_callable, time_measure


@time_measure("Collator")
def dict_collation_fn(
    samples: List, combine_tensors: bool = True, combine_scalars: bool = True
) -> Dict:
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])

    del samples
    del batched
    return result


def create_loader(
    datapipeline: Union[wds.DataPipeline, Union[DictConfig, Dict]],
    batch_size: int,
    num_workers: int,
    partial: bool = False,
    collation_fn: Optional[Union[Callable, Dict, DictConfig]] = None,
    batched_transforms: Optional[ListConfig] = None,
    loader_kwargs: Optional[Union[Dict, DictConfig]] = None,
    error_handler: Optional[Union[Callable, Dict, DictConfig]] = None,
) -> torch.utils.data.DataLoader:
    if not loader_kwargs:
        loader_kwargs = {}

    if not batched_transforms:
        batched_transforms = []

    if not collation_fn:
        collation_fn = dict_collation_fn

    if isinstance(collation_fn, (Dict, DictConfig)):
        collation_fn = make_callable(collation_fn)

    if not error_handler:
        error_handler = {"target": "sdata.warn_and_continue"}

    if isinstance(error_handler, (Dict, DictConfig)):
        error_handler = make_callable(error_handler)

    print("#" * 100)
    print("Building dataloader with the following parameters")
    print(f"batch_size: {batch_size}")
    print(f"num_workers: {num_workers}")
    for key in loader_kwargs:
        print(key, ": ", loader_kwargs[key])
    print("#" * 100)
    # create datapipeline from dict if not already instantiated
    if isinstance(datapipeline, (DictConfig, Dict)):
        datapipeline = instantiate(datapipeline)

    # batching
    datapipeline = datapipeline.batched(
        batch_size, partial=partial, collation_fn=collation_fn
    )

    # apply transforms which act on batched samples
    for i, trf in enumerate(batched_transforms):
        trf = instantiate(trf)
        if isinstance(trf, AbstractFilter):
            print(
                f"Adding filter {trf.__class__.__name__} as batched transform #{i} "
                f"to the datapipeline"
            )
            datapipeline = datapipeline.select(trf)
        elif isinstance(trf, AbstractMapper):
            print(
                f"Adding mapper {trf.__class__.__name__} as batched transform #{i} "
                f"to the datapipeline"
            )
            datapipeline = datapipeline.map(trf, handler=error_handler)
        else:
            raise TypeError(
                "chosen batched transform should be either a subclass of "
                "sdata.AbstractMapper or one of sdata.AbstractFilter"
                "but is none of both"
            )

    # create loader
    loader = torch.utils.data.DataLoader(
        datapipeline, batch_size=None, num_workers=num_workers, **loader_kwargs
    )
    return loader


def create_dataset(
    urls: Union[List, ListConfig, str],
    pipeline_config: Optional[Union[DictConfig, Dict]] = None,
    decoders: Optional[Union[ListConfig, str]] = "pil",
    additional_decoder_kwargs: Optional[DictConfig] = None,
    preprocessors: Optional[ListConfig] = None,
    postprocessors: Optional[ListConfig] = None,
    error_handler: Optional[Union[Callable, Dict]] = None,
) -> wds.DataPipeline:
    """
    Create a dataset from several (partly optional) configs and urls defining paths to shards in webdataset/torchdata format
    The shards should be located in the local filesystem and can be specified as directories or in braceexpand notation
    :param urls: the urls as paths to the shards
    :param pipeline_config: additional parameters for configuring the main datapipeline, for a list of
    available parameters, see sdata.
    :param decoders:
    :param additional_decoder_kwargs: Additional keyword args for the decoder. This can be e.g. used to define passthrough keys, which shall be
    decoded although not having a known decoder key
    :param preprocessors:
    :param postprocessors:
    :param error_handler: The error handler defining a strategy for handling errors in the stages in the pipeline
    :return:
    """
    if isinstance(urls, str):
        urls = [urls]

    if not pipeline_config:
        pipeline_config = {}

    if not error_handler:
        error_handler = {"target": "sdata.warn_and_continue"}

    pipeline_config.pop("handler", None)
    pipeline_config["handler"] = error_handler

    # default for all processors
    if not preprocessors:
        preprocessors = []

    if not postprocessors:
        postprocessors = []

    if not additional_decoder_kwargs:
        additional_decoder_kwargs = {}

    if decoders and not isinstance(decoders, (List, ListConfig)):
        # default case is assuming image decoding
        decoders = [decoders]

    elif not decoders:
        decoders = []

    datapipeline = StableDataPipeline(urls=urls, **pipeline_config)

    if isinstance(error_handler, Dict):
        error_handler = make_callable(error_handler)

    # instantiate all preprocessors
    for i, prepro_config in enumerate(preprocessors):
        prepro = instantiate(prepro_config)
        if isinstance(prepro, AbstractFilter):
            print(
                f"Adding filter {prepro.__class__.__name__} as preprocessor #{i} "
                f"to the datapipeline"
            )
            datapipeline = datapipeline.select(prepro)
        elif isinstance(prepro, AbstractMapper):
            print(
                f"Adding mapper {prepro.__class__.__name__} as preprocessor #{i} "
                f"to the datapipeline"
            )
            datapipeline = datapipeline.map(prepro, handler=error_handler)
        else:
            raise TypeError(
                f"chosen preprocessor {prepro.__class__.__name__} should be either a subclass of "
                "sdata.mappers.AbstractMapper or one of sdata.filters.AbstractFilter"
                "but is none of both"
            )

    # do decoding
    prepared_decoders = []
    for decoder_spec in decoders:
        if isinstance(decoder_spec, (Dict, DictConfig)):
            decoder = instantiate(decoder_spec)
            print(f"Adding decoder {decoder.__class__.__name__} to decoders.")
            prepared_decoders.append(decoder)
        elif isinstance(decoder_spec, str):
            assert (
                decoder_spec in wds.autodecode.imagespecs
                or decoder_spec in wds.autodecode.decoders
            ), (
                "when decoder is specified via a string, then it has to be a a "
                "decoder known to webdataset"
            )
            print(f"Adding decoder {decoder_spec} to decoders.")
            prepared_decoders.append(decoder_spec)
        else:
            raise TypeError(f"{decoder_spec} not a thing for decoders.")

    if decoders:
        # default behavior is setting partial to 'True' in decode
        partial = additional_decoder_kwargs.pop("partial", True)
        # add instantiated decoders to the datapipeline
        datapipeline = datapipeline.decode(
            *prepared_decoders,
            partial=partial,
            handler=error_handler,
            **additional_decoder_kwargs,
        )

    # instantiate all postprocessors
    for i, postro_config in enumerate(postprocessors):
        postpro = instantiate(postro_config)
        if isinstance(postpro, AbstractFilter):
            print(
                f"Adding filter {postpro.__class__.__name__} as postprocessor #{i} "
                f"to the datapipeline"
            )
            datapipeline = datapipeline.select(postpro)
        elif isinstance(postpro, AbstractMapper):
            print(
                f"Adding mapper {postpro.__class__.__name__} as postprocessor #{i} "
                f"to the datapipeline"
            )
            datapipeline = datapipeline.map(postpro, handler=error_handler)
        else:
            raise TypeError(
                "chosen postprocessor should be either a subclass of "
                "sdata.AbstractMapper or one of sdata.AbstractFilter"
                "but is none of both"
            )

    return datapipeline

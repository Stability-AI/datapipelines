# datapipelines


Iterable datapipelines for pytorch training.

The functions `sdata.create_dataset()` and `sdata.create_loader()` provide interfaces for your pytorch training code, where the former returns 
a dataset and the latter a wrapper around a pytorch dataloader.

A dataset as returned by `sdata.create_dataset()` consists of 5 main modules should be defined in a yaml-config:
1. A base [datapipeline](./sdata/datapipeline.py#L306), which reads data as tar files from local fs and assembles them to samples. Each sample comes as a python-dict. 
2. A list of [preprocessors](sdata/dataset.py#L129) which can be either used to transform the entries of a sample or to filter out unsuitable samples. The former kinds are called `mappers`, the latter `filters`. This repository provides a basic set of [mappers](sdata/mappers) and [filters](sdata/filters) which provide basic (not too application specific) data transforms and filters.
3. A list of [decoders](hsdata/dataset.py#L127) whose elements can be either defined as a string matching one of the predefined webdataset [image decoders](https://github.com/webdataset/webdataset/blob/039d74319ae55e5696dcef89829be9671802cf70/webdataset/autodecode.py#L238) decoders or some custom decoder (in the config-style) for handling more specific needs. Note that decoding will be skipped alltogether when setting `decoders=None` (or in config-style yaml `decoders: null`).
4. A list of [postprocessors](sdata/dataset.py#L130) which are used to filter or transform the data after it has been decoded and should again be either `mappers` or `filters`.
5. `error_handler`: A [webdataset-style function](https://github.com/webdataset/webdataset/blob/main/webdataset/handlers.py) for handling any errors which occur in the `datapipeline`, `preprocessors`, `decoders` or `postprocessors`.

A wrapper around a pytorch dataloader, which can be plugged in to your training, is returned by [`sdata.create_loader()`](sdata/dataset.py#L51). You can pass the dataset either as an `IterableDataset` as returned by `sdata.create_dataset()` or via the config which would instantiate this dataset. Apart from the known `batch_size`, `num_workers`, `partial` and `collation_fn` parameteters for pytorch dataloaders, the function can be configured via the following arguments.

1. `batched_transforms` of batched `mappers` and `filters` which transform an entire training batch before being passed to the dataloader defined in the same style than the `preprocessors` and `postprocessors` from above.
2. `loader_kwargs` defining additional keyword arguments for the dataloader (such as `prefetch_factor`, ...)
3. `error_handler`: A [webdataset-style function](https://github.com/webdataset/webdataset/blob/main/webdataset/handlers.py) for handling any errors which occur in the `batched_transforms`.


## Examples 

Here, it is most effective to look at the configs in `examples/configs/` for the following examples. These will show you how this works.

For a simple example, see [`examples/image_simple.py`](examples/image_simple.py), find config [here](examples/configs/example.yaml). 

**NOTE:** You have to add your dataset in tar-form which should follow the [webdataset-format](https://github.com/webdataset/webdataset). To find the parts which have to be adapted, search for comments conaining `USER:` in the respective config. 

## Installation

### Pytorch 2 and later

```bash
python3 -m venv .pt2
source .pt2/bin/activate
pip3 install wheel
pip3 install -r requirements_pt2.txt

```

### Pytorch 1.13 

```bash
python3 -m venv .pt1
source .pt1/bin/activate
pip3 install wheel
pip3 install -r requirements_pt1.txt

```


from typing import Dict

import webdataset as wds
from torchdata.datapipes.iter import IterDataPipe

from .dataset import create_dataset


class DummyIterator(IterDataPipe):
    def __init__(self, sample: Dict):
        super().__init__()
        self.sample = sample

    def __iter__(self):
        while True:
            yield self.sample


class DummyDataPipeline(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(self, datapipe: IterDataPipe):
        super().__init__()
        self.append(datapipe)


def create_dummy_dataset(*args, **kwargs):
    datapipe = create_dataset(*args, **kwargs)

    sample = next(iter(datapipe))
    del datapipe
    iterator = DummyIterator(sample)

    datapipeline = DummyDataPipeline(iterator)

    return datapipeline

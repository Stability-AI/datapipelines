from typing import Dict

from einops import rearrange, repeat, reduce

from .base import AbstractMapper
from ..datapipeline import time_measure


class BatchedEinopsTransform(AbstractMapper):
    transforms = {"rearrange": rearrange, "repeat": repeat, "reduce": reduce}

    def __init__(
        self, pattern: str, key: str, mode: str = "rearrange", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pattern = pattern
        self.key = key

        assert mode in self.transforms, (
            f"mode parameter for {self.__class__.__name__} has to be "
            f"in {list(self.transforms)}"
        )
        self.mode = mode

    @time_measure("BatchedEinopsTransform")
    def __call__(self, sample: Dict) -> Dict:
        if self.skip_this_sample(sample):
            return sample
        target = sample[self.key]

        sample[self.key] = self.transforms[self.mode](target, self.pattern)

        del target
        return sample

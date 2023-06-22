from abc import abstractmethod
from typing import Dict, Union, List, Optional, Callable

from omegaconf import ListConfig, DictConfig

from ..datapipeline import time_measure, make_callable


class AbstractFilter:
    def __init__(
        self,
        exclude_keys: Optional[Union[List[str], ListConfig, str]] = None,
        verbose: bool = False,
    ):
        if not exclude_keys:
            exclude_keys = []

        if isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        self.exclude_keys = exclude_keys

        self.verbose = verbose

    def skip_this_sample(self, sample: Dict) -> bool:
        res = any(map(lambda x: x in sample["__url__"], self.exclude_keys))
        del sample
        return res

    @abstractmethod
    def __call__(self, sample: Dict) -> bool:
        raise NotImplementedError("AbstractFilter should not be called but overwritten")


class LambdaFilter(AbstractFilter):
    def __init__(
        self,
        keys: Union[str, List[str], ListConfig],
        fn: Union[Dict, DictConfig, Callable],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if isinstance(keys, str):
            keys = [keys]

        self.keys = keys

        if isinstance(fn, Union[Dict, DictConfig]):
            fn = make_callable(fn)

        self.fn = fn

    @time_measure("LambdaMapper")
    def __call__(self, sample: Dict) -> bool:
        if self.skip_this_sample(sample):
            del sample
            return True

        let_pass = True
        for key in self.keys:
            let_pass &= self.fn(sample[key])

        del sample
        return let_pass

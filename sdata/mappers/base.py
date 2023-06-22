from abc import abstractmethod
from typing import Dict, Union, Optional, List, Callable
import functools

from omegaconf import ListConfig, DictConfig

from ..datapipeline import make_callable, time_measure


class AbstractMapper(object):
    timeout = None

    def __init__(
        self,
        exclude_keys: Optional[Union[List[str], ListConfig, str]] = None,
        timeout: Optional[float] = None,
        verbose:bool = False,
    ):
        self.timeout = timeout
        if not exclude_keys:
            exclude_keys = []

        self.verbose = verbose

        if isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        self.exclude_keys = exclude_keys

    def skip_this_sample(self, sample: Dict) -> bool:
        res = any(map(lambda x: x in sample["__url__"], self.exclude_keys))
        del sample
        return res

    @abstractmethod
    def __call__(self, sample: Dict) -> Union[Dict, None]:
        raise NotImplementedError("AbstractMapper should not be called but overwritten")



class LambdaMapper(AbstractMapper):
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
    def __call__(self, sample: Dict) -> Dict:
        if self.skip_this_sample(sample):
            return sample

        for key in self.keys:
            sample[key] = self.fn(sample[key])

        return sample

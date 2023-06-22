from typing import Dict, Optional, List, Union, Tuple

from omegaconf import DictConfig

from .base import AbstractFilter
from sdata.datapipeline import time_measure, timeout_wrapper


class SimpleKeyFilter(AbstractFilter):
    def __init__(self, keys: Union[str, List[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(keys, str):
            keys = [keys]

        self.keys = set(keys)

    @timeout_wrapper
    @time_measure("SimpleKeyFilter")
    def __call__(self, sample: Dict) -> bool:
        try:
            if self.skip_this_sample(sample):
                return True
            result = all(map(lambda x: x in sample, self.keys))
            del sample
            return result
        except Exception as e:
            print(f"{e.__class__.__name__} in {self.__class__.__name__}: {e}")
            return False


class SimpleSizeFilter(AbstractFilter):
    def __init__(
        self,
        size: Union[int, Tuple[int, int], List[int]],
        mode: str = "min",
        strict: Optional[Union[bool, Dict]] = None,
        width_key: str = "original_width",
        height_key: str = "original_height",
        *args,
        **kwargs,
    ):
        """
        Simple size filter based on metadata which is already decoded
        :param size: The desired min or max size
        :param mode: either to filter out all above a min size (min) or all below a max size (max)
        :param key: indicates the field in the sample, the field should be a dict
        :param subkeys: list of strings defining subkeys for nested dict, i.e. ['foo','bar'] would result
        in sample[self.key]['foo']['bar'] being the entry in a nested dict, where the size information sits
        :param strict: whether to return True or False when the key is not present
        :param width_key: the width key at the final level in the metadata dict i.e. for key='json',
        subkeys=['foo','bar'] and width_key = 'original_width', the entry sample['json']['foo']['bar']['original_width']
        would be used
        :param height_key: same as above but with width
        """
        super().__init__(*args, **kwargs)
        if isinstance(size, int):
            size = [size, size]

        self.size = size

        if mode == "min":
            self.relation = self.filter_min
        else:
            self.relation = self.filter_max

        self.strict = strict
        if not isinstance(self.strict, (bool, dict, DictConfig)):
            raise TypeError(
                f"strict in {self.__class__.__name__} should be bool or Dict"
            )

        self.height_key = height_key
        self.width_key = width_key

    def filter_min(self, height: int, width: int) -> bool:
        return height >= self.size[0] and width >= self.size[1]

    def filter_max(self, height: int, width: int) -> bool:
        return height <= self.size[0] and width <= self.size[1]

    @timeout_wrapper
    @time_measure("SimpleSizeFilter")
    def __call__(self, sample: Dict) -> bool:
        try:
            if self.skip_this_sample(sample):
                return True
            # get height and width
            original_width = sample[self.width_key]
            original_height = sample[self.height_key]

            result = self.relation(original_height, original_width)
            return result
        except Exception as e:
            if isinstance(self.strict, bool):
                return not self.strict
            elif isinstance(self.strict, (dict, DictConfig)):
                url = sample["__url__"]
                key = [k for k in self.strict if k in url][0]
                result = not self.strict[key]
                return result
            else:
                raise TypeError(
                    f"strict in {self.__class__.__name__} should be bool or Dict"
                )

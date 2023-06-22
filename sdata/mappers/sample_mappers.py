import warnings
from typing import Union, List, Dict
from omegaconf import DictConfig, ListConfig
import numpy as np
import torch
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode

from .base import AbstractMapper
from ..datapipeline import instantiate, time_measure,timeout_wrapper



class Rescaler(AbstractMapper):
    def __init__(
        self,
        key: Union[List[str], ListConfig, str] = "jpg",
        isfloat: bool = True,
        strict: bool = True,
        *args,
        **kwargs,
    ):
        """

        :param key: the key indicating the sample
        :param isfloat: bool indicating whether input is float in [0,1]
        or uint in [0.255]
        """
        # keeping name of first argument to be 'key' for the sake of backwards compatibility
        super().__init__(*args, **kwargs)
        if isinstance(key, str):
            key = [key]
        self.keys = set(key)
        self.isfloat = isfloat
        self.strict = strict
        self.has_warned = [False, False]

    @timeout_wrapper
    @time_measure("Rescaler")
    def __call__(self, sample: Dict) -> Dict:
        """

        :param sample: Dict containing the speficied key, which should be a torch.Tensor or numpy array
        :return:
        """
        if self.skip_this_sample(sample):
            return sample
        if not any(map(lambda x: x in sample, self.keys)):
            if self.strict:
                raise KeyError(
                    f"None of {self.keys} in current sample with keys {list(sample.keys())}"
                )
            else:
                if not self.has_warned[0]:
                    self.has_warned[0] = True
                    warnings.warn(
                        f"None of {self.keys} contained in sample"
                        f"(for sample with keys {list(sample.keys())}). "
                        f"Sample is returned unprocessed since strict mode not enabled"
                    )
                return sample

        matching_keys = set(self.keys.intersection(sample))
        if len(matching_keys) > 1:
            if self.strict:
                raise ValueError(
                    f"more than one matching key of {self.keys} in sample {list(sample.keys())}. This should not be the case"
                )
            else:
                if not self.has_warned[1]:
                    warnings.warn(
                        f"more than one matching key of {self.keys} in sample {list(sample.keys())}."
                        f" But strict mode disabled, so returning sample unchanged"
                    )
                    self.has_warned[1] = True
                return sample

        key = matching_keys.pop()

        if self.isfloat:
            sample[key] = sample[key] * 2 - 1.0
        else:
            sample[key] = sample[key] / 127.5 - 1.0

        return sample


class TorchVisionImageTransforms(AbstractMapper):
    def __init__(
        self,
        transforms: Union[Union[Dict, DictConfig], ListConfig],
        key: str = "jpg",
        strict: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strict = strict
        self.key = key
        chained_transforms = []

        if isinstance(transforms, (DictConfig, Dict)):
            transforms = [transforms]

        for trf in transforms:
            trf = instantiate(trf)
            chained_transforms.append(trf)

        self.transform = TT.Compose(chained_transforms)

    @timeout_wrapper
    @time_measure("TorchVisionImageTransforms")
    def __call__(self, sample: Dict) -> Union[Dict, None]:
        if self.skip_this_sample(sample):
            return sample
        if self.key not in sample:
            if self.strict:
                del sample
                return None
            else:
                return sample
        sample[self.key] = self.transform(sample[self.key])
        return sample



class AddOriginalImageSizeAsTupleAndCropToSquare(AbstractMapper):
    """
    Adds the original image size as params and crops to a square.
    Also adds cropping parameters. Requires that no RandomCrop/CenterCrop has been called before
    """

    def __init__(
        self,
        h_key: str = "original_height",
        w_key: str = "original_width",
        image_key: str = "jpg",
        use_data_key: bool = True,
        data_key: str = "json",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.h_key, self.w_key = h_key, w_key
        self.image_key = image_key
        self.data_key = data_key
        self.use_data_key = use_data_key

    @timeout_wrapper
    @time_measure("AddOriginalImageSizeAsTupleAndCropToSquare")
    def __call__(self, x: Dict) -> Dict:
        if self.skip_this_sample(x):
            return x
        if self.use_data_key:
            h, w = map(lambda y: x["json"][y], (self.h_key, self.w_key))
        else:
            h, w = map(lambda y: x[y], (self.h_key, self.w_key))
        x["original_size_as_tuple"] = torch.tensor([h, w])
        jpg = x[self.image_key]
        if not isinstance(jpg, torch.Tensor) and jpg.shape[0] not in [1, 3]:
            raise ValueError(
                f"{self.__class__.__name__} requires input image to be a torch.Tensor with channels-first"
            )
        # x['jpg'] should be chw tensor  in [-1, 1] at this point
        size = min(jpg.shape[1], jpg.shape[2])
        delta_h = jpg.shape[1] - size
        delta_w = jpg.shape[2] - size
        assert not all(
            [delta_h, delta_w]
        )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
        x[self.image_key] = TT.functional.crop(
            jpg, top=top, left=left, height=size, width=size
        )
        x["crop_coords_top_left"] = torch.tensor([top, left])
        return x

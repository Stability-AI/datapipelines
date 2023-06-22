import os
from typing import List

import torch
from omegaconf import OmegaConf
import numpy as np

from sdata import create_dataset, create_loader

num_it = 100
max_vis = 3

if __name__ == "__main__":
    filedir = os.path.realpath(os.path.dirname(__file__))
    config_path = os.path.join(filedir, "configs", "example.yaml")
    config = OmegaConf.load(config_path)

    # build config
    datapipeline = create_dataset(**config.dataset)

    # build loader
    loader = create_loader(datapipeline, **config.loader)

    print(f"Yielding {num_it} batches")

    for i, batch in enumerate(loader):
        if i >= num_it:
            break

        for key in batch:
            if isinstance(batch[key], (torch.Tensor, np.ndarray)):
                print(key, batch[key].shape)
            elif isinstance(batch[key], (List)):
                print(key)
                print(batch[key][:max_vis])

    print("ciao")

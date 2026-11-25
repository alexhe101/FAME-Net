"""Explicit NPZ data contract. No implicit 8-bit conversion or pairing by order."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_sample(path: str | Path, training: bool = False) -> dict[str, torch.Tensor]:
    keys = ("ms", "pan", "target", "mask") if training else ("ms", "pan")
    with np.load(path, allow_pickle=False) as data:
        sample = {}
        for key in keys:
            if key not in data:
                if key == "mask":
                    continue
                raise ValueError(f"{path}: missing {key}")
            value = np.asarray(data[key])
            if value.ndim != 3 or not np.issubdtype(value.dtype, np.floating):
                raise ValueError(f"{path}: {key} must be a floating CHW array")
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: {key} contains non-finite values")
            if key == "mask" and (value.shape[0] != 2 or value.min() < 0 or value.max() > 1
                                  or not np.allclose(value.sum(axis=0), 1, atol=1e-5)):
                raise ValueError(f"{path}: mask must be complementary HF/LF channels in [0,1]")
            sample[key] = torch.from_numpy(value.astype(np.float32, copy=True))
        return sample


class NPZDataset(Dataset):
    def __init__(self, root: str | Path):
        self.paths = sorted(Path(root).glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No NPZ samples found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return read_sample(self.paths[i], training=True)

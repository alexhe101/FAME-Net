"""Strict weight migration without executing pickled model objects."""
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from .model import FAMENet


def _known_unused(key: str, expected: Mapping[str, Tensor]) -> bool:
    # HIN identity and conv_3 were never used in the released forward.
    parent, _, suffix = key.rpartition(".")
    prefix, _, layer = parent.rpartition(".")
    return (layer in {"identity", "conv_3"} and suffix in {"weight", "bias"}
            and f"{prefix}.norm.weight" in expected
            and f"{prefix}.conv_1.weight" in expected)


def migrate_weights(model: FAMENet, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    expected = model.state_dict()
    out = {}
    for original, value in state.items():
        key = original.removeprefix("module.")
        if key not in expected:
            if _known_unused(key, expected):
                continue
            raise ValueError(f"Unexpected checkpoint key: {original}")
        if key in out or not isinstance(value, Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"Duplicate or incompatible checkpoint tensor: {original}")
        out[key] = value
    missing = expected.keys() - out.keys()
    if missing:
        raise ValueError(f"Missing checkpoint keys: {sorted(missing)}")
    return out


def load_weights(model: FAMENet, path: str | Path) -> None:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, Mapping):
        raise ValueError("Expected a tensor state dictionary")
    config = data.get("config")
    if isinstance(config, Mapping):
        for field in ("bands", "width", "experts", "top_k", "scale", "routing", "head"):
            if field in config and config[field] != model.config[field]:
                raise ValueError(f"Checkpoint {field}={config[field]!r} differs from model {model.config[field]!r}")
    for key in ("net", "state_dict", "model", "params"):
        if key in data and isinstance(data[key], Mapping):
            data = data[key]
            break
    model.load_state_dict(migrate_weights(model, data), strict=True)

"""Paper reconstruction, frequency-mask, and expert-importance loss."""
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import FAMEOutput


class LossTerms(NamedTuple):
    total: Tensor
    reconstruction: Tensor
    mask: Tensor
    load: Tensor


def load_loss(weights: Tensor, eps: float = 1e-8) -> Tensor:
    # Aggregate across images before SCV, rather than penalizing individual
    # sparse rows whose unselected experts must have zero weight.
    importance = weights.float().sum(dim=0)
    return importance.var(unbiased=False) / importance.mean().square().clamp_min(eps)


def fame_loss(result: FAMEOutput, target: Tensor, mask_target: Tensor | None = None,
              *, alpha: float = 0.001, beta: float = 0.1) -> LossTerms:
    if target.shape != result.hrms.shape:
        raise ValueError("HRMS target shape mismatch")
    if alpha < 0 or beta < 0:
        raise ValueError("Loss weights must be nonnegative")
    rec = F.l1_loss(result.hrms.float(), target.float())
    mask = rec.new_zeros(())
    if alpha:
        if mask_target is None or mask_target.shape != result.mask.shape:
            raise ValueError("A two-channel HF/LF mask target is required when alpha > 0")
        mask = F.l1_loss(result.mask.float(), mask_target.float())
    load = sum(load_loss(g) for g in result.gates)
    return LossTerms(rec + alpha * mask + beta * load, rec, mask, load)


def mask_weight(epoch: int, epochs: int, initial: float = 0.001) -> float:
    # Explicit linear approximation of the paper's annealing to zero at 70%.
    return initial * max(0.0, 1.0 - epoch / max(0.7 * epochs, 1.0))

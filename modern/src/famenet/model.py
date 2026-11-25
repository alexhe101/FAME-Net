"""FAME-Net with explicit inputs, outputs, and routing policies."""
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import FeatureEncoder, Refine
from .routing import Dispatch, ExpertMixture, Routing


class FAMEOutput(NamedTuple):
    hrms: Tensor
    mask: Tensor  # B, 2, H, W; channel 0 high, channel 1 low
    gates: tuple[Tensor, Tensor, Tensor]  # low, high, fusion; each B, E


class MaskPredictor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.spatial_mask = nn.Conv2d(channels, 2, 1, bias=False)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor, sample: bool) -> Tensor:
        logits = self.spatial_mask(self.relu(self.conv(x)))
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = logits.float() if logits.dtype != torch.float64 else logits
            if sample:
                mask = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)
            else:
                # Straight-through hard argmax, deterministic at inference.
                soft = logits.softmax(dim=1)
                hard = torch.zeros_like(soft).scatter(1, soft.argmax(1, keepdim=True), 1)
                mask = hard - soft.detach() + soft
        return mask.to(x.dtype)


class MOEInstance(nn.Module):
    def __init__(self, channels: int, experts: int, top_k: int, routing: Routing, dispatch: Dispatch):
        super().__init__()
        self.lf_experts = ExpertMixture(channels, experts, top_k, "low", routing, dispatch)
        self.hf_experts = ExpertMixture(channels, experts, top_k, "high", routing, dispatch)

    def forward(self, pan: Tensor, ms: Tensor, mask: Tensor):
        high, low = mask[:, :1], mask[:, 1:2]
        lf, lg = self.lf_experts(torch.cat((ms * low, pan * low), dim=1))
        hf, hg = self.hf_experts(torch.cat((ms * high, pan * high), dim=1))
        return lf, hf, lg, hg


class FAMENet(nn.Module):
    """Paper configuration: 4 MS bands, width 32, 4 experts, top-2, scale 4.

    Default legacy routing preserves released gate weighting; use paper routing
    for the equations in the paper. The original release has broken dispatch.
    """
    def __init__(
        self, bands: int = 4, width: int = 32, experts: int = 4,
        top_k: int = 2, scale: int = 4, routing: Routing = "legacy",
        dispatch: Dispatch = "sparse",
    ):
        super().__init__()
        if bands < 1 or width < 4 or width % 2 or scale < 1:
            raise ValueError("Require bands >= 1, even width >= 4 and scale >= 1")
        self.bands, self.scale = bands, scale
        self.config = dict(bands=bands, width=width, experts=experts, top_k=top_k,
                           scale=scale, routing=routing, dispatch=dispatch)
        self.msconv = nn.Conv2d(bands, width, 3, padding=1)
        self.pconv = nn.Conv2d(1, width, 3, padding=1)
        self.msencoder = FeatureEncoder(width)
        self.panencoder = FeatureEncoder(width)
        self.maskp = MaskPredictor(width)
        self.moeInstance = MOEInstance(width, experts, top_k, routing, dispatch)
        self.decoder = ExpertMixture(width, experts, top_k, "fusion", routing, dispatch)
        self.refine = Refine(width, bands)

    def _validate(self, ms: Tensor, pan: Tensor) -> None:
        if ms.ndim != 4 or pan.ndim != 4 or ms.shape[1] != self.bands or pan.shape[1] != 1:
            raise ValueError("Expected MS [B,bands,h,w] and PAN [B,1,H,W]")
        if ms.shape[0] != pan.shape[0] or ms.shape[0] == 0:
            raise ValueError("Batch dimensions must agree and be nonempty")
        if tuple(pan.shape[-2:]) != tuple(s * self.scale for s in ms.shape[-2:]):
            raise ValueError("PAN size must equal MS size times scale")
        if min(ms.shape[-2:]) < 1 or pan.shape[-2] * pan.shape[-1] < 2:
            raise ValueError("Need nonempty images and at least two HR pixels")
        if not ms.is_floating_point() or ms.dtype != pan.dtype or ms.device != pan.device:
            raise ValueError("Inputs must share a floating dtype and device")

    def forward(self, ms: Tensor, pan: Tensor, *, sample_mask: bool | None = None) -> FAMEOutput:
        self._validate(ms, pan)
        mhr = F.interpolate(ms, size=pan.shape[-2:], mode="bicubic", align_corners=True)
        msf = self.msencoder(self.msconv(mhr))
        panf = self.panencoder(self.pconv(pan))
        mask = self.maskp(torch.cat((msf, panf), dim=1), self.training if sample_mask is None else sample_mask)
        lf, hf, lg, hg = self.moeInstance(panf, msf, mask)
        dec, dg = self.decoder(torch.cat((msf, hf, panf, lf), dim=1))
        return FAMEOutput(self.refine(dec) + mhr, mask, (lg, hg, dg))

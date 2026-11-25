"""Per-image top-k expert routing and a dense numerical reference."""
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import ConvProce, HinResBlock, InvBlock

Routing = Literal["legacy", "paper"]
Dispatch = Literal["sparse", "dense"]


class GateNetwork(nn.Module):
    def __init__(self, channels: int, num_experts: int, top_k: int, routing: Routing):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts")
        if routing not in ("legacy", "paper"):
            raise ValueError("Unknown routing mode")
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing = routing
        self.fc0 = nn.Linear(channels, num_experts)
        self.fc1 = nn.Linear(channels, num_experts)
        nn.init.zeros_(self.fc1.weight)
        # In the released implementation fc0 affects discrete selection only;
        # it never receives a task gradient. Keep the tensor for weight loading.
        if routing == "legacy":
            self.fc0.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            value = x.to(dtype)
            pooled = value.amax(dim=(-2, -1)) + value.mean(dim=(-2, -1))
            aux = F.softplus(F.linear(pooled, self.fc0.weight.to(dtype), self.fc0.bias.to(dtype)))
            logits = F.linear(pooled, self.fc1.weight.to(dtype), self.fc1.bias.to(dtype))
            if self.routing == "legacy":
                logits = F.leaky_relu(logits, 0.2)
                if self.num_experts > 1:
                    centered = aux - aux.mean(dim=1, keepdim=True)
                    aux = centered / aux.std(dim=1, keepdim=True).clamp_min(1e-6)
                else:
                    aux = torch.zeros_like(aux)
                scores = logits + aux
                weights = logits
            else:
                # Paper equations 10–13: top-k and softmax on the same scores.
                scores = logits + aux
                weights = scores
            indices = scores.topk(self.top_k, dim=1).indices
            selected = torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)
            return weights.masked_fill(~selected, -torch.inf).softmax(dim=1)


class LfInstance(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.process = ConvProce(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.process(x)


class HfInstance(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.process = HinResBlock(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.process(x)


class ExpertMixture(nn.Module):
    def __init__(
        self, channels: int, num_experts: int, top_k: int,
        kind: Literal["low", "high", "fusion"], routing: Routing, dispatch: Dispatch,
    ):
        super().__init__()
        if dispatch not in ("sparse", "dense"):
            raise ValueError("Unknown dispatch mode")
        self.dispatch = dispatch
        self.gate = GateNetwork(channels, num_experts, top_k, routing)
        factory = {"low": LfInstance, "high": HfInstance, "fusion": ConvProce}[kind]
        if kind == "fusion" and routing == "paper":
            factory = lambda c: ConvProce(c, slope=0.0)
        in_channels = channels * (4 if kind == "fusion" else 2)
        self.pre_fuse = nn.Sequential(
            InvBlock(in_channels // 2, slope=1.0 if routing == "legacy" else 0.2),
            nn.Conv2d(in_channels, channels, 1)
        )
        self.expert_networks_d = nn.ModuleList(factory(channels) for _ in range(num_experts))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.pre_fuse(x)
        weights = self.gate(x)
        if self.dispatch == "dense":
            outputs = torch.stack([expert(x) for expert in self.expert_networks_d], dim=1)
            return (outputs * weights.to(outputs.dtype)[..., None, None, None]).sum(dim=1), weights
        out = torch.zeros_like(x)
        # Only evaluate selected batch items. No erroneous all()==0 test.
        # Nonzero incurs a dynamic shape/synchronization on CUDA; dense mode
        # is available for small batches and compiler-friendly benchmarking.
        for i, expert in enumerate(self.expert_networks_d):
            ids = weights[:, i].ne(0).nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                continue
            y = expert(x.index_select(0, ids))
            scale = weights[ids, i].to(y.dtype).view(-1, 1, 1, 1)
            out = out.index_add(0, ids, (y * scale).to(out.dtype))
        return out, weights

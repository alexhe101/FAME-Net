"""Active legacy blocks, with checkpoint names retained."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvProce(nn.Module):
    def __init__(self, channels: int, slope: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu1 = nn.LeakyReLU(slope)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.relu1(self.conv1(x))) + x


class HinResBlock(nn.Module):
    """Half instance normalization. The release's unused identity/conv_3 are omitted."""
    def __init__(self, channels: int, slope: float = 0.2):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("HIN requires positive even channels")
        self.conv_1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu_1 = nn.LeakyReLU(slope)
        self.conv_2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu_2 = nn.LeakyReLU(slope)
        self.norm = nn.InstanceNorm2d(channels // 2, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.relu_1(self.conv_1(x)).chunk(2, dim=1)
        return x + self.relu_2(self.conv_2(torch.cat((self.norm(a), b), dim=1)))


class FeatureEncoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = HinResBlock(channels)
        self.conv2 = HinResBlock(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class InvertibleConv1x1(nn.Module):
    """LU-parameterized forward mixer. No unused logdet or reverse computation."""
    def __init__(self, channels: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        p, lower, upper = torch.linalg.lu(q)
        diag = upper.diagonal()
        self.register_buffer("p", p)
        self.register_buffer("sign_s", diag.sign())
        self.l = nn.Parameter(lower)
        self.u = nn.Parameter(upper.triu(diagonal=1))
        self.log_s = nn.Parameter(diag.abs().log())
        self.register_buffer("l_mask", torch.ones_like(lower).tril(-1), persistent=False)
        self.register_buffer("eye", torch.eye(channels), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # Keep the LU reconstruction in full precision under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.float64 if self.l.dtype == torch.float64 else torch.float32
            lower = self.l.to(dtype) * self.l_mask.to(dtype) + self.eye.to(dtype)
            upper = self.u.to(dtype) * self.l_mask.T.to(dtype)
            upper = upper + torch.diag(self.sign_s.to(dtype) * self.log_s.to(dtype).exp())
            weight = self.p.to(dtype) @ (lower @ upper)
        return F.conv2d(x, weight.to(x.dtype)[..., None, None])


class InvBlock(nn.Module):
    def __init__(self, channels: int, clamp: float = 0.8, slope: float = 0.2):
        super().__init__()
        self.clamp = clamp
        self.F = HinResBlock(channels, slope=slope)
        self.G = HinResBlock(channels, slope=slope)
        self.H = HinResBlock(channels, slope=slope)
        self.invconv = InvertibleConv1x1(2 * channels)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.invconv(x).chunk(2, dim=1)
        y = a + self.F(b)
        s = self.clamp * (2 * self.H(y).sigmoid() - 1)
        return torch.cat((y, b * s.exp() + self.G(y)), dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.process = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.process(x)
        return x + residual * self.conv_du(self.avg_pool(residual))


class Refine(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, channels, 3, padding=1)
        self.process = nn.Sequential(ChannelAttention(channels))
        self.conv_last = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_last(self.process(self.conv_in(x)))

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Lightweight residual block with a stable residual scale."""

    def __init__(self, channels, bias=True, residual_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=bias),
        )
        self.residual_scale = residual_scale

    def forward(self, x):
        return x + self.residual_scale * self.body(x)


class ChannelAttention(nn.Module):
    """Low-cost channel/spectral attention used at low spatial resolution."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(self.pool(x))


class ReconstructionBlock(nn.Module):
    """Residual reconstruction block operating on the 64 x 64 features."""

    def __init__(self, channels, bias=True):
        super().__init__()
        self.body = nn.Sequential(
            ResBlock(channels, bias=bias),
            ResBlock(channels, bias=bias),
            ChannelAttention(channels),
        )

    def forward(self, x):
        return x + 0.1 * self.body(x)


class PixelUnshuffleDown(nn.Module):
    """Lossless spatial rearrangement followed by learned channel compression."""

    def __init__(self, in_channels, out_channels, factor, num_blocks=2, bias=True):
        super().__init__()
        expanded_channels = in_channels * factor * factor
        self.body = nn.Sequential(
            nn.PixelUnshuffle(factor),
            nn.Conv2d(expanded_channels, out_channels, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
            nn.GELU(),
            *[ResBlock(out_channels, bias=bias) for _ in range(num_blocks)],
        )

    def forward(self, x):
        return self.body(x)


class UNetRes(nn.Module):

    def __init__(
        self,
        in_nc=1,
        out_nc=None,
        nc=(32, 64, 128, 256),
        nb=2,
        body_blocks=4,
        act_mode='R',
        bias=True,
    ):
        super().__init__()
        del act_mode  # Kept only for backward-compatible constructor calls.

        if out_nc is None:
            raise ValueError("out_nc must be specified, for example out_nc=20")
        if len(nc) != 4:
            raise ValueError("nc must contain four channel widths")

        # 3840 -> 768. Every 5 x 5 input block is moved into 25 channels.
        self.m_head = PixelUnshuffleDown(
            in_nc, nc[0], factor=5, num_blocks=nb, bias=bias
        )

        # 768 -> 256 -> 128 -> 64. No max pooling is used.
        self.m_down1 = PixelUnshuffleDown(
            nc[0], nc[1], factor=3, num_blocks=nb, bias=bias
        )
        self.m_down2 = PixelUnshuffleDown(
            nc[1], nc[2], factor=2, num_blocks=nb, bias=bias
        )
        self.m_down3 = PixelUnshuffleDown(
            nc[2], nc[3], factor=2, num_blocks=nb, bias=bias
        )

        # Preserve the old attribute name used by some external scripts.
        self.m_down4 = nn.Identity()

        # Most nonlinear reconstruction is performed at only 64 x 64.
        self.m_body = nn.Sequential(
            *[ReconstructionBlock(nc[3], bias=bias) for _ in range(body_blocks)]
        )

        self.m_tail = nn.Sequential(
            nn.Conv2d(nc[3], 128, 3, 1, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(128, out_nc, 1, bias=True),
        )

    @staticmethod
    def _check_input_size(x):
        height, width = x.shape[-2:]
        if height % 60 != 0 or width % 60 != 0:
            raise ValueError(
                "Input height and width must be divisible by 60; "
                f"received {height} x {width}"
            )

    def forward(self, x0):
        self._check_input_size(x0)
        x1 = self.m_head(x0)     # [B,  32, 768, 768]
        x2 = self.m_down1(x1)    # [B,  64, 256, 256]
        x3 = self.m_down2(x2)    # [B, 128, 128, 128]
        x4 = self.m_down3(x3)    # [B, 256,  64,  64]
        x5 = self.m_down4(x4)
        x6 = self.m_body(x5)
        return self.m_tail(x6)   # [B, out_nc, 64, 64]


if __name__ == '__main__':
    net = UNetRes(in_nc=1, out_nc=20).eval()
    x = torch.rand(1, 1, 3840, 3840)
    with torch.no_grad():
        y = net(x)
    print(y.shape)

"""Squeeze-and-Excitation module used in SCOPE."""

from __future__ import annotations

from torch import Tensor, nn


class SqueezeExcitation(nn.Module):
    """Channel-wise feature recalibration.

    Args:
        channels: Number of input feature channels.
        reduction: Channel reduction ratio used in the bottleneck.
        min_channels: Minimum number of hidden channels.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        min_channels: int = 8,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be greater than zero.")

        if reduction <= 0:
            raise ValueError("reduction must be greater than zero.")

        hidden_channels = max(min_channels, channels // reduction)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        channel_weights = self.global_pool(x)
        channel_weights = self.excitation(channel_weights)

        return x * channel_weights

"""SE attention used only in SCOPE's adaptive BiFPN."""
from __future__ import annotations
from torch import Tensor, nn

class SqueezeExcitation(nn.Module):
    """GAP -> FC -> ReLU -> FC -> Sigmoid channel gate."""
    def __init__(self, channels:int, reduction:int=4, min_channels:int=8)->None:
        super().__init__()
        if channels<=0 or reduction<=0: raise ValueError("channels and reduction must be positive")
        hidden=max(min_channels, channels//reduction)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.excitation=nn.Sequential(
            nn.Conv2d(channels,hidden,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden,channels,1),
            nn.Sigmoid(),
        )
    def forward(self,x:Tensor)->Tensor:
        return x*self.excitation(self.pool(x))

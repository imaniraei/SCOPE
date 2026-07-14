"""CBAM used inside SCOPE MBConv blocks."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class ChannelAttention(nn.Module):
    def __init__(self, channels:int, reduction:int=16, min_channels:int=8)->None:
        super().__init__()
        hidden=max(min_channels, channels//reduction)
        self.mlp=nn.Sequential(
            nn.Conv2d(channels,hidden,1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden,channels,1,bias=False),
        )
        self.avg=nn.AdaptiveAvgPool2d(1); self.max=nn.AdaptiveMaxPool2d(1)
    def forward(self,x:Tensor)->Tensor:
        return x*torch.sigmoid(self.mlp(self.avg(x))+self.mlp(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size:int=7)->None:
        super().__init__()
        if kernel_size not in (3,7): raise ValueError("kernel_size must be 3 or 7")
        self.conv=nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
    def forward(self,x:Tensor)->Tensor:
        avg=torch.mean(x,dim=1,keepdim=True)
        mx=torch.amax(x,dim=1,keepdim=True)
        return x*torch.sigmoid(self.conv(torch.cat((avg,mx),dim=1)))

class CBAM(nn.Module):
    def __init__(self,channels:int,reduction:int=16,spatial_kernel_size:int=7)->None:
        super().__init__()
        self.channel=ChannelAttention(channels,reduction)
        self.spatial=SpatialAttention(spatial_kernel_size)
    def forward(self,x:Tensor)->Tensor:
        return self.spatial(self.channel(x))

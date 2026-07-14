"""MBConv with CBAM replacing EfficientNet SE."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from .cbam import CBAM

class ConvNormAct(nn.Sequential):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,stride:int=1,groups:int=1,activate:bool=True)->None:
        layers=[nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,groups=groups,bias=False),nn.BatchNorm2d(out_channels)]
        if activate: layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)

class StochasticDepth(nn.Module):
    def __init__(self,p:float=0.0)->None:
        super().__init__(); self.p=p
    def forward(self,x:Tensor)->Tensor:
        if not self.training or self.p==0: return x
        keep=1-self.p
        shape=(x.shape[0],)+(1,)*(x.ndim-1)
        mask=torch.empty(shape,device=x.device,dtype=x.dtype).bernoulli_(keep)
        return x*mask/keep

class MBConvCBAM(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,expand_ratio:int,kernel_size:int,stride:int,drop_path:float=0.0,cbam_reduction:int=16)->None:
        super().__init__()
        hidden=in_channels*expand_ratio
        self.use_residual=stride==1 and in_channels==out_channels
        self.expand=ConvNormAct(in_channels,hidden,1) if expand_ratio!=1 else nn.Identity()
        self.depthwise=ConvNormAct(hidden,hidden,kernel_size,stride,groups=hidden)
        self.project=ConvNormAct(hidden,out_channels,1,activate=False)
        self.cbam=CBAM(out_channels,cbam_reduction)
        self.drop=StochasticDepth(drop_path)
    def forward(self,x:Tensor)->Tensor:
        residual=x
        x=self.cbam(self.project(self.depthwise(self.expand(x))))
        return residual+self.drop(x) if self.use_residual else x

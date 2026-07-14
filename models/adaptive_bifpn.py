"""Single adaptive BiFPN layer with SE-style attention gates."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .mbconv import ConvNormAct
from .se import SqueezeExcitation

class SeparableConv(nn.Module):
    def __init__(self,channels:int)->None:
        super().__init__()
        self.dw=nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
        self.pw=nn.Conv2d(channels,channels,1,bias=False)
        self.bn=nn.BatchNorm2d(channels)
    def forward(self,x:Tensor)->Tensor:
        return F.silu(self.bn(self.pw(self.dw(x))),inplace=True)

class FastNormalizedFusion(nn.Module):
    def __init__(self,n:int,eps:float=1e-4)->None:
        super().__init__(); self.weights=nn.Parameter(torch.ones(n)); self.eps=eps
    def forward(self,features:Sequence[Tensor])->Tensor:
        w=F.relu(self.weights); w=w/(w.sum()+self.eps)
        out=torch.zeros_like(features[0])
        for wi,fi in zip(w,features): out=out+wi*fi
        return out

class AttentionFusionNode(nn.Module):
    def __init__(self,channels:int,n_inputs:int,se_reduction:int=4)->None:
        super().__init__()
        self.fusion=FastNormalizedFusion(n_inputs)
        self.gate=SqueezeExcitation(channels,se_reduction)
        self.conv=SeparableConv(channels)
    def forward(self,features:Sequence[Tensor])->Tensor:
        return self.conv(self.gate(self.fusion(features)))

class AdaptiveBiFPN(nn.Module):
    def __init__(self,input_channels:tuple[int,int,int],output_channels:int,se_reduction:int=4)->None:
        super().__init__()
        c3,c4,c5=input_channels
        self.p3=ConvNormAct(c3,output_channels,1); self.p4=ConvNormAct(c4,output_channels,1); self.p5=ConvNormAct(c5,output_channels,1)
        self.p6=ConvNormAct(c5,output_channels,3,stride=2); self.p7=ConvNormAct(output_channels,output_channels,3,stride=2)
        self.p6td=AttentionFusionNode(output_channels,2,se_reduction)
        self.p5td=AttentionFusionNode(output_channels,2,se_reduction)
        self.p4td=AttentionFusionNode(output_channels,2,se_reduction)
        self.p3out=AttentionFusionNode(output_channels,2,se_reduction)
        self.p4out=AttentionFusionNode(output_channels,3,se_reduction)
        self.p5out=AttentionFusionNode(output_channels,3,se_reduction)
        self.p6out=AttentionFusionNode(output_channels,3,se_reduction)
        self.p7out=AttentionFusionNode(output_channels,2,se_reduction)
    @staticmethod
    def up(x:Tensor,target:Tensor)->Tensor: return F.interpolate(x,size=target.shape[-2:],mode="nearest")
    @staticmethod
    def down(x:Tensor,target:Tensor)->Tensor: return F.adaptive_max_pool2d(x,target.shape[-2:])
    def forward(self,features:tuple[Tensor,Tensor,Tensor])->list[Tensor]:
        c3,c4,c5=features
        p3,p4,p5=self.p3(c3),self.p4(c4),self.p5(c5)
        p6=self.p6(c5); p7=self.p7(F.silu(p6,inplace=False))
        p6td=self.p6td((p6,self.up(p7,p6)))
        p5td=self.p5td((p5,self.up(p6td,p5)))
        p4td=self.p4td((p4,self.up(p5td,p4)))
        p3o=self.p3out((p3,self.up(p4td,p3)))
        p4o=self.p4out((p4,p4td,self.down(p3o,p4)))
        p5o=self.p5out((p5,p5td,self.down(p4o,p5)))
        p6o=self.p6out((p6,p6td,self.down(p5o,p6)))
        p7o=self.p7out((p7,self.down(p6o,p7)))
        return [p3o,p4o,p5o,p6o,p7o]

"""EfficientNet-style backbone with CBAM-enhanced MBConv."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence
from torch import Tensor, nn
from .mbconv import ConvNormAct, MBConvCBAM

@dataclass(frozen=True)
class StageConfig:
    expand_ratio:int; channels:int; repeats:int; stride:int; kernel_size:int

BASE_STAGES:Sequence[StageConfig]=(
    StageConfig(1,16,1,1,3),StageConfig(6,24,2,2,3),StageConfig(6,40,2,2,5),
    StageConfig(6,80,3,2,3),StageConfig(6,112,3,1,5),StageConfig(6,192,4,2,5),
    StageConfig(6,320,1,1,3),
)
def round_channels(c:int,w:float,divisor:int=8)->int:
    v=c*w; r=max(divisor,int(v+divisor/2)//divisor*divisor)
    return r+divisor if r<0.9*v else r
def round_repeats(r:int,d:float)->int: return int(math.ceil(r*d))

class EfficientNetCBAMBackbone(nn.Module):
    def __init__(self,width_multiplier:float=1.0,depth_multiplier:float=1.0,drop_path_rate:float=0.2)->None:
        super().__init__()
        stem=round_channels(32,width_multiplier)
        self.stem=ConvNormAct(3,stem,3,stride=2)
        total=sum(round_repeats(s.repeats,depth_multiplier) for s in BASE_STAGES)
        stages=[]; stage_channels=[]; in_c=stem; index=0
        for cfg in BASE_STAGES:
            out_c=round_channels(cfg.channels,width_multiplier)
            blocks=[]
            for i in range(round_repeats(cfg.repeats,depth_multiplier)):
                blocks.append(MBConvCBAM(in_c,out_c,cfg.expand_ratio,cfg.kernel_size,cfg.stride if i==0 else 1,drop_path_rate*index/max(1,total-1)))
                in_c=out_c; index+=1
            stages.append(nn.Sequential(*blocks)); stage_channels.append(out_c)
        self.stages=nn.ModuleList(stages)
        self.out_channels=(stage_channels[2],stage_channels[4],stage_channels[6])
    def forward(self,x:Tensor)->tuple[Tensor,Tensor,Tensor]:
        x=self.stem(x); selected=[]
        for i,stage in enumerate(self.stages):
            x=stage(x)
            if i in (2,4,6): selected.append(x)
        return selected[0],selected[1],selected[2]

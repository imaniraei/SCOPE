"""SCOPE classification and box regression heads."""
from __future__ import annotations
import math
from typing import Sequence
from torch import Tensor, nn
from .adaptive_bifpn import SeparableConv

class SharedHead(nn.Module):
    def __init__(self,channels:int,depth:int,anchors_per_location:int,outputs_per_anchor:int,prior_probability:float|None=None)->None:
        super().__init__()
        self.blocks=nn.ModuleList(SeparableConv(channels) for _ in range(depth))
        self.output=nn.Conv2d(channels,anchors_per_location*outputs_per_anchor,3,padding=1)
        if prior_probability is not None:
            nn.init.constant_(self.output.bias,-math.log((1-prior_probability)/prior_probability))
    def forward(self,features:Sequence[Tensor])->list[Tensor]:
        outputs=[]
        for feature in features:
            x=feature
            for block in self.blocks: x=block(x)
            outputs.append(self.output(x))
        return outputs

class ClassificationHead(SharedHead):
    def __init__(self,channels:int,depth:int,anchors_per_location:int,number_of_classes:int)->None:
        super().__init__(channels,depth,anchors_per_location,number_of_classes,0.01)

class BoxRegressionHead(SharedHead):
    def __init__(self,channels:int,depth:int,anchors_per_location:int)->None:
        super().__init__(channels,depth,anchors_per_location,4,None)

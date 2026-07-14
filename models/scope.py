"""Top-level SCOPE architecture."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from torch import Tensor, nn
from .efficientnet import EfficientNetCBAMBackbone
from .adaptive_bifpn import AdaptiveBiFPN
from .heads import ClassificationHead, BoxRegressionHead

@dataclass(frozen=True)
class ScopeScaleConfig:
    input_resolution:int; width_multiplier:float; depth_multiplier:float; bifpn_channels:int; head_depth:int

SCOPE_SCALES:Dict[str,ScopeScaleConfig]={
    "s0":ScopeScaleConfig(512,1.00,1.00,64,3),
    "s1":ScopeScaleConfig(640,1.00,1.10,88,3),
    "s2":ScopeScaleConfig(768,1.10,1.20,112,3),
    "s3":ScopeScaleConfig(896,1.20,1.40,160,4),
    "s4":ScopeScaleConfig(1024,1.40,1.80,224,4),
    "s5":ScopeScaleConfig(1280,1.60,2.20,288,4),
    "s6":ScopeScaleConfig(1280,1.80,2.60,384,5),
    "s7":ScopeScaleConfig(1536,2.00,3.10,384,5),
}

class ScopeDetector(nn.Module):
    """Backbone (MBConv+CBAM) -> one adaptive BiFPN (SE gates) -> heads."""
    def __init__(self,number_of_classes:int,scale:str="s3",anchors_per_location:int=9)->None:
        super().__init__()
        scale=scale.lower()
        if scale not in SCOPE_SCALES: raise ValueError(f"unknown scale: {scale}")
        cfg=SCOPE_SCALES[scale]
        self.scale=scale; self.input_resolution=cfg.input_resolution
        self.number_of_classes=number_of_classes
        self.backbone=EfficientNetCBAMBackbone(cfg.width_multiplier,cfg.depth_multiplier)
        self.adaptive_bifpn=AdaptiveBiFPN(self.backbone.out_channels,cfg.bifpn_channels)
        self.classification_head=ClassificationHead(cfg.bifpn_channels,cfg.head_depth,anchors_per_location,number_of_classes)
        self.box_regression_head=BoxRegressionHead(cfg.bifpn_channels,cfg.head_depth,anchors_per_location)
    def forward(self,images:Tensor)->dict[str,list[Tensor]]:
        features=self.adaptive_bifpn(self.backbone(images))
        return {"features":features,"class_logits":self.classification_head(features),"box_regression":self.box_regression_head(features)}

def build_scope(number_of_classes:int,scale:str="s3",anchors_per_location:int=9)->ScopeDetector:
    return ScopeDetector(number_of_classes,scale,anchors_per_location)

from __future__ import annotations
import math
from typing import Sequence, Tuple
import torch
from torch import Tensor

class AnchorGenerator:
    def __init__(self, pyramid_levels: Sequence[int]=(3,4,5,6,7),
                 aspect_ratios: Sequence[float]=(0.5,1.0,2.0),
                 scales: Sequence[float]=(1.0,2**(1/3),2**(2/3)),
                 anchor_scale: float=4.0) -> None:
        self.levels=tuple(pyramid_levels); self.ratios=tuple(aspect_ratios)
        self.scales=tuple(scales); self.anchor_scale=float(anchor_scale)

    @property
    def anchors_per_location(self) -> int:
        return len(self.ratios)*len(self.scales)

    def __call__(self, feature_shapes: Sequence[Tuple[int,int]],
                 image_size: Tuple[int,int], device: torch.device,
                 dtype: torch.dtype) -> Tensor:
        if len(feature_shapes)!=len(self.levels):
            raise ValueError("feature_shapes must match pyramid levels")
        image_h,image_w=image_size; output=[]
        for level,(h,w) in zip(self.levels,feature_shapes):
            sy=image_h/float(h); sx=image_w/float(w)
            xs=(torch.arange(w,device=device,dtype=dtype)+0.5)*sx
            ys=(torch.arange(h,device=device,dtype=dtype)+0.5)*sy
            gy,gx=torch.meshgrid(ys,xs,indexing="ij")
            centers=torch.stack((gx,gy,gx,gy),dim=-1).reshape(-1,4)
            base=self.anchor_scale*(2**level); level_anchors=[]
            for scale in self.scales:
                for ratio in self.ratios:
                    area=(base*scale)**2
                    aw=math.sqrt(area/ratio); ah=aw*ratio
                    offset=torch.tensor([-aw/2,-ah/2,aw/2,ah/2],device=device,dtype=dtype)
                    level_anchors.append(centers+offset)
            output.append(torch.cat(level_anchors,dim=0))
        return torch.cat(output,dim=0)

def encode_boxes(boxes: Tensor, anchors: Tensor) -> Tensor:
    asz=(anchors[:,2:]-anchors[:,:2]).clamp(min=1e-6)
    actr=anchors[:,:2]+0.5*asz
    bsz=(boxes[:,2:]-boxes[:,:2]).clamp(min=1e-6)
    bctr=boxes[:,:2]+0.5*bsz
    return torch.cat(((bctr-actr)/asz, torch.log(bsz/asz)),dim=1)

def decode_boxes(deltas: Tensor, anchors: Tensor) -> Tensor:
    asz=(anchors[:,2:]-anchors[:,:2]).clamp(min=1e-6)
    actr=anchors[:,:2]+0.5*asz
    ctr=deltas[:,:2]*asz+actr
    size=torch.exp(deltas[:,2:].clamp(max=4.0))*asz
    return torch.cat((ctr-0.5*size,ctr+0.5*size),dim=1)

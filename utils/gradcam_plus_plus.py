from __future__ import annotations
from typing import Callable, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class GradCAMPlusPlus:
    def __init__(self,model: nn.Module,target_layer: nn.Module,
                 score_function: Callable[[object],Tensor])->None:
        self.model=model; self.target_layer=target_layer
        self.score_function=score_function; self.activations: Optional[Tensor]=None
        self.gradients: Optional[Tensor]=None
        self.fwd=target_layer.register_forward_hook(self._save_activations)
        self.bwd=target_layer.register_full_backward_hook(self._save_gradients)
    def _save_activations(self,module,inputs,output)->None:
        self.activations=output
    def _save_gradients(self,module,grad_input,grad_output)->None:
        self.gradients=grad_output[0]
    def remove_hooks(self)->None:
        self.fwd.remove(); self.bwd.remove()
    def __call__(self,images: Tensor,normalize: bool=True)->Tensor:
        self.model.zero_grad(set_to_none=True)
        outputs=self.model(images); scores=self.score_function(outputs)
        scores.sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM++ hooks captured no tensors")
        a=self.activations; g=self.gradients
        g2=g.pow(2); g3=g.pow(3)
        denom=2*g2+(a*g3).sum(dim=(2,3),keepdim=True)
        denom=torch.where(denom!=0,denom,torch.ones_like(denom))
        alpha=g2/denom
        weights=(alpha*F.relu(g)).sum(dim=(2,3),keepdim=True)
        heatmap=F.relu((weights*a).sum(dim=1,keepdim=True))
        heatmap=F.interpolate(heatmap,size=images.shape[-2:],mode="bilinear",align_corners=False)
        if normalize:
            b=heatmap.shape[0]; flat=heatmap.view(b,-1)
            mn=flat.min(dim=1)[0].view(b,1,1,1)
            mx=flat.max(dim=1)[0].view(b,1,1,1)
            heatmap=(heatmap-mn)/(mx-mn+1e-8)
        return heatmap.detach()

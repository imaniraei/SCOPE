from __future__ import annotations
from typing import Dict, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import box_iou, sigmoid_focal_loss
from .anchors import encode_boxes

class DetectionLoss(nn.Module):
    def __init__(self, number_of_classes: int, positive_iou_threshold: float=0.5,
                 negative_iou_threshold: float=0.4, focal_alpha: float=0.25,
                 focal_gamma: float=2.0, smooth_l1_beta: float=1/9,
                 box_loss_weight: float=1.0) -> None:
        super().__init__()
        self.num_classes=number_of_classes; self.pos_iou=positive_iou_threshold
        self.neg_iou=negative_iou_threshold; self.alpha=focal_alpha
        self.gamma=focal_gamma; self.beta=smooth_l1_beta
        self.box_weight=box_loss_weight

    def forward(self, class_logits: Tensor, box_regression: Tensor,
                anchors: Tensor, targets: Sequence[Dict[str,Tensor]]) -> Dict[str,Tensor]:
        cls_losses=[]; box_losses=[]
        for logits,deltas,target in zip(class_logits,box_regression,targets):
            gt_boxes=target["boxes"].to(anchors)
            gt_labels=target["labels"].to(device=anchors.device,dtype=torch.long)
            cls_target=torch.zeros_like(logits)
            if gt_boxes.numel()==0:
                cls_losses.append(sigmoid_focal_loss(
                    logits,cls_target,alpha=self.alpha,gamma=self.gamma,reduction="sum"
                )/max(1,anchors.shape[0]))
                box_losses.append(deltas.sum()*0.0); continue
            ious=box_iou(anchors,gt_boxes)
            best_iou,matched=ious.max(dim=1)
            positive=best_iou>=self.pos_iou; negative=best_iou<self.neg_iou
            valid=positive|negative
            matched_labels=gt_labels[matched].clamp(0,self.num_classes-1)
            cls_target[positive,matched_labels[positive]]=1.0
            npos=positive.sum().clamp(min=1).float()
            cls_losses.append(sigmoid_focal_loss(
                logits[valid],cls_target[valid],alpha=self.alpha,gamma=self.gamma,reduction="sum"
            )/npos)
            if positive.any():
                reg_target=encode_boxes(gt_boxes[matched[positive]],anchors[positive])
                box_loss=F.smooth_l1_loss(
                    deltas[positive],reg_target,beta=self.beta,reduction="sum"
                )/npos
            else:
                box_loss=deltas.sum()*0.0
            box_losses.append(box_loss)
        cls=torch.stack(cls_losses).mean()
        box=torch.stack(box_losses).mean()
        total=cls+self.box_weight*box
        return {"classification_loss":cls,"box_regression_loss":box,"total_loss":total}

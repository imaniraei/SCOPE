from __future__ import annotations
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

def build_learning_rate_scheduler(optimizer: Optimizer,
                                  milestones: tuple[int,int]=(50,90),
                                  gamma: float=0.1) -> MultiStepLR:
    """0.01 -> 0.001 at epoch 50 -> 0.0001 at epoch 90."""
    if len(milestones)!=2 or milestones[0]>=milestones[1]:
        raise ValueError("milestones must be two increasing epochs")
    return MultiStepLR(optimizer,milestones=list(milestones),gamma=gamma)

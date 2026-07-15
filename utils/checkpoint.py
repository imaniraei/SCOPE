from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch import nn
from torch.optim import Optimizer

def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optimizer: Optional[Optimizer]=None,
                    scheduler: Optional[Any]=None, scaler: Optional[Any]=None,
                    best_metric: Optional[float]=None,
                    metadata: Optional[Dict[str,Any]]=None) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    state={"epoch":int(epoch),"model_state_dict":model.state_dict(),
           "best_metric":best_metric,"metadata":metadata or {}}
    if optimizer is not None: state["optimizer_state_dict"]=optimizer.state_dict()
    if scheduler is not None: state["scheduler_state_dict"]=scheduler.state_dict()
    if scaler is not None: state["scaler_state_dict"]=scaler.state_dict()
    torch.save(state,p)

def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[Optimizer]=None,
                    scheduler: Optional[Any]=None, scaler: Optional[Any]=None,
                    map_location: str|torch.device="cpu",
                    strict: bool=True) -> Dict[str,Any]:
    checkpoint=torch.load(path,map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"],strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint

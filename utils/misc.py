from __future__ import annotations
import random,time
from dataclasses import dataclass
import numpy as np
import torch

def seed_everything(seed: int=42)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def select_device(prefer_cuda: bool=True)->torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")

@dataclass
class AverageMeter:
    value: float=0.0; average: float=0.0; total: float=0.0; count: int=0
    def reset(self)->None:
        self.value=self.average=self.total=0.0; self.count=0
    def update(self,value: float,number: int=1)->None:
        self.value=float(value); self.total+=float(value)*int(number)
        self.count+=int(number); self.average=self.total/max(self.count,1)

class Timer:
    def __enter__(self):
        self.start=time.perf_counter(); return self
    def __exit__(self,*_):
        self.elapsed_seconds=time.perf_counter()-self.start

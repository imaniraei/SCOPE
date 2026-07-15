from __future__ import annotations
import time
from typing import Dict
import torch
from torch import Tensor, nn

def count_parameters(model: nn.Module) -> Dict[str,float]:
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters":float(total),
            "trainable_parameters":float(trainable),
            "total_parameters_millions":total/1e6,
            "trainable_parameters_millions":trainable/1e6}

def estimate_flops(model: nn.Module, input_tensor: Tensor) -> Dict[str,float]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as e:
        raise ImportError("Install fvcore with `pip install fvcore`.") from e
    was_training=model.training; model.eval()
    with torch.no_grad():
        total=float(FlopCountAnalysis(model,input_tensor).total())
    model.train(was_training)
    return {"flops":total,"flops_billions":total/1e9}

def measure_inference_latency(model: nn.Module, input_tensor: Tensor,
                              warmup_iterations: int=20,
                              measured_iterations: int=100) -> Dict[str,float]:
    if measured_iterations<=0: raise ValueError("measured_iterations must be positive")
    device=input_tensor.device; use_cuda=device.type=="cuda"
    was_training=model.training; model.eval()
    times=[]
    with torch.inference_mode():
        for _ in range(warmup_iterations): model(input_tensor)
        if use_cuda: torch.cuda.synchronize(device)
        for _ in range(measured_iterations):
            if use_cuda: torch.cuda.synchronize(device)
            start=time.perf_counter(); model(input_tensor)
            if use_cuda: torch.cuda.synchronize(device)
            times.append(time.perf_counter()-start)
    model.train(was_training)
    values=torch.tensor(times,dtype=torch.float64)
    mean=float(values.mean()); std=float(values.std(unbiased=False))
    return {"latency_seconds":mean,"latency_milliseconds":mean*1000,
            "latency_std_seconds":std,"latency_std_milliseconds":std*1000,
            "frames_per_second":input_tensor.shape[0]/mean if mean>0 else float("inf")}

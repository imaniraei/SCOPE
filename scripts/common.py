from __future__ import annotations
from typing import Dict, List, Sequence
import torch
from torch import Tensor

def flatten_prediction_outputs(outputs: Sequence[Tensor], outputs_per_anchor: int) -> Tensor:
    flattened = []
    for output in outputs:
        batch, channels, height, width = output.shape
        if channels % outputs_per_anchor != 0:
            raise ValueError("Invalid head output shape.")
        anchors = channels // outputs_per_anchor
        output = output.view(batch, anchors, outputs_per_anchor, height, width)
        flattened.append(output.permute(0, 3, 4, 1, 2).reshape(batch, -1, outputs_per_anchor))
    return torch.cat(flattened, dim=1)

def move_targets_to_device(targets: Sequence[Dict[str, Tensor]], device: torch.device) -> List[Dict[str, Tensor]]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]

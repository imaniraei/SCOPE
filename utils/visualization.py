from __future__ import annotations
from pathlib import Path
from typing import Sequence
from PIL import Image,ImageDraw,ImageFont
from torch import Tensor

def draw_detections(image: Image.Image, boxes: Tensor, scores: Tensor,
                    labels: Tensor, class_names: Sequence[str],
                    score_threshold: float=0.3,
                    output_path: str|None=None)->Image.Image:
    result=image.copy(); draw=ImageDraw.Draw(result); font=ImageFont.load_default()
    for box,score,label in zip(boxes,scores,labels):
        confidence=float(score)
        if confidence<score_threshold: continue
        idx=int(label); name=class_names[idx] if idx<len(class_names) else str(idx)
        x1,y1,x2,y2=[float(v) for v in box]; text=f"{name}: {confidence:.2f}"
        draw.rectangle((x1,y1,x2,y2),outline="red",width=3)
        tb=draw.textbbox((x1,y1),text,font=font)
        draw.rectangle(tb,fill="red"); draw.text((x1,y1),text,fill="white",font=font)
    if output_path:
        p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True); result.save(p)
    return result

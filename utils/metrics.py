from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCODetectionEvaluator:
    """Compute mAP (0.50:0.95), mAP@0.50 and mAP@0.75."""
    def __init__(self, annotation_file: str) -> None:
        self.annotation_file=str(annotation_file)
        self.ground_truth=COCO(self.annotation_file)

    def evaluate(self, predictions: List[Dict],
                 output_json: str|None=None) -> Dict[str,float]:
        if not predictions:
            return {"mAP":0.0,"mAP_50":0.0,"mAP_75":0.0}
        if output_json:
            p=Path(output_json); p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(predictions),encoding="utf-8")
        detections=self.ground_truth.loadRes(predictions)
        evaluator=COCOeval(self.ground_truth,detections,iouType="bbox")
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        return {"mAP":float(evaluator.stats[0]),
                "mAP_50":float(evaluator.stats[1]),
                "mAP_75":float(evaluator.stats[2])}

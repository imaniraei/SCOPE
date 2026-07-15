from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, List
import torch
from torchvision.ops import batched_nms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import build_dataloader
from models import build_scope
from utils import AnchorGenerator, COCODetectionEvaluator, decode_boxes, load_checkpoint, select_device
from scripts.common import flatten_prediction_outputs

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SCOPE.")
    p.add_argument("--dataset", required=True, choices=("udacity","kitti","bdd100k","nuscenes"))
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--score-threshold", type=float, default=0.001)
    p.add_argument("--nms-threshold", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=300)
    p.add_argument("--output-json", default="outputs/evaluation/predictions.json")
    return p.parse_args()

def main():
    args = parse_args()
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint.get("metadata", {})
    scale = metadata.get("scale", "s3")
    num_classes = int(metadata.get("number_of_classes", 1))
    image_size = int(metadata.get("image_size", 896))

    model = build_scope(num_classes, scale).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    loader = build_dataloader(
        dataset_name=args.dataset, images_directory=args.images,
        annotation_file=args.annotations, image_size=image_size,
        batch_size=args.batch_size, training=False, number_of_workers=args.workers,
    )
    anchor_generator = AnchorGenerator()
    predictions: List[Dict] = []

    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="evaluation"):
            images = images.to(device)
            outputs = model(images)
            class_logits = flatten_prediction_outputs(outputs["class_logits"], num_classes)
            box_regression = flatten_prediction_outputs(outputs["box_regression"], 4)
            feature_shapes = [tuple(feature.shape[-2:]) for feature in outputs["features"]]
            anchors = anchor_generator(feature_shapes, tuple(images.shape[-2:]), device, images.dtype)
            probabilities = torch.sigmoid(class_logits)

            for probs, deltas, target in zip(probabilities, box_regression, targets):
                scores, labels = probs.max(dim=1)
                keep = scores >= args.score_threshold
                if not keep.any():
                    continue
                boxes = decode_boxes(deltas[keep], anchors[keep])
                scores, labels = scores[keep], labels[keep]
                boxes[:, 0::2].clamp_(0, image_size)
                boxes[:, 1::2].clamp_(0, image_size)
                keep_nms = batched_nms(boxes, scores, labels, args.nms_threshold)[:args.max_detections]
                boxes, scores, labels = boxes[keep_nms].cpu(), scores[keep_nms].cpu(), labels[keep_nms].cpu()

                original_h, original_w = target["original_size"].tolist()
                boxes[:, 0::2] *= original_w / float(image_size)
                boxes[:, 1::2] *= original_h / float(image_size)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                image_id = int(target["image_id"])

                for box, score, label in zip(boxes, scores, labels):
                    predictions.append({
                        "image_id": image_id,
                        "category_id": loader.dataset.get_coco_category_id(int(label)),
                        "bbox": [float(v) for v in box],
                        "score": float(score),
                    })

    metrics = COCODetectionEvaluator(args.annotations).evaluate(predictions, args.output_json)
    print(f"mAP={metrics['mAP']:.4f} | mAP@0.5={metrics['mAP_50']:.4f} | mAP@0.75={metrics['mAP_75']:.4f}")

if __name__ == "__main__":
    main()

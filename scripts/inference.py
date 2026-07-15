from __future__ import annotations
import argparse, sys
from pathlib import Path
from PIL import Image
import torch
from torchvision.ops import batched_nms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_scope
from utils import AnchorGenerator, decode_boxes, draw_detections, load_checkpoint, select_device
from scripts.common import flatten_prediction_outputs

def parse_args():
    p = argparse.ArgumentParser(description="Run SCOPE inference.")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/inference/result.jpg")
    p.add_argument("--score-threshold", type=float, default=0.30)
    p.add_argument("--nms-threshold", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=300)
    return p.parse_args()

def main():
    args = parse_args()
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint.get("metadata", {})
    scale = metadata.get("scale", "s3")
    num_classes = int(metadata.get("number_of_classes", 1))
    class_names = metadata.get("class_names", [str(i) for i in range(num_classes)])
    image_size = int(metadata.get("image_size", 896))

    model = build_scope(num_classes, scale).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    original = Image.open(args.image).convert("RGB")
    original_w, original_h = original.size
    resized = original.resize((image_size, image_size))
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        logits = flatten_prediction_outputs(outputs["class_logits"], num_classes)[0]
        deltas = flatten_prediction_outputs(outputs["box_regression"], 4)[0]
        feature_shapes = [tuple(feature.shape[-2:]) for feature in outputs["features"]]
        anchors = AnchorGenerator()(feature_shapes, (image_size, image_size), device, tensor.dtype)
        scores, labels = torch.sigmoid(logits).max(dim=1)
        keep = scores >= args.score_threshold
        boxes = decode_boxes(deltas[keep], anchors[keep])
        scores, labels = scores[keep], labels[keep]
        boxes[:, 0::2].clamp_(0, image_size)
        boxes[:, 1::2].clamp_(0, image_size)
        keep_nms = batched_nms(boxes, scores, labels, args.nms_threshold)[:args.max_detections]
        boxes, scores, labels = boxes[keep_nms].cpu(), scores[keep_nms].cpu(), labels[keep_nms].cpu()

    boxes[:, 0::2] *= original_w / float(image_size)
    boxes[:, 1::2] *= original_h / float(image_size)
    draw_detections(original, boxes, scores, labels, class_names,
                    score_threshold=args.score_threshold, output_path=args.output)
    print(f"Saved inference result to {args.output}")

if __name__ == "__main__":
    main()

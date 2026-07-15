from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_scope
from utils import GradCAMPlusPlus, load_checkpoint, select_device

def main():
    p = argparse.ArgumentParser(description="Generate SCOPE Grad-CAM++.")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/gradcam_plus_plus/heatmap.png")
    args = p.parse_args()

    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint.get("metadata", {})
    num_classes = int(metadata.get("number_of_classes", 1))
    scale = metadata.get("scale", "s3")
    image_size = int(metadata.get("image_size", 896))

    model = build_scope(num_classes, scale).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    resized = image.resize((image_size, image_size))
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    target_layer = model.adaptive_bifpn.p3_output.output_conv.pointwise

    def score_function(outputs):
        level_scores = [level.sigmoid().amax(dim=(1,2,3)) for level in outputs["class_logits"]]
        return torch.stack(level_scores, dim=1).amax(dim=1)

    gradcam = GradCAMPlusPlus(model, target_layer, score_function)
    heatmap = gradcam(tensor)[0, 0].cpu().numpy()
    gradcam.remove_hooks()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,8))
    plt.imshow(resized)
    plt.imshow(heatmap, alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved Grad-CAM++ visualization to {path}")

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import SCOPE_SCALES, build_scope
from utils import count_parameters, estimate_flops, measure_inference_latency, select_device

def main():
    p = argparse.ArgumentParser(description="Benchmark SCOPE efficiency.")
    p.add_argument("--scale", default="s3", choices=tuple(sorted(SCOPE_SCALES)))
    p.add_argument("--number-of-classes", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output", default="outputs/benchmark/scope_metrics.json")
    args = p.parse_args()

    device = select_device()
    model = build_scope(args.number_of_classes, args.scale).to(device)
    size = SCOPE_SCALES[args.scale].input_resolution
    sample = torch.randn(args.batch_size, 3, size, size, device=device)

    results = {
        "scale": args.scale,
        "input_resolution": size,
        **count_parameters(model),
        **estimate_flops(model, sample),
        **measure_inference_latency(model, sample),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

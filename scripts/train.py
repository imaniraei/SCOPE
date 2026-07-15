from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import build_train_val_loaders
from models import SCOPE_SCALES, build_scope
from utils import (
    AnchorGenerator, AverageMeter, CSVLogger, JSONLogger, DetectionLoss,
    build_learning_rate_scheduler, load_checkpoint, save_checkpoint,
    seed_everything, select_device,
)
from scripts.common import flatten_prediction_outputs, move_targets_to_device

def parse_args():
    p = argparse.ArgumentParser(description="Train SCOPE.")
    p.add_argument("--dataset", required=True, choices=("udacity","kitti","bdd100k","nuscenes"))
    p.add_argument("--train-images", required=True)
    p.add_argument("--train-annotations", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--val-annotations", required=True)
    p.add_argument("--scale", default="s3", choices=tuple(sorted(SCOPE_SCALES)))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=4e-5)
    p.add_argument("--output-directory", default="outputs/scope_s3")
    p.add_argument("--resume")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-amp", action="store_true")
    return p.parse_args()

def run_epoch(model, loader, criterion, anchor_generator, device, optimizer=None, scaler=None, use_amp=True):
    training = optimizer is not None
    model.train(training)
    cls_meter, box_meter, total_meter = AverageMeter(), AverageMeter(), AverageMeter()

    for images, targets in tqdm(loader, desc="train" if training else "validation", leave=False):
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(enabled=use_amp):
                outputs = model(images)
                class_logits = flatten_prediction_outputs(outputs["class_logits"], model.number_of_classes)
                box_regression = flatten_prediction_outputs(outputs["box_regression"], 4)
                feature_shapes = [tuple(feature.shape[-2:]) for feature in outputs["features"]]
                anchors = anchor_generator(feature_shapes, tuple(images.shape[-2:]), images.device, images.dtype)
                losses = criterion(class_logits, box_regression, anchors, targets)
                total_loss = losses["total_loss"]

            if training:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.shape[0]
        cls_meter.update(float(losses["classification_loss"].detach()), batch_size)
        box_meter.update(float(losses["box_regression_loss"].detach()), batch_size)
        total_meter.update(float(total_loss.detach()), batch_size)

    return {
        "classification_loss": cls_meter.average,
        "box_regression_loss": box_meter.average,
        "total_loss": total_meter.average,
    }

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = SCOPE_SCALES[args.scale].input_resolution
    train_loader, val_loader = build_train_val_loaders(
        dataset_name=args.dataset,
        train_images_directory=args.train_images,
        train_annotation_file=args.train_annotations,
        val_images_directory=args.val_images,
        val_annotation_file=args.val_annotations,
        image_size=image_size,
        batch_size=args.batch_size,
        number_of_workers=args.workers,
        seed=args.seed,
    )

    num_classes = train_loader.dataset.number_of_classes
    model = build_scope(num_classes, args.scale).to(device)
    anchors = AnchorGenerator()
    criterion = DetectionLoss(num_classes)
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    scheduler = build_learning_rate_scheduler(optimizer, milestones=(50, 90), gamma=0.1)
    use_amp = not args.disable_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(state["epoch"]) + 1
        if state.get("best_metric") is not None:
            best_val = float(state["best_metric"])

    csv_logger = CSVLogger(str(output_dir / "training_log.csv"))
    json_logger = JSONLogger(str(output_dir / "training_log.json"))
    metadata = {
        "dataset": args.dataset, "scale": args.scale, "number_of_classes": num_classes,
        "class_names": train_loader.dataset.class_names, "image_size": image_size,
        "epochs": args.epochs, "batch_size": args.batch_size,
    }

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, anchors, device, optimizer, scaler, use_amp)
        val_metrics = run_epoch(model, val_loader, criterion, anchors, device, None, scaler, use_amp)
        scheduler.step()

        record = {
            "epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_total_loss": train_metrics["total_loss"],
            "validation_total_loss": val_metrics["total_loss"],
        }
        csv_logger.log(record)
        json_logger.log(record)

        save_checkpoint(str(output_dir / "last.pth"), epoch, model, optimizer, scheduler,
                        scaler, best_val, metadata)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                str(output_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pth"),
                epoch, model, optimizer, scheduler, scaler, best_val, metadata
            )

        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            save_checkpoint(str(output_dir / "best.pth"), epoch, model, optimizer,
                            scheduler, scaler, best_val, metadata)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train={train_metrics['total_loss']:.4f} | "
            f"val={val_metrics['total_loss']:.4f} | best={best_val:.4f}"
        )

if __name__ == "__main__":
    main()

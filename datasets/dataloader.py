"""DataLoader builders for SCOPE datasets."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

import torch
from torch.utils.data import DataLoader

from .bdd100k import BDD100KDataset
from .coco_dataset import COCODetectionDataset
from .kitti import KITTIDataset
from .nuscenes import NuScenesDataset
from .udacity import UdacityDataset
from .utils import detection_collate_fn, seed_worker


DATASET_REGISTRY: Dict[str, Type[COCODetectionDataset]] = {
    "udacity": UdacityDataset,
    "kitti": KITTIDataset,
    "bdd100k": BDD100KDataset,
    "nuscenes": NuScenesDataset,
}


def resolve_dataset_class(
    dataset_name: str,
) -> Type[COCODetectionDataset]:
    normalized_name = dataset_name.lower()

    if normalized_name not in DATASET_REGISTRY:
        valid_names = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            "Unknown dataset '{}'. Valid datasets: {}.".format(
                dataset_name,
                valid_names,
            )
        )

    return DATASET_REGISTRY[normalized_name]


def build_dataloader(
    dataset_name: str,
    images_directory: str,
    annotation_file: str,
    image_size: int,
    batch_size: int,
    training: bool,
    number_of_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build one train, validation or test DataLoader."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if number_of_workers < 0:
        raise ValueError("number_of_workers cannot be negative.")

    dataset_class = resolve_dataset_class(dataset_name)

    dataset = dataset_class(
        images_directory=images_directory,
        annotation_file=annotation_file,
        image_size=image_size,
        training=training,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if training else False,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=number_of_workers > 0,
    )


def build_train_val_loaders(
    dataset_name: str,
    train_images_directory: str,
    train_annotation_file: str,
    val_images_directory: str,
    val_annotation_file: str,
    image_size: int,
    batch_size: int,
    number_of_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build matching training and validation DataLoaders."""

    train_loader = build_dataloader(
        dataset_name=dataset_name,
        images_directory=train_images_directory,
        annotation_file=train_annotation_file,
        image_size=image_size,
        batch_size=batch_size,
        training=True,
        number_of_workers=number_of_workers,
        seed=seed,
    )

    val_loader = build_dataloader(
        dataset_name=dataset_name,
        images_directory=val_images_directory,
        annotation_file=val_annotation_file,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        number_of_workers=number_of_workers,
        seed=seed,
    )

    if (
        train_loader.dataset.class_names
        != val_loader.dataset.class_names
    ):
        raise ValueError(
            "Training and validation category definitions do not match."
        )

    return train_loader, val_loader

"""Generic COCO JSON detection dataset used by all SCOPE datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import Target, build_transforms


class COCODetectionDataset(Dataset):
    """Load object-detection data stored in COCO JSON format.

    Args:
        images_directory: Directory containing image files.
        annotation_file: COCO JSON annotation file.
        image_size: Square input resolution used by SCOPE.
        training: Whether to enable training augmentation.
        transforms: Optional custom joint image/target transform.
        dataset_name: Human-readable dataset name.
        include_crowd: Whether to retain annotations marked as crowd.
    """

    def __init__(
        self,
        images_directory: str,
        annotation_file: str,
        image_size: int,
        training: bool = False,
        transforms: Optional[Callable] = None,
        dataset_name: str = "COCO",
        include_crowd: bool = False,
    ) -> None:
        super().__init__()

        self.images_directory = Path(images_directory)
        self.annotation_file = Path(annotation_file)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.dataset_name = dataset_name
        self.include_crowd = include_crowd

        if not self.images_directory.is_dir():
            raise FileNotFoundError(
                "Image directory does not exist: {}".format(
                    self.images_directory
                )
            )
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                "Annotation file does not exist: {}".format(
                    self.annotation_file
                )
            )
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            coco_data = json.load(handle)

        required_keys = {"images", "annotations", "categories"}
        missing_keys = required_keys.difference(coco_data)
        if missing_keys:
            raise ValueError(
                "COCO annotation file is missing keys: {}".format(
                    sorted(missing_keys)
                )
            )

        self.images: List[Dict] = sorted(
            coco_data["images"],
            key=lambda item: int(item["id"]),
        )

        categories = sorted(
            coco_data["categories"],
            key=lambda item: int(item["id"]),
        )
        self.category_id_to_label = {
            int(category["id"]): label
            for label, category in enumerate(categories)
        }
        self.label_to_category_id = {
            label: category_id
            for category_id, label in self.category_id_to_label.items()
        }
        self.class_names = [
            str(category["name"])
            for category in categories
        ]

        annotations_by_image: DefaultDict[int, List[Dict]] = defaultdict(list)
        for annotation in coco_data["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        self.annotations_by_image = dict(annotations_by_image)

        self.transforms = (
            transforms
            if transforms is not None
            else build_transforms(
                image_size=self.image_size,
                training=self.training,
            )
        )

    @property
    def number_of_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, image_info: Dict) -> Image.Image:
        image_path = self.images_directory / str(image_info["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(
                "Image listed in annotations was not found: {}".format(
                    image_path
                )
            )
        return Image.open(image_path).convert("RGB")

    def _build_target(
        self,
        image_info: Dict,
        image: Image.Image,
    ) -> Target:
        image_id = int(image_info["id"])
        image_width, image_height = image.size

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        crowd_flags: List[int] = []

        for annotation in self.annotations_by_image.get(image_id, []):
            is_crowd = int(annotation.get("iscrowd", 0))
            if is_crowd and not self.include_crowd:
                continue

            category_id = int(annotation["category_id"])
            if category_id not in self.category_id_to_label:
                continue

            x, y, width, height = [
                float(value)
                for value in annotation["bbox"]
            ]

            if width <= 1.0 or height <= 1.0:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(image_width), x + width)
            y2 = min(float(image_height), y + height)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[category_id])
            areas.append(float(annotation.get("area", width * height)))
            crowd_flags.append(is_crowd)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target: Target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowd_flags, dtype=torch.int64),
            "original_size": torch.tensor(
                [image_height, image_width],
                dtype=torch.int64,
            ),
            "size": torch.tensor(
                [image_height, image_width],
                dtype=torch.int64,
            ),
        }
        return target

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Tensor, Target]:
        image_info = self.images[index]
        image = self._load_image(image_info)
        target = self._build_target(image_info, image)
        image, target = self.transforms(image, target)
        return image, target

    def get_coco_category_id(self, contiguous_label: int) -> int:
        """Map a contiguous zero-based model label to the original COCO id."""

        if contiguous_label not in self.label_to_category_id:
            raise KeyError(
                "Unknown contiguous label: {}".format(contiguous_label)
            )
        return self.label_to_category_id[contiguous_label]

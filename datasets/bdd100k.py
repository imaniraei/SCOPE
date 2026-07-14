"""BDD100K dataset wrapper."""

from __future__ import annotations

from typing import Callable, Optional

from .coco_dataset import COCODetectionDataset


class BDD100KDataset(COCODetectionDataset):
    """BDD100K dataset converted to COCO JSON format."""

    def __init__(
        self,
        images_directory: str,
        annotation_file: str,
        image_size: int,
        training: bool = False,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            images_directory=images_directory,
            annotation_file=annotation_file,
            image_size=image_size,
            training=training,
            transforms=transforms,
            dataset_name="BDD100K",
        )

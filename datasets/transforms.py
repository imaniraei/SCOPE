"""Detection-aware image transforms for SCOPE."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


Target = Dict[str, Tensor]


class Compose:
    """Compose transforms that operate on both images and detection targets."""

    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self,
        image: Image.Image,
        target: Target,
    ) -> Tuple[Tensor, Target]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class Resize:
    """Resize an image and scale its bounding boxes."""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.height = int(size[0])
        self.width = int(size[1])

        if self.height <= 0 or self.width <= 0:
            raise ValueError("Resize dimensions must be positive.")

    def __call__(
        self,
        image: Image.Image,
        target: Target,
    ) -> Tuple[Image.Image, Target]:
        original_width, original_height = image.size
        image = image.resize(
            (self.width, self.height),
            resample=Image.BILINEAR,
        )

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            scale_x = self.width / float(original_width)
            scale_y = self.height / float(original_height)
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y

        target = dict(target)
        target["boxes"] = boxes
        target["size"] = torch.tensor(
            [self.height, self.width],
            dtype=torch.int64,
        )
        return image, target


class RandomHorizontalFlip:
    """Randomly flip an image and its bounding boxes."""

    def __init__(self, probability: float = 0.5) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")
        self.probability = probability

    def __call__(
        self,
        image: Image.Image,
        target: Target,
    ) -> Tuple[Image.Image, Target]:
        if random.random() >= self.probability:
            return image, target

        width, _ = image.size
        image = TF.hflip(image)

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            x_min = boxes[:, 0].clone()
            x_max = boxes[:, 2].clone()
            boxes[:, 0] = width - x_max
            boxes[:, 2] = width - x_min

        target = dict(target)
        target["boxes"] = boxes
        return image, target


class RandomColorJitter:
    """Apply brightness, contrast, saturation and hue augmentation."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        probability: float = 0.5,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")

        self.probability = probability
        self.jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self,
        image: Image.Image,
        target: Target,
    ) -> Tuple[Image.Image, Target]:
        if random.random() < self.probability:
            image = self.jitter(image)
        return image, target


class ToTensor:
    """Convert a PIL image to a float tensor in [0, 1]."""

    def __call__(
        self,
        image: Image.Image,
        target: Target,
    ) -> Tuple[Tensor, Target]:
        return TF.to_tensor(image), target


class Normalize:
    """Normalize image tensors using ImageNet statistics."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = list(mean)
        self.std = list(std)

    def __call__(
        self,
        image: Tensor,
        target: Target,
    ) -> Tuple[Tensor, Target]:
        return TF.normalize(image, self.mean, self.std), target


def build_transforms(
    image_size: int,
    training: bool,
) -> Compose:
    """Build the default SCOPE preprocessing and augmentation pipeline."""

    transforms: List[Callable] = [
        Resize((image_size, image_size)),
    ]

    if training:
        transforms.extend(
            [
                RandomHorizontalFlip(probability=0.5),
                RandomColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    probability=0.5,
                ),
            ]
        )

    transforms.extend(
        [
            ToTensor(),
            Normalize(),
        ]
    )

    return Compose(transforms)

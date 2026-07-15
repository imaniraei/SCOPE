# Dataset Preparation

This document describes how to prepare the datasets used in the SCOPE project.

SCOPE currently supports the following datasets:

- Udacity Self-Driving Car Dataset
- KITTI
- BDD100K
- nuScenes

All datasets must be converted to **COCO JSON** format before training or evaluation.

---

# Expected Directory Structure

The repository expects the following directory structure.

```
datasets/

├── udacity/
│   ├── train/
│   │   ├── images/
│   │   └── annotations.json
│   │
│   ├── val/
│   │   ├── images/
│   │   └── annotations.json
│   │
│   └── test/
│       ├── images/
│       └── annotations.json
│
├── kitti/
│   ├── train/
│   ├── val/
│   └── test/
│
├── bdd100k/
│   ├── train/
│   ├── val/
│   └── test/
│
└── nuscenes/
    ├── train/
    ├── val/
    └── test/
```

---

# Supported Annotation Format

SCOPE expects annotations in the standard COCO detection format.

Each dataset should contain:

- images/
- annotations.json

The annotation file must include:

- images
- annotations
- categories

following the official COCO specification.

---

# Dataset Configuration

Update the corresponding YAML configuration inside

```
configs/
```

For example

```
configs/dataset_udacity.yaml
```

and replace

```
/path/to/...
```

with your local dataset paths.

---

# Training Dataset

Example

```
train_images

datasets/udacity/train/images

train_annotations

datasets/udacity/train/annotations.json
```

---

# Validation Dataset

Example

```
val_images

datasets/udacity/val/images

val_annotations

datasets/udacity/val/annotations.json
```

---

# Test Dataset

Example

```
test_images

datasets/udacity/test/images

test_annotations

datasets/udacity/test/annotations.json
```

---

# Supported Datasets

## Udacity Self-Driving Car Dataset

Used as the primary training dataset for SCOPE.

---

## KITTI

Used for cross-dataset evaluation and benchmarking.

---

## BDD100K

Used to evaluate the generalization capability of SCOPE under diverse driving scenarios.

---

## nuScenes

Used to evaluate robustness in large-scale autonomous driving environments.

---

# Notes

- All images should be stored in RGB format.
- Bounding boxes must follow the COCO format.
- Category IDs must be consistent with the annotation file.
- The repository automatically loads the appropriate dataset based on the selected configuration.

---

# Dataset Citation

Please cite the original dataset papers if you use these datasets in your research.

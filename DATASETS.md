# Dataset Preparation

This document describes how to prepare the datasets used in the SCOPE project.

SCOPE currently supports the following datasets:

- Udacity Self-Driving Car Dataset
- KITTI
- BDD100K
- nuScenes

All datasets must be converted to **COCO JSON** format before training or evaluation.

---

# Download the Datasets

The datasets are **not included** in this repository.

Please download each dataset from its official source and store it in a location of your choice (e.g., your local machine, external storage, or HPC filesystem).

---

# Expected Dataset Structure

Each dataset should follow the standard COCO directory structure.

Example:

```
udacity/

├── train/
│   ├── images/
│   └── annotations.json
│
├── val/
│   ├── images/
│   └── annotations.json
│
└── test/
    ├── images/
    └── annotations.json
```

The same structure applies to:

- KITTI
- BDD100K
- nuScenes

The datasets **do not need to be stored inside the SCOPE repository**.

---

# Supported Annotation Format

SCOPE expects annotations in the standard **COCO Detection** format.

Each dataset split should contain

- images/
- annotations.json

The annotation file must include the standard COCO fields:

- images
- annotations
- categories

---

# Dataset Configuration

Specify the dataset locations inside the corresponding YAML configuration file located in

```
configs/
```

For example,

```
configs/dataset_udacity.yaml
```

Replace all placeholder paths

```
/path/to/...
```

with the actual locations of your datasets.

For example,

```
train_images:
/speed-scratch/username/datasets/udacity/train/images

train_annotations:
/speed-scratch/username/datasets/udacity/train/annotations.json
```

The datasets may be stored anywhere on your system or HPC storage.

---

# Supported Datasets

## Udacity Self-Driving Car Dataset

Primary dataset used for training and evaluation.

---

## KITTI

Used for benchmarking and cross-dataset evaluation.

---

## BDD100K

Used to evaluate the generalization capability of SCOPE under diverse driving scenarios.

---

## nuScenes

Used to evaluate robustness on large-scale autonomous driving scenes.

---

# Notes

- Images must be stored in RGB format.
- Annotations must follow the COCO detection format.
- Category IDs must be consistent with the corresponding annotation file.
- Dataset paths are defined through the configuration files and can point to any valid location.
- The repository does **not** require datasets to be copied into the project directory.

---

# Dataset Citation

If you use any of these datasets in your research, please cite the corresponding original dataset papers.

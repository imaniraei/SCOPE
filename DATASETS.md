# Dataset Preparation

This document describes how to prepare the datasets used in the SCOPE project.

SCOPE supports the following autonomous driving datasets:

- Udacity Self-Driving Car Dataset
- KITTI
- BDD100K
- nuScenes

All datasets must be converted to the **COCO Detection JSON** format before training or evaluation.

---

# Download the Datasets

The datasets are **not included** in this repository.

Please download each dataset from its official source and store it in a location of your choice (e.g., a local machine, external storage, or an HPC filesystem).

The repository only requires the paths to these datasets during training and evaluation.

---

# Official Dataset Links

| Dataset | Official Website |
|----------|------------------|
| Udacity Self-Driving Car Dataset | https://github.com/udacity/self-driving-car |
| KITTI Vision Benchmark Suite | https://www.cvlibs.net/datasets/kitti/ |
| BDD100K | https://bdd-data.berkeley.edu/ |
| nuScenes | https://www.nuscenes.org/ |

Please follow the official instructions provided by each dataset for downloading and licensing.

---

# Expected Dataset Structure

Each dataset should follow the standard COCO Detection directory structure.

Example

```
dataset_name/

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

The datasets **do not need to be stored inside the SCOPE repository**.

They may be stored anywhere on your local machine or HPC storage.

---

# Supported Annotation Format

SCOPE expects annotations in the standard **COCO Detection** format.

Each dataset split should contain

- images/
- annotations.json

The annotation file must contain the standard COCO fields:

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

with the actual dataset locations.

Example

```yaml
train_images: /speed-scratch/username/datasets/udacity/train/images

train_annotations: /speed-scratch/username/datasets/udacity/train/annotations.json

val_images: /speed-scratch/username/datasets/udacity/val/images

val_annotations: /speed-scratch/username/datasets/udacity/val/annotations.json
```

Dataset paths may point to any valid location on your local machine or HPC filesystem.

---

# Experimental Protocol

SCOPE is evaluated under two experimental settings.

## In-domain Evaluation

The proposed method is trained and evaluated on:

- Udacity Self-Driving Car Dataset
- KITTI

These experiments measure object detection performance within the same domain.

---

## Cross-domain Generalization

To evaluate the generalization capability of SCOPE, the model is trained **exclusively on the Udacity Self-Driving Car Dataset** and directly evaluated on unseen autonomous driving datasets **without target-domain fine-tuning**.

The cross-domain benchmarks include:

- BDD100K
- nuScenes

This protocol evaluates the robustness and transferability of SCOPE across different autonomous driving environments.

---

# Experimental Summary

| Dataset | Evaluation Setting | COCO Format | Used for Training | Used for Evaluation |
|----------|-------------------|:-----------:|:----------------:|:-------------------:|
| Udacity Self-Driving Car | In-domain | ✅ | ✅ | ✅ |
| KITTI | In-domain | ✅ | ✅ | ✅ |
| BDD100K | Cross-domain | ✅ | ❌ | ✅ |
| nuScenes | Cross-domain | ✅ | ❌ | ✅ |

---

# Supported Datasets

## Udacity Self-Driving Car Dataset

Primary dataset used for model training and in-domain evaluation.

---

## KITTI

Used for additional in-domain evaluation and benchmarking.

---

## BDD100K

Used exclusively for cross-domain generalization evaluation.

---

## nuScenes

Used exclusively for cross-domain generalization evaluation.

---

# Notes

- All datasets must be converted to the COCO Detection format.
- Images should be stored in RGB format.
- Annotation files must follow the official COCO Detection specification.
- Dataset paths are specified through the YAML configuration files.
- The datasets are **not included** in this repository.
- During cross-domain evaluation, **no target-domain fine-tuning** is performed on BDD100K or nuScenes.
- Dataset locations may reside anywhere on a local machine or HPC filesystem.

---

# Dataset Citation

If you use any of these datasets in your research, please cite the corresponding original dataset papers.

Please also respect the license and terms of use provided by each dataset.

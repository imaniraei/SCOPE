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

Please download each dataset from its official source and store it in a location of your choice, such as a local machine, external storage, or an HPC filesystem.

The repository only requires the paths to these datasets during training and evaluation.

---

# Official Dataset Links

| Dataset | Official Website |
|----------|------------------|
| Udacity Self-Driving Car Dataset | https://public.roboflow.com/object-detection/self-driving-car/ |
| KITTI Vision Benchmark Suite | https://www.cvlibs.net/datasets/kitti/ |
| BDD100K | https://bdd-data.berkeley.edu/ |
| nuScenes | https://www.nuscenes.org/ |

Please follow the official instructions provided by each dataset for downloading and licensing.


# Expected Dataset Structure

Each dataset should follow the standard COCO Detection directory structure.

Example:

```text
dataset_name/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```



# Supported Annotation Format

SCOPE expects annotations in the standard **COCO Detection** format.

Each dataset split should contain:

- `images/`
- `annotations.json`

The annotation file must contain the standard COCO fields:

- `images`
- `annotations`
- `categories`


# Dataset Configuration

Specify the dataset locations inside the corresponding YAML configuration file located in:

```text
configs/
```

For example:

```text
configs/dataset_udacity.yaml
```

Replace all placeholder paths:

```text
/path/to/...
```

with the actual dataset locations.

Example:

```yaml
train_images: /speed-scratch/username/datasets/udacity/train/images
train_annotations: /speed-scratch/username/datasets/udacity/train/annotations.json

val_images: /speed-scratch/username/datasets/udacity/val/images
val_annotations: /speed-scratch/username/datasets/udacity/val/annotations.json

test_images: /speed-scratch/username/datasets/udacity/test/images
test_annotations: /speed-scratch/username/datasets/udacity/test/annotations.json
```

Dataset paths may point to any valid location on a local machine or HPC filesystem.

---

# Experimental Protocol

SCOPE is evaluated under two experimental settings:

1. In-domain evaluation
2. Cross-domain generalization

---

## In-domain Evaluation

For in-domain experiments, SCOPE is trained and evaluated separately on each dataset.

| Dataset | Training | Evaluation |
|----------|:--------:|:----------:|
| Udacity Self-Driving Car Dataset | ✅ | ✅ |
| KITTI | ✅ | ✅ |
| BDD100K | ✅ | ✅ |
| nuScenes | ✅ | ✅ |

These experiments measure object detection performance when the training and evaluation data belong to the same domain.

---

## Cross-domain Generalization

For cross-domain experiments, SCOPE is trained exclusively on the Udacity Self-Driving Car Dataset and directly evaluated on unseen target-domain datasets.

| Training Dataset | Evaluation Dataset | Target-Domain Fine-Tuning |
|------------------|--------------------|:-------------------------:|
| Udacity Self-Driving Car Dataset | BDD100K | ❌ |
| Udacity Self-Driving Car Dataset | nuScenes | ❌ |

No target-domain fine-tuning is performed during cross-domain evaluation.

This protocol evaluates the robustness and transferability of SCOPE across different autonomous driving environments.

---

# Supported Datasets

## Udacity Self-Driving Car Dataset

Used for in-domain training and evaluation.

It is also used as the source-domain training dataset for cross-domain evaluation on BDD100K and nuScenes.

---

## KITTI

Used for in-domain training, evaluation, and benchmarking.

---

## BDD100K

Used in two settings:

- In-domain training and evaluation
- Cross-domain evaluation using a model trained on Udacity

---

## nuScenes

Used in two settings:

- In-domain training and evaluation
- Cross-domain evaluation using a model trained on Udacity

---

# Notes

- All datasets must be converted to the COCO Detection format.
- Images should be stored in RGB format.
- Annotation files must follow the official COCO Detection specification.
- Dataset paths are specified through YAML configuration files.
- The datasets are **not included** in this repository.
- During cross-domain evaluation, no target-domain fine-tuning is performed on BDD100K or nuScenes.
- Dataset locations may reside anywhere on a local machine or HPC filesystem.

---

# Dataset Citation

If you use any of these datasets in your research, please cite the corresponding original dataset papers.

Please also respect the license and terms of use provided by each dataset.

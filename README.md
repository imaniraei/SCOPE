# SCOPE

[![Paper](https://img.shields.io/badge/Paper-T--ITS-green)]()
[![Framework](https://img.shields.io/badge/Overview-SCOPE-blue)](#framework-overview)
[![Datasets](https://img.shields.io/badge/Datasets-Udacity%20|%20KITTI%20|%20BDD100K%20|%20nuScenes-yellow)](#datasets)
[![Code](https://img.shields.io/badge/Code-Coming_Soon-lightgrey)]()
[![Status](https://img.shields.io/badge/Status-Under_Review-orange)]()
[![BibTeX](https://img.shields.io/badge/BibTeX-Cite-purple)](#citation)


Official repository of:

**SCOPE: Spatial-Channel Optimization with Efficient Adaptive Fusion for Real-Time Object Detection in Autonomous Vehicles**

### Authors

Iman Iraei, M. Omair Ahmad, and M. N. S. Swamy

---


## 🔥 Teaser

<p align="center">
  <img src="assets/teaser.png" width="1000">
</p>

<p align="center">
SCOPE achieves state-of-the-art accuracy while reducing computational complexity and inference latency across multiple autonomous driving datasets.
</p>

---

## Abstract

Reliable object detection is a critical component of perception systems in autonomous driving. SCOPE introduces a latency-aware adaptive fusion framework that enhances EfficientDet through spatial-channel optimization, adaptive feature fusion, and computationally efficient architecture design.

---

## Framework Overview

The overall architecture of SCOPE is illustrated below.

<p align="center">
  <img src="assets/scope_architecture.png" width="900">
</p>

---

## Key Contributions

- CBAM-enhanced EfficientNet backbone for improved spatial and channel feature representation.
- Adaptive Attention BiFPN for efficient multi-scale feature fusion.
- Reduced inference latency through lightweight adaptive fusion.
- Improved detection performance for distant and occluded objects.
- Real-time deployment capability for autonomous driving systems.

---

## 📂 Datasets

- Udacity Self-Driving Car Dataset
- KITTI Object Detection Benchmark
- BDD100K Dataset
- nuScenes Dataset

---

## 🚀 Code and Models

Code release is coming soon.

The repository will include:

- Training scripts
- Evaluation scripts
- Dataset preparation tools
- Pretrained checkpoints
- Configuration files

---


## 📊 Experimental Results

### Udacity Dataset

<p align="center">
  <img src="assets/udacity_results.png" width="1000">
</p>

<p align="center">
Performance evaluation and Grad-CAM visualization on the Udacity Self-Driving Car dataset.
</p>

## Quantitative Evaluation Across Model Scales on Udacity

| Model | Scale | Input Res. | BiFPN Channels | BiFPN Layers | Box/Class Layers | mAP@0.5 | mAP@0.75 | mAP | Params (M) | FLOPs (B) | Latency (s) |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EfficientDet | 0 | 512 | 64 | 3 | 3 | 66.41 | 54.15 | 52.32 | 3.94 | 2.55 | 0.0810 |
| **SCOPE** | 0 | 512 | 64 | 1 | 3 | **68.32** | **56.22** | **54.41** | **3.16** | **1.91** | **0.0782** |
| EfficientDet | 1 | 640 | 88 | 4 | 3 | 72.15 | 58.09 | 56.19 | 4.31 | 4.88 | 0.0868 |
| **SCOPE** | 1 | 640 | 88 | 1 | 3 | **73.60** | **59.59** | **57.78** | **3.57** | **4.12** | **0.0813** |
| EfficientDet | 2 | 768 | 112 | 5 | 3 | 75.88 | 60.62 | 58.65 | 4.98 | 9.61 | 0.0943 |
| **SCOPE** | 2 | 768 | 112 | 1 | 3 | **76.86** | **61.62** | **59.89** | **4.29** | **8.62** | **0.0856** |
| EfficientDet | 3 | 896 | 160 | 6 | 4 | 78.32 | 62.27 | 60.42 | 6.42 | 19.22 | 0.1042 |
| **SCOPE** | 3 | 896 | 160 | 1 | 4 | **79.00** | **62.96** | **61.05** | **5.94** | **17.75** | **0.0913** |
| EfficientDet | 4 | 1024 | 224 | 7 | 4 | 79.91 | 63.33 | 61.45 | 9.25 | 38.72 | 0.1171 |
| **SCOPE** | 4 | 1024 | 224 | 1 | 4 | **80.41** | **63.83** | **62.01** | **8.96** | **36.27** | **0.0987** |
| EfficientDet | 5 | 1280 | 288 | 7 | 4 | 80.95 | 63.99 | 62.11 | 16.71 | 88.30 | 0.1340 |
| **SCOPE** | 5 | 1280 | 288 | 1 | 4 | **81.14** | **64.19** | **62.49** | **16.62** | **81.89** | **0.1081** |
| EfficientDet | 6 | 1280 | 384 | 8 | 5 | 81.62 | 64.41 | 62.66 | 29.86 | 178.67 | 0.1661 |
| **SCOPE** | 6 | 1280 | 384 | 1 | 5 | **81.68** | **64.47** | **62.71** | **27.44** | **145.24** | **0.1217** |
| EfficientDet | 7 | 1536 | 384 | 8 | 5 | **82.04** | **64.78** | **63.09** | 51.11 | 321.82 | 0.1850 |
| **SCOPE** | 7 | 1536 | 384 | 1 | 5 | 81.97 | 64.68 | 62.98 | **46.51** | **258.25** | **0.1382** |

---

### KITTI Dataset

<p align="center">
  <img src="assets/kitti_results.png" width="1000">
</p>

<p align="center">
Performance evaluation and Grad-CAM visualization on the KITTI benchmark.
</p>

---

### BDD100K Dataset

<p align="center">
  <img src="assets/bdd100k_results.png" width="1000">
</p>

<p align="center">
Performance evaluation and Grad-CAM visualization on the BDD100K dataset.
</p>

---

### nuScenes Dataset

<p align="center">
  <img src="assets/nuscenes_results.png" width="1000">
</p>

<p align="center">
Performance evaluation and Grad-CAM visualization on the nuScenes dataset.
</p>

---

---

## Status

🚧 Code release coming soon.

---

## Citation

```bibtex
@article{iraei2026scope,
  title={SCOPE: Spatial-Channel Optimization with Efficient Adaptive Fusion for Real-Time Object Detection in Autonomous Vehicles},
  author={Iraei, Iman and Ahmad, M. Omair and Swamy, M. N. S.},
  journal={Submitted to IEEE Transactions on Intelligent Transportation Systems (T-ITS)},
  year={2026}
}
```

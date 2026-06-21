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

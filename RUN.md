# Running SCOPE

This document explains how to train, evaluate, and run inference using the official implementation of **SCOPE: Spatial-Channel Optimization with Efficient Adaptive Fusion for Real-Time Object Detection in Autonomous Vehicles**.

---

# Training

Training is performed on an HPC cluster using the SLURM workload manager.

Modify the training configuration inside

```text
configs/
```

and update the dataset paths if necessary.

Launch training using

```bash
sbatch scripts/slurm_train_template.sh
```

or

```bash
python scripts/train.py
```

Future releases will support configuration-based training:

```bash
python scripts/train.py --config configs/scope_s3.yaml
```

---

# Evaluation

Evaluate a trained checkpoint using

```bash
python scripts/evaluate.py
```

Future releases will support

```bash
python scripts/evaluate.py \
    --config configs/scope_s3.yaml \
    --checkpoint outputs/checkpoints/scope_s3_best.pth
```

---

# Inference

Run inference on a single image

```bash
python scripts/inference.py
```

Future releases will support

```bash
python scripts/inference.py \
    --checkpoint outputs/checkpoints/scope_s3_best.pth \
    --image demo.jpg
```

---

# Google Colab Demo

A lightweight Google Colab notebook is provided for inference only.

Open

```text
notebooks/SCOPE_Inference.ipynb
```

Then

1. Upload the pretrained checkpoint.
2. Upload an input image.
3. Run all notebook cells.

---

# Output Files

Training outputs are automatically saved inside

```text
outputs/
```

Typical output files include

```text
outputs/

├── checkpoints/
├── logs/
├── tensorboard/
├── predictions/
├── visualizations/
└── results/
```

---

# Monitoring Training on HPC

Check running jobs

```bash
squeue -u USERNAME
```

Cancel a running job

```bash
scancel JOB_ID
```

Display GPU utilization

```bash
nvidia-smi
```

---

# Reproducing the Results

The experimental results reported in the paper can be reproduced by

1. Preparing the datasets according to **DATASETS.md**
2. Installing the environment following **INSTALL.md**
3. Launching training using the provided SLURM script
4. Evaluating the trained checkpoints
5. Running inference using the provided Colab notebook or inference script

---

# Notes

- Training is designed for HPC environments using NVIDIA A100 GPUs.
- Inference can be performed either locally or using Google Colab.
- The Colab notebook is intended for demonstration purposes and is **not** recommended for model training.

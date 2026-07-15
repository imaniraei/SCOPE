# Installation Guide

This guide explains how to install and run the official implementation of **SCOPE: Spatial-Channel Optimization with Efficient Adaptive Fusion for Real-Time Object Detection in Autonomous Vehicles**.



# 1. System Requirements

The project has been developed and tested on Linux using NVIDIA GPUs.

Minimum requirements:

- Ubuntu 22.04
- Python 3.8
- CUDA 11.5
- NVIDIA GPU (CUDA-enabled)

Recommended:

- NVIDIA A100 GPU
- At least 16 GB GPU memory



# 2. Tested Software Environment

| Component | Version |
|-----------|---------|
| Operating System | Ubuntu 22.04 |
| Python | 3.8 |
| CUDA | 11.5 |
| PyTorch | 2.0.1 |
| GPU | NVIDIA A100 |
| Training Platform | HPC (SLURM) |



# 3. Clone the Repository

```bash
git clone https://github.com/imaniraei/SCOPE.git

cd SCOPE
```



# 4. Create a Python Virtual Environment

```bash
python3 -m venv scope_env

source scope_env/bin/activate
```



# 5. Install Dependencies

Upgrade pip

```bash
pip install --upgrade pip
```

Install all required packages

```bash
pip install -r requirements.txt
```



# 6. Repository Structure

```
SCOPE/

├── assets/
├── configs/
├── datasets/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── utils/

├── README.md
├── INSTALL.md
├── DATASETS.md
├── requirements.txt
```



# 7. Prepare the Dataset

SCOPE uses datasets in **COCO JSON** format.

Currently supported datasets:

- Udacity Self-Driving Car Dataset
- KITTI
- BDD100K
- nuScenes

Please refer to **DATASETS.md** for the required directory structure and dataset preparation instructions.



# 8. Training on HPC (SLURM)

An example SLURM job script is provided in

```
scripts/slurm_train_template.sh
```

Modify the dataset paths and your HPC account information before submission.

Launch training using

```bash
sbatch scripts/slurm_train_template.sh
```



# 9. Running Inference (Google Colab)

A Google Colab notebook is provided for inference.

Open

```
notebooks/SCOPE_Inference.ipynb
```

Then

1. Upload the trained checkpoint
2. Upload an input image
3. Run all notebook cells



# 10. Verify the Installation

Verify that the installation was successful by running

```bash
python scripts/inference.py --help
```

If the help message is displayed successfully, the installation is complete.



# 11. Troubleshooting

### CUDA version mismatch

Ensure that your installed CUDA version matches the PyTorch CUDA version.



### Missing Python packages

Run

```bash
pip install -r requirements.txt
```

again.



### Dataset not found

Verify that all dataset paths are correctly specified.



### CUDA Out of Memory

Reduce the batch size in the training configuration.



# Citation

If you find this repository useful in your research, please consider citing our paper.

See the BibTeX entry provided in the repository README.

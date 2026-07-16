# Installation Guide

This guide explains how to install, configure, and run the official implementation of **SCOPE: Spatial-Channel Optimization with Efficient Adaptive Fusion for Real-Time Object Detection in Autonomous Vehicles**.

---

# 1. System Requirements

The project has been developed and tested on Linux using NVIDIA GPUs.

## Minimum Requirements

- Ubuntu 22.04
- Python 3.8
- CUDA 11.5
- NVIDIA CUDA-enabled GPU

## Recommended Hardware

- NVIDIA A100 GPU
- CUDA 11.5
- At least 16 GB GPU memory
- HPC cluster with SLURM

---

# 2. Tested Software Environment

| Component | Version |
|-----------|---------|
| Operating System | Ubuntu 22.04 |
| Python | 3.8 |
| CUDA | 11.5 |
| PyTorch | 2.0.1 |
| GPU | NVIDIA A100 |
| Training Platform | HPC (SLURM) |

---

# 3. Clone the Repository

```bash
git clone https://github.com/imaniraei/SCOPE.git

cd SCOPE
```

---

# 4. Create a Python Virtual Environment

Create a virtual environment

```bash
python3 -m venv scope_env
```

Activate it

```bash
source scope_env/bin/activate
```

Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# 5. Install Dependencies

## Install PyTorch

```bash
pip install torch==2.0.1 torchvision torchaudio
```

---

## Install Scientific Packages

```bash
pip install numpy pandas matplotlib tqdm
```

---

## Install Computer Vision Packages

```bash
pip install opencv-python

pip install pycocotools

pip install scikit-image

pip install cython
```

---

## Install Additional Packages

```bash
pip install requests

pip install urllib3<2
```

---

## Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

# 6. Verify the Installation

Verify the installed PyTorch version

```bash
python -c "import torch; print(torch.__version__)"
```

Verify CUDA availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Display the GPU name

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Display GPU information

```bash
nvidia-smi
```

---

# 7. Repository Structure

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
├── LICENSE
```

---

# 8. Prepare the Dataset

SCOPE supports the following datasets.

- Udacity Self-Driving Car Dataset
- KITTI
- BDD100K
- nuScenes

Please refer to **DATASETS.md** for dataset preparation, directory organization, and experimental protocols.

---

# 9. Training on HPC (SLURM)

SCOPE is primarily designed to be trained on HPC clusters using the SLURM workload manager.

An example SLURM script is provided in

```
scripts/slurm_train_template.sh
```

Load the required modules

```bash
module load python/3.8.3

module load cuda/11.5
```

Create the virtual environment

```bash
python3 -m venv scope_env

source scope_env/bin/activate
```

Install all required packages

```bash
pip install -r requirements.txt
```

Modify the following items inside the SLURM script before submission.

- Dataset paths
- Output directory
- HPC account name
- Email address
- Number of GPUs
- Memory
- Training time

Submit the training job

```bash
sbatch scripts/slurm_train_template.sh
```

Monitor running jobs

```bash
squeue -u USERNAME
```

Cancel a job

```bash
scancel JOB_ID
```

---

# 10. Model Training

The current implementation supports training using the provided training script.

```bash
python scripts/train.py
```

Future releases will also support configuration-based training using

```bash
python scripts/train.py --config configs/scope_s3.yaml
```

---

# 11. Model Evaluation

Evaluate a trained model

```bash
python scripts/evaluate.py
```

Support for YAML configuration files will be added in a future release.

---

# 12. Running Inference

## Local Inference

```bash
python scripts/inference.py
```

---

## Google Colab

A Google Colab notebook is provided for inference.

Open

```
notebooks/SCOPE_Inference.ipynb
```

Then

1. Upload a trained checkpoint.
2. Upload an input image.
3. Run all notebook cells.

---

# 13. Output Files

Training outputs are saved in

```
outputs/
```

Typical output files include

- Training checkpoints
- Best model weights
- TensorBoard logs
- Evaluation results
- Visualizations

---

# 14. Troubleshooting

## CUDA version mismatch

Ensure that your CUDA version is compatible with the installed PyTorch version.

---

## Missing Python packages

Run

```bash
pip install -r requirements.txt
```

again.

---

## Dataset not found

Verify that all dataset paths are correctly specified in the configuration file.

---

## CUDA Out of Memory

Reduce the training batch size.

---

## SLURM Job Failed

Check the generated

```
*.out
```

and

```
*.err
```

files for detailed error messages.

---

# Citation

If you find SCOPE useful in your research, please cite our paper using the BibTeX entry provided in the repository.

---

# License

This project is released for academic research purposes.

Please refer to the LICENSE file for more information.

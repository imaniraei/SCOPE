
# Installation Guide

This guide describes how to set up and run **SCOPE** in different execution environments.

Supported platforms:

- 💻 Local Linux/Ubuntu
- ☁️ Google Colab
- 🖥️ High-Performance Computing (HPC) clusters with Slurm



## 1. System Requirements

The following software and hardware are recommended for running SCOPE:

- Ubuntu 22.04 or later
- Python 3.8+
- NVIDIA CUDA 11.5 or later
- PyTorch 2.0.1
- torchvision 0.15.2
- NVIDIA A100 GPU (32 GB) or equivalent
- At least 16 GB system memory



## 2. Tested Software Environment

The official implementation of SCOPE has been developed and evaluated using the following software and hardware environment.

| Component | Version |
|-----------|---------|
| Operating System | Ubuntu 22.04 |
| Python | 3.8 |
| CUDA | 11.5 |
| PyTorch | 2.0.1 |
| torchvision | 0.15.2 |
| GPU | NVIDIA A100 (32 GB) |



## 3. Clone the Repository

Clone the official SCOPE repository from GitHub:

```bash
git clone https://github.com/imaniraei/SCOPE.git
cd SCOPE
```



## 4. Create a Python Virtual Environment

We recommend creating a dedicated Python virtual environment before installing the project dependencies.

```bash
python3 -m venv scope_env
source scope_env/bin/activate
```

For Windows users:

```bash
scope_env\Scripts\activate
```



## 5. Install Dependencies

Install all required Python packages using:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

After the installation is complete, SCOPE is ready to use.




## 6. Verify the Installation

To verify that the installation has been completed successfully, run:

```bash
python -c "import torch; print(torch.__version__)"
```

If PyTorch is installed correctly, the command should print the installed version (e.g., `2.0.1`).

You can also verify GPU availability by running:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The command should return:

```text
True
```


## 7. Running on Google Colab

Google Colab is recommended for **inference and quick experimentation only**.

The official Colab notebook demonstrates how to perform object detection using a pretrained SCOPE model on custom images.

> **Note:** Full model training is computationally intensive and is **not recommended** on the standard Google Colab runtime due to GPU availability and execution time limits.

### Inference Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imaniraei/SCOPE/blob/main/notebooks/SCOPE-Inference.ipynb)




## 8. Running on HPC (SLURM)

SCOPE supports distributed training on High-Performance Computing (HPC) clusters managed by the Slurm workload manager.

An example Slurm batch script is provided in:

```text
scripts/slurm_train_template.sh
```

Before submitting the job, update the following fields according to your HPC environment:

- Slurm account
- Partition name
- Email address
- Project directory
- Python virtual environment

Once the script has been configured, submit the training job using:

```bash
sbatch scripts/slurm_train_template.sh
```

Training logs will be written to the `logs/` directory, while model checkpoints will be saved under the `outputs/` directory.



## 9. Troubleshooting

Coming soon.

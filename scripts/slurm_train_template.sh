
#!/bin/bash

#SBATCH --job-name=scope-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=<your-partition>
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>
#SBATCH --output=logs/scope_%j.out
#SBATCH --error=logs/scope_%j.err

# Stop the job if any command fails.
set -e

# Move to the root directory of the SCOPE repository.
cd /path/to/SCOPE

# Create output directories if they do not already exist.
mkdir -p logs
mkdir -p outputs/checkpoints

# Load the software modules available on your HPC cluster.
# Change these versions according to your cluster.
module purge
module load python/3.8.3
module load cuda/11.5

# Activate an existing Python environment.
# Create and install this environment before submitting the job.
source /path/to/your/virtual-environment/bin/activate

# Display basic environment information in the Slurm log.
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
python --version
nvidia-smi

# Run SCOPE training.
python scripts/train.py \
  --config configs/scope_s3.yaml \
  --dataset udacity \
  --output-dir outputs/checkpoints/scope_s3


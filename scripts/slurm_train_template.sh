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

set -e
cd /path/to/SCOPE

mkdir -p logs outputs/scope_s3/checkpoints

module purge
module load python/3.8.3
module load cuda/11.5
source /path/to/your/virtual-environment/bin/activate

python scripts/train.py \
  --dataset udacity \
  --train-images /path/to/udacity/train/images \
  --train-annotations /path/to/udacity/train/annotations.json \
  --val-images /path/to/udacity/val/images \
  --val-annotations /path/to/udacity/val/annotations.json \
  --scale s3 \
  --epochs 120 \
  --batch-size 8 \
  --workers 4 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.00004 \
  --output-directory outputs/scope_s3

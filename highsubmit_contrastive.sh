#!/bin/bash
#BATCH --job-name=contrastive_gpu
#SBATCH --output=logs/contrastive_%j.out
#SBATCH --error=logs/contrastive_%j.err
#SBATCH --partition=mb-h100
#SBATCH --account=joycelab-niall
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48

# Print info for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

# Move to project directory
cd /home/nmille39/contrastive_ruins || exit 1

# Activate your virtual environment
source ~/python_projects/venv/bin/activate

# Make sure logs dir exists
mkdir -p logs

# Run your Python script
python -u main.py --n_samples 10000 --epochs 99999 --raster_limit 99999

echo "Finished at: $(date)"


#!/bin/bash
#SBATCH --account=ai4wy-eap
#SBATCH --partition=ai4wy
#SBATCH --job-name=contrastive_ruins
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2

# ---- CPU / memory
# On 2xH200 nodes you have 144 CPUs and ~1.1 TB RAM available.
# On 1xH200 nodes you have 72 CPUs and ~572 GB RAM available.
#SBATCH --cpus-per-task=64
#SBATCH --mem=0                 # take all memory on the node (safe w/ exclusive)
#SBATCH --exclusive             # full node: best for big DDPM runs

# ---- Time (AI4WY partition shows 1-00:00:00)
#SBATCH --time=1-00:00:00

# ---- Logs
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

set -euo pipefail



echo "JOB START  : $(date)"
echo "HOST       : $(hostname)"
echo "WORKDIR    : $(pwd)"
echo "SLURM JOB  : ${SLURM_JOB_ID}"
echo "GPUS       : ${SLURM_GPUS:-unset}"
echo "CUDA VIS   : ${CUDA_VISIBLE_DEVICES:-unset}"

# --- (Optional) modules, only if your site requires them
# module load arcc gcc slurm 2>/dev/null || true
# module load cuda 2>/dev/null || true

# --- Activate your “always use” Python env (Miniforge example)
if [ -f "$HOME/opt/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
  conda activate ai4wy
fi

source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
conda activate ai4wy

# --- Sanity checks (won’t crash if nvidia-smi unavailable on login image)
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true


# Move to project directory
cd /cluster/medbow/home/nmille39/contrastive_ruins || exit 1
# Make sure logs dir exists
mkdir -p logs

# Run your Python script
python -u main.py --n_samples 37000 --epochs 99999999999999 --raster_limit 99999

echo "Finished at: $(date)"


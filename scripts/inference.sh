#!/usr/bin/bash
#SBATCH -J faas-infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 0-4
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out

echo "[INFER] Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | $(date)"
cd /nas2/data/suaveh97/final_project || exit 1
source /nas2/data/suaveh97/anaconda3/etc/profile.d/conda.sh
conda activate faas_env
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python model/inference.py --freq 1min --adapter_dir checkpoints/best_adapter --pred_dir predictions

echo "[INFER] Done | Exit $? | $(date)"
#!/usr/bin/bash
#SBATCH -J faas-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 0-1
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out

echo "[EVAL] Job $SLURM_JOB_ID | $(date)"
cd /nas2/data/suaveh97/final_project || exit 1
source /nas2/data/suaveh97/anaconda3/etc/profile.d/conda.sh
conda activate faas_env
export PYTHONPATH="$(pwd):$PYTHONPATH"

python eval/metrics.py --pred_dir predictions

echo "[EVAL] Done | Exit $? | $(date)"
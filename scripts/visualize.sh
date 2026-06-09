#!/usr/bin/bash
#SBATCH -J faas-viz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 0-1
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out

echo "[VIZ] Job $SLURM_JOB_ID | $(date)"
cd /nas2/data/suaveh97/final_project || exit 1
source /nas2/data/suaveh97/anaconda3/etc/profile.d/conda.sh
conda activate faas_env
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Failure cases (모델 불필요, predictions.npz만)
python viz/viz_failure.py

# 2) Embeddings + Attention (모델 필요, GPU)
python viz/viz_embed_attn.py

echo "[VIZ] Done | Exit $? | $(date)"
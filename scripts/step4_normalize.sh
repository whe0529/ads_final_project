#!/usr/bin/bash
#SBATCH -J faas-step4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 1-0
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out

echo "[STEP 4] Normalize | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | $(date)"
cd /nas2/data/suaveh97/final_project || exit 1
source /nas2/data/suaveh97/anaconda3/etc/profile.d/conda.sh
conda activate faas_env
export PYTHONPATH="$(pwd):$PYTHONPATH"
python preprocess/step4_normalize.py --load_dir data/processed/03_timeseries --save_dir data/processed/04_normalized --freq 1min --context_length 512
echo "[STEP 4] Done | Exit $? | $(date)"

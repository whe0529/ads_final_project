#!/usr/bin/bash
#SBATCH -J faas-step1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 1-0
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out

echo "[STEP 1] Load Raw | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | $(date)"
cd /nas2/data/suaveh97/final_project || exit 1
source /nas2/data/suaveh97/anaconda3/etc/profile.d/conda.sh
conda activate faas_env
export PYTHONPATH="$(pwd):$PYTHONPATH"
python preprocess/step1_load_raw.py --raw_dir data/raw --save_dir data/processed/01_loaded
echo "[STEP 1] Done | Exit $? | $(date)"

#!/usr/bin/bash
#SBATCH -J faas-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 1-0
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out

echo "[TRAIN] Moirai+LoRA | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | $(date)"
nvidia-smi

cd /nas2/data/suaveh97/final_project || exit 1
source /nas2/data/suaveh97/anaconda3/etc/profile.d/conda.sh
conda activate faas_env
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python model/train_lora.py \
    --freq              1min \
    --context_length    512 \
    --prediction_length 24 \
    --patch_size        64 \
    --batch_size        1 \
    --lr                1e-4 \
    --num_epochs        20 \
    --ckpt_dir          checkpoints

echo "[TRAIN] Done | Exit $? | $(date)"

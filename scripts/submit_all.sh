#!/usr/bin/bash
# =============================================================================
# scripts/submit_all.sh
# 전처리 → 학습 → 추론 → 평가 전체를 하나의 dependency 체인으로 제출.
#
#   step1 → step2 → step3 → step4 → train → inference → evaluate
#
# 전처리만 돌리려면 scripts/submit_pipeline.sh 사용.
# =============================================================================

#SBATCH -J test-run
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 1-0
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out

cd /nas2/data/suaveh97/final_project

eval "$(conda shell.bash hook)"
conda activate faas_env

set -euo pipefail
cd /nas2/data/suaveh97/final_project || exit 1

echo "==================================================="
echo " FaaS Cold Start — FULL Pipeline (preprocess + model)"
echo "==================================================="

J1=$(sbatch --parsable scripts/step1_load_raw.sh)
echo "[submit] step1     → $J1"
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/step2_compute_features.sh)
echo "[submit] step2     → $J2  (after $J1)"
J3=$(sbatch --parsable --dependency=afterok:$J2 scripts/step3_build_timeseries.sh)
echo "[submit] step3     → $J3  (after $J2)"
J4=$(sbatch --parsable --dependency=afterok:$J3 scripts/step4_normalize.sh)
echo "[submit] step4     → $J4  (after $J3)"
JT=$(sbatch --parsable --dependency=afterok:$J4 scripts/train_lora.sh)
echo "[submit] train     → $JT  (after $J4)"
JI=$(sbatch --parsable --dependency=afterok:$JT scripts/inference.sh)
echo "[submit] inference → $JI  (after $JT)"
JE=$(sbatch --parsable --dependency=afterok:$JI scripts/evaluate.sh)
echo "[submit] evaluate  → $JE  (after $JI)"

echo "==================================================="
echo " All jobs submitted. Monitor:  squeue -u \$USER"
echo " Final metrics:  predictions/metrics.json"
echo "==================================================="

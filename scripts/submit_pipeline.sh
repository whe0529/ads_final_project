#!/bin/bash
# =============================================================================
# scripts/submit_pipeline.sh
# 전처리 파이프라인 전체를 Slurm job dependency 체인으로 제출합니다.
#
# 의존성 구조 (afterok = 선행 job이 성공해야 다음 실행):
#
#   step1  ──►  step2 (array)  ──►  step2_merge  ──►  step3 (array)
#                                                          │
#                                       step4  ◄──  step3_merge
#
# array job 의존성에는 'afterok' 사용 → 모든 array task 성공 시 다음 진행
#
# 사용법:
#   bash scripts/submit_pipeline.sh
#   bash scripts/submit_pipeline.sh --n_chunks 100   # 청크 수 변경
# =============================================================================

set -euo pipefail

cd /nas2/data/suaveh97/final_project || exit 1
mkdir -p logs

# ── 파라미터 ─────────────────────────────────────────────────────
N_CHUNKS=50
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_chunks) N_CHUNKS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
ARRAY_RANGE="0-$((N_CHUNKS-1))"

echo "==================================================="
echo " FaaS Cold Start — Preprocessing Pipeline Submit"
echo " N_CHUNKS    : $N_CHUNKS"
echo " ARRAY_RANGE : $ARRAY_RANGE"
echo "==================================================="

# ── STEP 1 : 로드 ────────────────────────────────────────────────
JID1=$(sbatch --parsable scripts/step1_load_raw.sh)
echo "[submit] step1            → JobID $JID1"

# ── STEP 2 : 피처 계산 (array) ──────────────────────────────────
# step1 성공 후 실행. --array 를 런타임에 덮어씀
JID2=$(sbatch --parsable \
        --dependency=afterok:$JID1 \
        --array=$ARRAY_RANGE \
        --export=ALL,N_CHUNKS=$N_CHUNKS \
        scripts/step2_compute_features.sh)
echo "[submit] step2 (array)    → JobID $JID2  (after $JID1)"

# ── STEP 2-merge : 청크 병합 ────────────────────────────────────
# array job 전체(afterok)가 성공해야 실행
JID2M=$(sbatch --parsable \
        --dependency=afterok:$JID2 \
        scripts/step2_merge.sh)
echo "[submit] step2_merge      → JobID $JID2M (after $JID2)"

# ── STEP 3 : 시계열 변환 (array) ────────────────────────────────
JID3=$(sbatch --parsable \
        --dependency=afterok:$JID2M \
        --array=$ARRAY_RANGE \
        --export=ALL,N_CHUNKS=$N_CHUNKS \
        scripts/step3_build_timeseries.sh)
echo "[submit] step3 (array)    → JobID $JID3  (after $JID2M)"

# ── STEP 3-merge : 청크 병합 ────────────────────────────────────
JID3M=$(sbatch --parsable \
        --dependency=afterok:$JID3 \
        scripts/step3_merge.sh)
echo "[submit] step3_merge      → JobID $JID3M (after $JID3)"

# ── STEP 4 : 정규화 ─────────────────────────────────────────────
JID4=$(sbatch --parsable \
        --dependency=afterok:$JID3M \
        scripts/step4_normalize.sh)
echo "[submit] step4            → JobID $JID4  (after $JID3M)"

echo "==================================================="
echo " All jobs submitted."
echo " Monitor with:  squeue -u \$USER"
echo " Final output:  data/processed/04_normalized/"
echo "==================================================="

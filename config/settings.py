# =============================================================================
# config/settings.py
# 전체 파이프라인에서 공통으로 사용하는 상수 및 경로 설정
# =============================================================================

from pathlib import Path

# ── 프로젝트 루트 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 데이터 경로 ───────────────────────────────────────────────────
DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_STEP1      = PROJECT_ROOT / "data" / "processed" / "01_loaded"
DATA_STEP2      = PROJECT_ROOT / "data" / "processed" / "02_features"
DATA_STEP3      = PROJECT_ROOT / "data" / "processed" / "03_timeseries"
DATA_STEP4      = PROJECT_ROOT / "data" / "processed" / "04_normalized"

# ── 컬럼명 ───────────────────────────────────────────────────────
COL_DATE    = "date"
COL_DUR     = "duration"          # 원본 ms 단위
COL_WEEKDAY = "weekday"
COL_FUNC    = "app_func_id_encoded"

FEATURE_COLS = ["inter_arrival_sec", "duration_sec", "concurrency"]

# ── 시계열 설정 ───────────────────────────────────────────────────
FREQ              = "1min"   # 시계열 해상도
MIN_INVOCATIONS   = 50       # 최소 호출 수 (이 이하 함수는 제외)
CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 24       # freq 단위 예측 구간

# ── 정규화 ────────────────────────────────────────────────────────
NORM_METHOD  = "robust"
OUTLIER_Q    = 0.999

# ── Moirai ────────────────────────────────────────────────────────
PATCH_SIZES  = [8, 16, 32, 64, 128]
MOIRAI_SIZE  = "base"        # "small" | "base" | "large"
PATCH_SIZE   = 64            # context_length(512)의 약수. "auto" 대신 고정값 권장
NUM_VARIATES = 3             # inter_arrival_sec, duration_sec, concurrency

# HuggingFace 모델 ID
MOIRAI_HF_ID = f"Salesforce/moirai-1.1-R-{MOIRAI_SIZE}"

# ── 모델 산출물 경로 ─────────────────────────────────────────────
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
PRED_DIR     = PROJECT_ROOT / "predictions"

# ── LoRA ──────────────────────────────────────────────────────────
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
# Moirai transformer의 어텐션/FFN projection 레이어 이름
LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "out_proj"]

# ── 학습 ──────────────────────────────────────────────────────────
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 20
WEIGHT_DECAY   = 1e-2
WARMUP_RATIO   = 0.05
GRAD_CLIP      = 1.0
SEED           = 42
NUM_WORKERS    = 8           # cpus-per-gpu 와 맞춤

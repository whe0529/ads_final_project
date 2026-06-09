"""
model/diagnose_inference.py
───────────────────────────
추론 진단: 학습된 LoRA 어댑터를 로드하고, test 배치 하나로 예측이
정상적으로 나오는지, 예측 텐서 shape이 무엇인지 확인합니다.

전체 inference.py 를 짜기 전에 이걸로 먼저 확인:
    python model/diagnose_inference.py
"""

import sys
import math
import logging
import numpy as np
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP4, CKPT_DIR, FREQ,
    CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE, NUM_VARIATES,
    MOIRAI_HF_ID,
)
from model.dataset import load_wide_array

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 데이터 ───────────────────────────────────────────────────
    arr, func_ids, _ = load_wide_array(DATA_STEP4 / f"timeseries_norm_{FREQ}.parquet")
    D = arr.shape[0]
    log.info(f"D = {D} variates")

    # ── MoiraiFinetune 재생성 + LoRA 어댑터 로드 ────────────────
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
    from uni2ts.data.loader import PackCollate
    from peft import PeftModel

    module = MoiraiModule.from_pretrained(MOIRAI_HF_ID)
    finetune = MoiraiFinetune(
        min_patches=2, min_mask_ratio=0.15, max_mask_ratio=0.5,
        max_dim=max(D + 1, 128),
        num_training_steps=1, num_warmup_steps=0,
        module=module,
        context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH,
        patch_size=PATCH_SIZE,
    )

    # 학습된 LoRA 어댑터 로드 (module 에 얹기)
    adapter_dir = CKPT_DIR / "best_adapter"
    log.info(f"Loading LoRA adapter from {adapter_dir}")
    finetune.module = PeftModel.from_pretrained(finetune.module, str(adapter_dir))
    finetune.to(device).eval()
    log.info("Adapter loaded, model in eval mode")

    # ── test 윈도우 하나 만들기 ──────────────────────────────────
    win = CONTEXT_LENGTH + PREDICTION_LENGTH
    T = arr.shape[1]
    test_start = int(T * 0.9)
    window = arr[:, test_start:test_start + win]
    if window.shape[1] < win:
        window = arr[:, -win:]
    log.info(f"test window: {window.shape}")

    # ── val transform 적용 (예측 시엔 val transform 사용) ───────
    transform = finetune.train_transform_map["_"](
        distance=PREDICTION_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        patch_size=PATCH_SIZE,
    )
    data_entry = {
        "freq": FREQ, "window": 0,
        "target": [window[i].astype(np.float32).copy() for i in range(D)],
    }
    out = transform(data_entry)

    # ── PackCollate (배치 1개) ──────────────────────────────────
    patches_per_var = math.ceil(CONTEXT_LENGTH / PATCH_SIZE) + math.ceil(PREDICTION_LENGTH / PATCH_SIZE)
    max_len = D * patches_per_var + 64
    collate = PackCollate(max_length=max_len, seq_fields=finetune.seq_fields, target_field="target")
    batch = collate([out])

    # GPU로
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    # ── forward → 분포 ──────────────────────────────────────────
    log.info("Running forward...")
    with torch.no_grad():
        distr = finetune(
            **{f: batch[f] for f in list(finetune.seq_fields) + ["sample_id"]}
        )
    log.info(f"Distribution type: {type(distr).__name__}")

    # 점예측: 분포의 median (샘플링) 또는 mean
    try:
        samples = distr.sample(torch.Size([100]))   # (100, B, seq, patch)
        point = samples.median(dim=0).values
        log.info(f"sample-based median: shape={tuple(point.shape)}")
    except Exception as e:
        log.warning(f"sample failed ({e}), trying .mean")
        point = distr.mean
        log.info(f"mean: shape={tuple(point.shape)}")

    # ── prediction_mask로 미래 구간만 추출 확인 ─────────────────
    pred_mask = batch["prediction_mask"]    # (B, seq)
    log.info(f"prediction_mask: shape={tuple(pred_mask.shape)}, "
             f"#True(future)={pred_mask.sum().item()}")
    log.info(f"target: shape={tuple(batch['target'].shape)}")
    log.info(f"patch_size: unique={torch.unique(batch['patch_size']).tolist()}")
    log.info(f"variate_id: shape={tuple(batch['variate_id'].shape)}, "
             f"range=[{batch['variate_id'].min().item()}, {batch['variate_id'].max().item()}]")

    log.info("\n진단 완료. 위 shape들을 inference.py 작성에 반영합니다.")
    log.info("핵심: point 예측 텐서를 prediction_mask로 잘라 → variate_id별로 모아 → "
             "patch 풀어서 → 역정규화 하면 됩니다.")


if __name__ == "__main__":
    main()

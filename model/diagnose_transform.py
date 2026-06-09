"""
model/diagnose_transform.py
───────────────────────────
본 학습 전에 uni2ts transform이 우리 데이터를 올바른 batch로
변환하는지 확인하는 진단 스크립트.

이게 통과하면 train_lora.py 가 돌아갈 준비가 된 거예요.
GPU 노드(ariel-v4)에서 직접 실행:

    python model/diagnose_transform.py
"""

import sys
import logging
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP4, FREQ, CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE,
    NUM_VARIATES, MOIRAI_HF_ID,
)
from model.dataset import load_wide_array

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # 1) 우리 데이터 로드 → wide array (D, T)
    arr, func_ids, _ = load_wide_array(DATA_STEP4 / f"timeseries_norm_{FREQ}.parquet")
    D, T = arr.shape
    log.info(f"Wide array: {arr.shape}  (D={D} variates)")

    # 2) transform 입력 형식으로 변환
    #    GetPatchSize.__call__ 가 기대하는 형식:
    #      {"freq": str, "target": [1d array per variate]}
    #    우리 wide array (D,T) → D개의 1D 시계열 리스트
    #    단, 학습 윈도우 하나만 잘라서 테스트 (앞쪽 context+pred 구간)
    win = CONTEXT_LENGTH + PREDICTION_LENGTH
    window = arr[:, :win]                       # (D, win)
    data_entry = {
        "freq":   FREQ,
        "window": 0,                                # 몇 번째 슬라이딩 윈도우 (0=첫 윈도우)
        "target": [window[i].copy() for i in range(D)],   # list of (win,) arrays
    }
    log.info(f"data_entry: freq={data_entry['freq']}, window={data_entry['window']}, "
             f"target=list of {len(data_entry['target'])} arrays, "
             f"each shape={data_entry['target'][0].shape}")

    # 3) MoiraiFinetune 생성 (transform_map 사용 목적)
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule

    module = MoiraiModule.from_pretrained(MOIRAI_HF_ID)
    log.info(f"patch_sizes available: {module.patch_sizes}")

    finetune = MoiraiFinetune(
        min_patches        = 2,
        min_mask_ratio     = 0.15,
        max_mask_ratio     = 0.5,
        max_dim            = max(D + 1, 128),   # 변수 수 이상으로 잡아야 함
        num_training_steps = 1000,
        num_warmup_steps   = 0,
        module             = module,
        context_length     = CONTEXT_LENGTH,
        prediction_length  = PREDICTION_LENGTH,
        patch_size         = PATCH_SIZE,
    )

    # 4) train_transform 가져와서 적용
    #    transform_map 은 defaultdict(lambda: default_train_transform) 이므로
    #    아무 키로나 호출하면 transform 함수가 나옴
    transform_factory = finetune.train_transform_map["any_key"]
    transform = transform_factory(
        distance          = PREDICTION_LENGTH,
        prediction_length = PREDICTION_LENGTH,
        context_length    = CONTEXT_LENGTH,
        patch_size        = PATCH_SIZE,
    )
    log.info("Transform built. Applying to data_entry...")

    out = transform(data_entry)

    log.info("=== Transform output (batch dict) ===")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            log.info(f"  {k:18s} shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, torch.Tensor):
            log.info(f"  {k:18s} shape={tuple(v.shape)} dtype={v.dtype} (tensor)")
        else:
            log.info(f"  {k:18s} type={type(v).__name__}")

    # 5) seq_fields 가 다 있는지 확인
    missing = [f for f in finetune.seq_fields if f not in out]
    if missing:
        log.error(f"MISSING fields: {missing}")
    else:
        log.info(f"✓ All seq_fields present: {finetune.seq_fields}")

    # 6) PackCollate 로 배치 묶기 테스트 (2개 샘플)
    from uni2ts.data.loader import PackCollate
    max_len = math_ceil_seq_len(D, CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE)
    collate = PackCollate(
        max_length = max_len,
        seq_fields = finetune.seq_fields,
        target_field = "target",
    )
    out2 = transform({"freq": FREQ, "window": 0,
                      "target": [window[i].copy() for i in range(D)]})
    try:
        batch = collate([out, out2])
        log.info("=== After PackCollate (batch of 2) ===")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                log.info(f"  {k:18s} shape={tuple(v.shape)} dtype={v.dtype}")
        log.info("✓ PackCollate OK")
    except Exception as e:
        log.error(f"PackCollate failed: {e}")
        log.error("→ max_length 를 조정해야 할 수 있어요. 위 single-sample 출력의 "
                  "target seq_len 을 보고 max_length 를 키우세요.")

    log.info("\n진단 완료. 위 출력의 shape 들을 train_lora.py 설정에 반영하세요.")


def math_ceil_seq_len(D, ctx, pred, patch):
    """
    packing 후 최대 시퀀스 길이 추정.
    변수 D개 × (context+prediction)/patch 패치 수.
    여유있게 잡음.
    """
    import math
    patches_per_var = math.ceil(ctx / patch) + math.ceil(pred / patch)
    return D * patches_per_var + 10


if __name__ == "__main__":
    main()

"""
model/inference.py
──────────────────
학습된 LoRA 어댑터로 test 구간을 예측하고 컨테이너 스케줄을 생성.

검증된 구조 (diagnose_inference.py 로 확인):
  forward → AffineTransformed 분포
  점예측 (B, seq, max_patch),  prediction_mask True = 변수당 1패치(=510개)
  → prediction_mask로 미래 패치 추출 → variate_id로 함수/채널 매핑
  → 패치 앞 prediction_length개 = 예측값 → 역정규화 → 스케줄

출력:
  predictions/predictions.npz       (preds, trues, func_ids)
  predictions/container_schedule.csv
"""

import sys
import math
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP4, CKPT_DIR, PRED_DIR, FREQ,
    CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE, NUM_VARIATES,
    MOIRAI_HF_ID,
)
from model.dataset import load_wide_array, load_scalers
# pickle이 저장된 RobustScaler 클래스를 찾을 수 있도록 import (이름만 있으면 됨)
from preprocess.step4_normalize import RobustScaler  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
def build_model(D, adapter_dir, device):
    """MoiraiFinetune 재생성 + LoRA 어댑터 로드."""
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
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
    finetune.module = PeftModel.from_pretrained(finetune.module, str(adapter_dir))
    finetune.to(device).eval()
    log.info(f"Model + LoRA adapter loaded from {adapter_dir}")
    return finetune


def make_transform(finetune):
    return finetune.train_transform_map["_"](
        distance=PREDICTION_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        patch_size=PATCH_SIZE,
    )


def make_collate(finetune, D):
    from uni2ts.data.loader import PackCollate
    ppv = math.ceil(CONTEXT_LENGTH / PATCH_SIZE) + math.ceil(PREDICTION_LENGTH / PATCH_SIZE)
    max_len = D * ppv + 64
    return PackCollate(max_length=max_len, seq_fields=finetune.seq_fields, target_field="target")


# ════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_window(finetune, transform, collate, window, D, device, num_samples=100):
    """
    단일 test 윈도우 (D, win) → 변수별 예측 (D, prediction_length).

    점예측: 분포에서 num_samples개 샘플 → median.
    prediction_mask True 위치(변수당 1패치)에서 앞 prediction_length개 추출.
    """
    data_entry = {
        "freq": FREQ, "window": 0,
        "target": [window[i].astype(np.float32).copy() for i in range(D)],
    }
    out   = transform(data_entry)
    batch = collate([out])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    distr = finetune(**{f: batch[f] for f in list(finetune.seq_fields) + ["sample_id"]})

    # 점예측 (median of samples)
    samples = distr.sample(torch.Size([num_samples]))   # (S, B, seq, patch)
    point   = samples.median(dim=0).values              # (B, seq, patch)
    point   = point[0]                                  # (seq, patch)  B=1

    pred_mask  = batch["prediction_mask"][0]            # (seq,)
    variate_id = batch["variate_id"][0]                 # (seq,)

    # 미래 패치 위치만 추출
    fut_idx     = torch.where(pred_mask)[0]             # 변수당 1개 → 총 D개
    fut_patches = point[fut_idx]                        # (D, patch)
    fut_varids  = variate_id[fut_idx].cpu().numpy()     # (D,)  각 패치의 변수 id

    # 변수별로 정렬해 앞 prediction_length개 값 취함
    preds = np.zeros((D, PREDICTION_LENGTH), dtype=np.float32)
    for row in range(fut_patches.shape[0]):
        vid = int(fut_varids[row])
        if vid < D:
            vals = fut_patches[row, :PREDICTION_LENGTH].float().cpu().numpy()
            preds[vid] = vals
    return preds


def denormalize(preds, func_ids, scalers):
    """(D, pred) → 역정규화. D = V*C."""
    C = NUM_VARIATES
    out = preds.copy()
    for vi, fid in enumerate(func_ids):
        sc = scalers.get(int(fid))
        if sc is None:
            continue
        block = preds[vi*C:(vi+1)*C, :].T          # (pred, C)
        block = sc.inverse_transform(block)        # (pred, C)
        out[vi*C:(vi+1)*C, :] = block.T
    return out


def to_schedule(preds_real, func_ids):
    """역정규화 예측 → 컨테이너 스케줄 DataFrame."""
    C = NUM_VARIATES
    rows = []
    for vi, fid in enumerate(func_ids):
        iat  = preds_real[vi*C + 0]
        dur  = preds_real[vi*C + 1]
        conc = preds_real[vi*C + 2]
        for t in range(preds_real.shape[1]):
            n = int(max(0, round(conc[t])))
            if n == 0 and iat[t] <= 0:
                continue
            rows.append({
                "func_id": int(fid),
                "horizon_step": t,
                "next_invocation_in_sec": round(float(iat[t]), 3),
                "hold_duration_sec": round(float(dur[t]), 3),
                "num_containers": n,
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    arr, func_ids, _ = load_wide_array(DATA_STEP4 / f"timeseries_norm_{args.freq}.parquet")
    D = arr.shape[0]

    finetune  = build_model(D, Path(args.adapter_dir), device)
    transform = make_transform(finetune)
    collate   = make_collate(finetune, D)

    # test 구간에서 여러 윈도우 평가
    win = CONTEXT_LENGTH + PREDICTION_LENGTH
    T   = arr.shape[1]
    test_start = int(T * (1 - args.test_ratio))

    starts = list(range(test_start, T - win + 1, PREDICTION_LENGTH))
    if not starts:
        starts = [max(0, T - win)]
    log.info(f"Test windows: {len(starts)}")

    all_preds, all_trues = [], []
    for i, s in enumerate(starts):
        window = arr[:, s:s + win]
        if window.shape[1] < win:
            continue
        preds = predict_window(finetune, transform, collate, window, D, device,
                               num_samples=args.num_samples)
        # 실제 미래값: 윈도우의 context 이후 prediction_length 구간
        trues = window[:, CONTEXT_LENGTH:CONTEXT_LENGTH + PREDICTION_LENGTH]
        all_preds.append(preds)
        all_trues.append(trues)
        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{len(starts)} windows done")

    preds = np.stack(all_preds)   # (N, D, pred)
    trues = np.stack(all_trues)
    log.info(f"Predictions: {preds.shape}")

    # 역정규화
    scalers = load_scalers()
    preds_real = np.stack([denormalize(p, func_ids, scalers) for p in preds])
    trues_real = np.stack([denormalize(t, func_ids, scalers) for t in trues])

    out_dir = Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "predictions.npz",
                        preds=preds_real, trues=trues_real,
                        func_ids=np.array(func_ids))
    log.info(f"Saved → {out_dir/'predictions.npz'}")

    # 컨테이너 스케줄 (마지막 윈도우 기준)
    schedule = to_schedule(preds_real[-1], func_ids)
    schedule.to_csv(out_dir / "container_schedule.csv", index=False)
    log.info(f"Saved container schedule ({len(schedule)} rows) → "
             f"{out_dir/'container_schedule.csv'}")
    log.info(f"\nSample schedule:\n{schedule.head(10).to_string()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq",         default=FREQ)
    p.add_argument("--adapter_dir",  default=str(CKPT_DIR / "best_adapter"))
    p.add_argument("--pred_dir",     default=str(PRED_DIR))
    p.add_argument("--test_ratio",   type=float, default=0.1)
    p.add_argument("--num_samples",  type=int, default=100)
    main(p.parse_args())
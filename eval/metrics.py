"""
eval/metrics.py
───────────────
예측 결과(predictions.npz)에 대해 MSE / MAE / MAPE를 계산합니다.
제안서 Experimental Plan의 Metric: MSE / MAE / MAPE.

채널별로 따로 평가합니다:
  [0] inter_arrival_sec
  [1] duration_sec
  [2] concurrency

호출이 없는 타임스텝(true=0)은 MAPE에서 0 나눗셈을 유발하므로,
MAPE는 true>0 인 지점만으로 계산합니다.
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PRED_DIR, NUM_VARIATES, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps=1e-8):
    """true > 0 인 지점만으로 계산 (0 나눗셈 방지)."""
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ─────────────────────────────────────────────────────────────────
def evaluate_per_channel(preds: np.ndarray,
                         trues: np.ndarray,
                         func_ids: np.ndarray) -> dict:
    """
    preds, trues: (N, D, pred)  D = V*C
    채널별로 전체 함수를 합쳐 metric 계산.
    """
    C = NUM_VARIATES
    V = len(func_ids)
    results = {}

    for ci, ch_name in enumerate(FEATURE_COLS):
        # 모든 함수의 해당 채널 인덱스: ci, ci+C, ci+2C, ...
        ch_idx = [vi * C + ci for vi in range(V)]
        yt = trues[:, ch_idx, :].ravel()
        yp = preds[:, ch_idx, :].ravel()

        results[ch_name] = {
            "MSE":  round(mse(yt, yp), 6),
            "MAE":  round(mae(yt, yp), 6),
            "MAPE": round(mape(yt, yp), 4),
        }

    return results


def evaluate_concurrency_accuracy(preds, trues, func_ids):
    """
    concurrency(컨테이너 수)는 정수라서 추가로 분류 관점 지표를 봅니다.
      - exact match  : 정확히 맞춘 비율
      - under-provision rate : 예측 < 실제 (Cold Start 위험!)
      - over-provision rate  : 예측 > 실제 (자원 낭비)
    """
    C = NUM_VARIATES
    V = len(func_ids)
    ch_idx = [vi * C + 2 for vi in range(V)]   # concurrency = 채널 2

    yt = np.round(trues[:, ch_idx, :].ravel())
    yp = np.round(preds[:, ch_idx, :].ravel())

    exact = float(np.mean(yt == yp))
    under = float(np.mean(yp < yt))   # 부족 → Cold Start 발생 위험
    over  = float(np.mean(yp > yt))   # 과다 → 자원 낭비

    return {
        "exact_match":          round(exact, 4),
        "under_provision_rate": round(under, 4),
        "over_provision_rate":  round(over, 4),
    }


# ─────────────────────────────────────────────────────────────────
def main(args):
    data = np.load(Path(args.pred_dir) / "predictions.npz")
    preds, trues, func_ids = data["preds"], data["trues"], data["func_ids"]
    log.info(f"Loaded predictions: {preds.shape}")

    per_channel = evaluate_per_channel(preds, trues, func_ids)
    conc_acc    = evaluate_concurrency_accuracy(preds, trues, func_ids)

    log.info("── Per-channel metrics ──")
    for ch, m in per_channel.items():
        log.info(f"  {ch:20s} MSE={m['MSE']:.4f}  MAE={m['MAE']:.4f}  MAPE={m['MAPE']:.2f}%")

    log.info("── Concurrency (container count) ──")
    log.info(f"  exact match          : {conc_acc['exact_match']*100:.2f}%")
    log.info(f"  under-provision rate : {conc_acc['under_provision_rate']*100:.2f}%  (Cold Start 위험)")
    log.info(f"  over-provision rate  : {conc_acc['over_provision_rate']*100:.2f}%  (자원 낭비)")

    # JSON 저장
    out = {
        "per_channel": per_channel,
        "concurrency_accuracy": conc_acc,
    }
    out_path = Path(args.pred_dir) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info(f"Saved metrics → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default=str(PRED_DIR))
    main(parser.parse_args())

"""
viz/viz_failure.py
──────────────────
Failure case 분석 — 예측 vs 실제를 시각화하고 어디서 틀렸는지 분석.

predictions.npz (preds, trues, func_ids) 만 있으면 됨. 모델 불필요.

생성물 (viz/figures/):
  1. error_by_channel.png      채널별 오차 분포 (어느 타겟이 어려운가)
  2. pred_vs_true_<ch>.png      예측 vs 실제 산점도 (채널별)
  3. worst_functions.png        함수별 오차 랭킹 (어떤 함수가 어려운가)
  4. timeseries_examples.png    잘 맞춘 / 못 맞춘 함수의 시계열 예시
  5. concurrency_confusion.png  컨테이너 수 예측의 under/over 분석

실행:
    python viz/viz_failure.py
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # GUI 없는 서버용
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PRED_DIR, NUM_VARIATES, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CH_NAMES = FEATURE_COLS   # ["inter_arrival_sec", "duration_sec", "concurrency"]


# ─────────────────────────────────────────────────────────────────
def load_predictions():
    data = np.load(PRED_DIR / "predictions.npz")
    return data["preds"], data["trues"], data["func_ids"]
    # preds, trues: (N, D, pred)  D = V*C


def split_channels(arr, V):
    """(N, D, pred) → 채널별 dict. 각 (N, V, pred)."""
    C = NUM_VARIATES
    return {CH_NAMES[c]: arr[:, [v*C + c for v in range(V)], :] for c in range(C)}


# ─── 1. 채널별 오차 분포 ──────────────────────────────────────────
def plot_error_by_channel(preds, trues, V):
    pc = split_channels(preds, V)
    tc = split_channels(trues, V)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ch in zip(axes, CH_NAMES):
        err = np.abs(pc[ch] - tc[ch]).ravel()
        # 0이 아닌 실제값 지점만 (호출 있는 구간)
        mask = tc[ch].ravel() > 0
        err_active = err[mask]
        ax.hist(err_active, bins=50, color="steelblue", edgecolor="white")
        ax.set_title(f"{ch}\nMAE={err_active.mean():.4f}", fontsize=11)
        ax.set_xlabel("absolute error")
        ax.set_ylabel("count")
        ax.set_yscale("log")
    fig.suptitle("Absolute Error Distribution by Channel (active timesteps)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "error_by_channel.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


# ─── 2. 예측 vs 실제 산점도 ───────────────────────────────────────
def plot_pred_vs_true(preds, trues, V):
    pc = split_channels(preds, V)
    tc = split_channels(trues, V)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ch in zip(axes, CH_NAMES):
        t = tc[ch].ravel()
        p = pc[ch].ravel()
        mask = t > 0
        t, p = t[mask], p[mask]
        # 너무 많으면 샘플링
        if len(t) > 20000:
            idx = np.random.choice(len(t), 20000, replace=False)
            t, p = t[idx], p[idx]
        ax.scatter(t, p, s=3, alpha=0.2, color="darkorange")
        lim = max(t.max(), p.max()) if len(t) else 1
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
        ax.set_title(ch, fontsize=11)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.legend()
    fig.suptitle("Predicted vs True (points on diagonal = correct)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "pred_vs_true.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


# ─── 3. 함수별 오차 랭킹 ──────────────────────────────────────────
def plot_worst_functions(preds, trues, func_ids, V, top=20):
    C = NUM_VARIATES
    # 함수별 평균 절대오차 (IAT 채널 기준 — 가장 어려운 타겟)
    rows = []
    for vi, fid in enumerate(func_ids):
        for ci, ch in enumerate(CH_NAMES):
            idx = vi*C + ci
            t = trues[:, idx, :].ravel()
            p = preds[:, idx, :].ravel()
            m = t > 0
            mae = np.abs(p[m] - t[m]).mean() if m.sum() else 0.0
            rows.append({"func_id": int(fid), "channel": ch, "mae": mae})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ch in zip(axes, CH_NAMES):
        sub = df[df.channel == ch].nlargest(top, "mae")
        ax.barh(sub["func_id"].astype(str), sub["mae"], color="crimson")
        ax.set_title(f"Top {top} hardest functions\n({ch})", fontsize=11)
        ax.set_xlabel("MAE")
        ax.invert_yaxis()
    fig.tight_layout()
    out = FIG_DIR / "worst_functions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")
    df.to_csv(FIG_DIR / "function_errors.csv", index=False)


# ─── 4. 시계열 예시 (best / worst) ───────────────────────────────
def plot_timeseries_examples(preds, trues, func_ids, V):
    C = NUM_VARIATES
    # concurrency 채널로 best/worst 함수 선정
    maes = []
    for vi in range(V):
        idx = vi*C + 2   # concurrency
        t = trues[:, idx, :].ravel()
        p = preds[:, idx, :].ravel()
        maes.append(np.abs(p - t).mean())
    maes = np.array(maes)
    best_vi  = maes.argmin()
    worst_vi = maes.argmax()

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    # 첫 윈도우(N=0)의 24스텝 예측
    for col, (vi, label) in enumerate([(best_vi, "BEST"), (worst_vi, "WORST")]):
        for ci, ch in enumerate(CH_NAMES):
            ax = axes[ci, col]
            t = trues[0, vi*C + ci, :]
            p = preds[0, vi*C + ci, :]
            ax.plot(t, "o-", label="true", color="black", ms=3)
            ax.plot(p, "s--", label="pred", color="red", ms=3)
            ax.set_title(f"{label} func (id={int(func_ids[vi])}) — {ch}", fontsize=10)
            ax.set_xlabel("horizon step (min)")
            ax.legend(fontsize=8)
    fig.suptitle("Forecast Examples: best vs worst function (by concurrency MAE)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "timeseries_examples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


# ─── 5. concurrency under/over 분석 ──────────────────────────────
def plot_concurrency_confusion(preds, trues, V):
    C = NUM_VARIATES
    idx = [vi*C + 2 for vi in range(V)]
    t = np.round(trues[:, idx, :].ravel()).astype(int)
    p = np.round(preds[:, idx, :].ravel()).astype(int)

    diff = p - t   # +면 over, -면 under
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 좌: 오차 분포
    vals, counts = np.unique(diff, return_counts=True)
    colors = ["crimson" if v < 0 else ("steelblue" if v > 0 else "green") for v in vals]
    ax1.bar(vals, counts, color=colors)
    ax1.set_yscale("log")
    ax1.set_title("Container count error (pred - true)\n"
                  "red=under(ColdStart risk), blue=over(waste), green=exact", fontsize=10)
    ax1.set_xlabel("pred - true")
    ax1.set_ylabel("count (log)")
    ax1.set_xlim(-10, 10)

    # 우: under/over/exact 비율
    under = (diff < 0).mean() * 100
    over  = (diff > 0).mean() * 100
    exact = (diff == 0).mean() * 100
    ax2.bar(["under\n(ColdStart)", "exact", "over\n(waste)"],
            [under, exact, over],
            color=["crimson", "green", "steelblue"])
    for i, v in enumerate([under, exact, over]):
        ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11)
    ax2.set_ylabel("percentage")
    ax2.set_title("Provisioning outcome", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "concurrency_confusion.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


# ─────────────────────────────────────────────────────────────────
def main():
    preds, trues, func_ids = load_predictions()
    V = len(func_ids)
    log.info(f"preds {preds.shape}, V={V} functions")

    plot_error_by_channel(preds, trues, V)
    plot_pred_vs_true(preds, trues, V)
    plot_worst_functions(preds, trues, func_ids, V)
    plot_timeseries_examples(preds, trues, func_ids, V)
    plot_concurrency_confusion(preds, trues, V)

    log.info(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
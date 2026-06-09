"""
viz/viz_tsne_compare.py
───────────────────────
"클러스터가 모델 덕인가, 원본 데이터 덕인가"를 검증.

두 가지를 같은 방식으로 t-SNE 해서 나란히 비교:
  (A) 원본 시계열 (step4 정규화 데이터, 모델 안 거침)
  (B) 모델 표현 (Moirai 768차원, viz_embed_attn.py 와 동일)

두 그림이 비슷하면 → 클러스터는 원본 데이터 특징 (모델 기여 적음)
두 그림이 다르면   → 모델이 추가로 학습한 구조가 있음

(A)만 단독으로도 의미있음: 모델 없이 원본만으로 클러스터가 나오는지 확인.

실행:
    python viz/viz_tsne_compare.py            # (A)만 (빠름, GPU 불필요)
    python viz/viz_tsne_compare.py --with_model   # (A)+(B) 둘 다 (GPU 필요)
"""

import sys
import math
import logging
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP4, CKPT_DIR, FREQ, COL_FUNC,
    CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE, NUM_VARIATES,
    MOIRAI_HF_ID,
)
from model.dataset import load_wide_array

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
def raw_features(arr):
    """
    (A) 원본 시계열 특징.
    각 변수(함수×채널)의 시계열을 간단한 통계 벡터로 요약.
    모델을 전혀 안 거침 — 순수 원본 데이터.

    각 변수당 특징: [평균, 표준편차, 최소, 최대, 최근값, 0이아닌비율, ...]
    (전체 시계열을 그대로 쓰면 차원이 너무 커서 통계로 요약)
    """
    D, T = arr.shape
    feats = []
    for i in range(D):
        s = arr[i]
        nz = s[s != 0]
        feats.append([
            s.mean(), s.std(), s.min(), s.max(),
            s[-1],                          # 최근값
            (s != 0).mean(),                # 호출 있는 비율
            np.median(nz) if len(nz) else 0,
            np.percentile(nz, 90) if len(nz) else 0,
        ])
    return np.array(feats, dtype=np.float32)   # (D, 8)


def model_features(arr, D, device):
    """(B) 모델 표현 — viz_embed_attn.py 와 동일 로직."""
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
    from uni2ts.data.loader import PackCollate
    from peft import PeftModel
    import torch

    module = MoiraiModule.from_pretrained(MOIRAI_HF_ID)
    ft = MoiraiFinetune(
        min_patches=2, min_mask_ratio=0.15, max_mask_ratio=0.5,
        max_dim=max(D + 1, 128),
        num_training_steps=1, num_warmup_steps=0, module=module,
        context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH,
        patch_size=PATCH_SIZE,
    )
    ft.module = PeftModel.from_pretrained(ft.module, str(CKPT_DIR / "best_adapter"))
    ft.to(device).eval()

    win = CONTEXT_LENGTH + PREDICTION_LENGTH
    T = arr.shape[1]
    s = int(T * 0.9)
    window = arr[:, s:s+win] if s+win <= T else arr[:, -win:]
    transform = ft.train_transform_map["_"](
        distance=PREDICTION_LENGTH, prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH, patch_size=PATCH_SIZE,
    )
    out = transform({"freq": FREQ, "window": 0,
                     "target": [window[i].astype(np.float32).copy() for i in range(D)]})
    ppv = math.ceil(CONTEXT_LENGTH/PATCH_SIZE) + math.ceil(PREDICTION_LENGTH/PATCH_SIZE)
    collate = PackCollate(max_length=D*ppv+64, seq_fields=ft.seq_fields, target_field="target")
    batch = collate([out])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    reprs = {}
    def hook(m, i, o): reprs["e"] = o[0] if isinstance(o, tuple) else o
    enc = None
    for name, m in ft.module.named_modules():
        if "encoder" in name.lower() and len(list(m.children())) > 0:
            enc = m; break
    h = enc.register_forward_hook(hook)
    with torch.no_grad():
        ft(**{f: batch[f] for f in list(ft.seq_fields) + ["sample_id"]})
    h.remove()

    rep = reprs["e"][0].float().cpu().numpy()
    var_id = batch["variate_id"][0].cpu().numpy()
    feats = np.zeros((D, rep.shape[1]), dtype=np.float32)
    for vid in range(D):
        mask = var_id == vid
        if mask.sum():
            feats[vid] = rep[mask].mean(axis=0)
    return feats


# ─────────────────────────────────────────────────────────────────
def run_tsne(X):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    return TSNE(n_components=2, perplexity=min(30, len(X)-1),
               random_state=42).fit_transform(Xs)


def plot_compare(emb_raw, emb_model, D):
    C = NUM_VARIATES
    channels = np.array([vid % C for vid in range(D)])
    ch_names = ["IAT", "duration", "concurrency"]

    n_panels = 2 if emb_model is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    sc = axes[0].scatter(emb_raw[:,0], emb_raw[:,1], c=channels, cmap="viridis", s=25, alpha=0.7)
    axes[0].set_title("(A) RAW data t-SNE\n(no model — just preprocessed series)", fontsize=12)
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
    cb = plt.colorbar(sc, ax=axes[0], ticks=[0,1,2]); cb.ax.set_yticklabels(ch_names)

    if emb_model is not None:
        sc2 = axes[1].scatter(emb_model[:,0], emb_model[:,1], c=channels, cmap="viridis", s=25, alpha=0.7)
        axes[1].set_title("(B) MODEL representation t-SNE\n(Moirai 768-dim)", fontsize=12)
        axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
        cb2 = plt.colorbar(sc2, ax=axes[1], ticks=[0,1,2]); cb2.ax.set_yticklabels(ch_names)

    fig.suptitle("Raw vs Model representation — are clusters from the model or the data?",
                 fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "tsne_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


# ─────────────────────────────────────────────────────────────────
def main(args):
    arr, func_ids, _ = load_wide_array(DATA_STEP4 / f"timeseries_norm_{FREQ}.parquet")
    D = arr.shape[0]

    # (A) 원본 — 모델 불필요
    log.info("Computing RAW feature t-SNE ...")
    Xr = raw_features(arr)
    emb_raw = run_tsne(Xr)

    # (B) 모델 표현 — 옵션
    emb_model = None
    if args.with_model:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Computing MODEL representation t-SNE (device={device}) ...")
        Xm = model_features(arr, D, device)
        emb_model = run_tsne(Xm)

    plot_compare(emb_raw, emb_model, D)
    log.info("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with_model", action="store_true",
                   help="모델 표현도 함께 비교 (GPU 필요)")
    main(p.parse_args())
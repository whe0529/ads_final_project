"""
viz/viz_embed_attn.py
─────────────────────
Feature embeddings (t-SNE/UMAP) + Attention maps 시각화.

모델에서 중간 출력을 hook으로 추출:
  - representation : Moirai 인코더의 출력 (각 함수/패치의 표현 벡터)
  - attention      : transformer 레이어의 attention 가중치

생성물 (viz/figures/):
  6. embedding_tsne.png        함수 representation을 t-SNE로 2D 투영 + 클러스터
  7. embedding_by_channel.png  채널(IAT/dur/conc)별로 색칠
  8. attention_map.png         변수 간 attention 히트맵 (함수 간 종속성)

실행:
    python viz/viz_embed_attn.py
"""

import sys
import math
import logging
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
def build_model(D, device):
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
    from peft import PeftModel

    module = MoiraiModule.from_pretrained(MOIRAI_HF_ID)
    ft = MoiraiFinetune(
        min_patches=2, min_mask_ratio=0.15, max_mask_ratio=0.5,
        max_dim=max(D + 1, 128),
        num_training_steps=1, num_warmup_steps=0,
        module=module,
        context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH,
        patch_size=PATCH_SIZE,
    )
    ft.module = PeftModel.from_pretrained(ft.module, str(CKPT_DIR / "best_adapter"))
    ft.to(device).eval()
    return ft


def make_batch(arr, D, finetune, device):
    """test 윈도우 하나를 batch로 변환."""
    from uni2ts.data.loader import PackCollate
    win = CONTEXT_LENGTH + PREDICTION_LENGTH
    T = arr.shape[1]
    s = int(T * 0.9)
    window = arr[:, s:s+win] if s+win <= T else arr[:, -win:]

    transform = finetune.train_transform_map["_"](
        distance=PREDICTION_LENGTH, prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH, patch_size=PATCH_SIZE,
    )
    out = transform({"freq": FREQ, "window": 0,
                     "target": [window[i].astype(np.float32).copy() for i in range(D)]})
    ppv = math.ceil(CONTEXT_LENGTH/PATCH_SIZE) + math.ceil(PREDICTION_LENGTH/PATCH_SIZE)
    collate = PackCollate(max_length=D*ppv+64, seq_fields=finetune.seq_fields, target_field="target")
    batch = collate([out])
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


# ─────────────────────────────────────────────────────────────────
# representation 추출 (forward hook)
# ─────────────────────────────────────────────────────────────────
def extract_representation(finetune, batch):
    """
    Moirai 인코더의 출력을 hook으로 추출.
    encoder 모듈의 forward 출력 = 각 패치의 표현 벡터.
    """
    reprs = {}

    def hook(module, inp, out):
        # out: (B, seq, d_model) 또는 tuple
        reprs["enc"] = out[0] if isinstance(out, tuple) else out

    # encoder 찾기
    enc = None
    for name, m in finetune.module.named_modules():
        if name.endswith("encoder") and m.__class__.__name__.lower().find("transformer") >= 0:
            enc = m
            break
    if enc is None:
        # fallback: 이름에 encoder 포함된 첫 모듈
        for name, m in finetune.module.named_modules():
            if "encoder" in name.lower() and len(list(m.children())) > 0:
                enc = m
                break

    if enc is None:
        log.warning("encoder 모듈을 못 찾음. representation 추출 skip")
        return None

    h = enc.register_forward_hook(hook)
    with torch.no_grad():
        finetune(**{f: batch[f] for f in list(finetune.seq_fields) + ["sample_id"]})
    h.remove()

    return reprs.get("enc")   # (B, seq, d_model)


# ─────────────────────────────────────────────────────────────────
# 1) t-SNE 임베딩
# ─────────────────────────────────────────────────────────────────
def plot_embeddings(repr_tensor, batch, D):
    if repr_tensor is None:
        return
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    rep = repr_tensor[0].float().cpu().numpy()       # (seq, d_model)
    var_id = batch["variate_id"][0].cpu().numpy()    # (seq,)

    # 각 변수(함수×채널)별 평균 representation
    C = NUM_VARIATES
    V = D // C
    var_reprs, var_labels, var_channels = [], [], []
    for vid in range(D):
        mask = var_id == vid
        if mask.sum() == 0:
            continue
        var_reprs.append(rep[mask].mean(axis=0))
        var_labels.append(vid // C)        # 함수 번호
        var_channels.append(vid % C)       # 채널 번호
    var_reprs = np.array(var_reprs)
    if len(var_reprs) < 5:
        log.warning("representation 샘플이 너무 적어 t-SNE skip")
        return

    log.info(f"t-SNE on {var_reprs.shape} ...")
    emb = TSNE(n_components=2, perplexity=min(30, len(var_reprs)-1),
               random_state=42).fit_transform(var_reprs)

    # (a) 클러스터로 색칠
    n_clusters = min(6, len(var_reprs))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(var_reprs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sc1 = ax1.scatter(emb[:, 0], emb[:, 1], c=km.labels_, cmap="tab10", s=25, alpha=0.7)
    ax1.set_title(f"Function representations (t-SNE)\nKMeans {n_clusters} clusters", fontsize=12)
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")
    plt.colorbar(sc1, ax=ax1, label="cluster")

    # (b) 채널로 색칠
    ch_names = ["IAT", "duration", "concurrency"]
    colors = np.array(var_channels)
    sc2 = ax2.scatter(emb[:, 0], emb[:, 1], c=colors, cmap="viridis", s=25, alpha=0.7)
    ax2.set_title("Same embedding, colored by channel", fontsize=12)
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    cbar = plt.colorbar(sc2, ax=ax2, ticks=[0,1,2])
    cbar.ax.set_yticklabels(ch_names)

    fig.tight_layout()
    out = FIG_DIR / "embedding_tsne.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


# ─────────────────────────────────────────────────────────────────
# 2) Attention map
# ─────────────────────────────────────────────────────────────────
def plot_attention(finetune, batch, D):
    """
    transformer self-attention 가중치를 hook으로 추출해 히트맵.
    변수 간 평균 attention으로 "함수 간 종속성"을 봄.
    """
    attns = {}

    def make_hook(name):
        def hook(module, inp, out):
            # attention 모듈마다 출력 형태가 다름. softmax 직후 weight를 노림.
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                attns[name] = out[1]
        return hook

    handles = []
    for name, m in finetune.module.named_modules():
        if name.lower().endswith("self_attn") or "attention" in m.__class__.__name__.lower():
            handles.append(m.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        finetune(**{f: batch[f] for f in list(finetune.seq_fields) + ["sample_id"]})
    for h in handles:
        h.remove()

    if not attns:
        log.warning("attention 가중치를 hook으로 못 잡음 (Moirai가 내부에서 "
                    "scaled_dot_product_attention을 쓰면 weight가 노출 안 됨). "
                    "attention map skip. 대신 variate_id 기반 구조만 표시.")
        _plot_variate_structure(batch, D)
        return

    # 첫 레이어 attention 사용
    key = list(attns.keys())[0]
    attn = attns[key]
    log.info(f"attention from {key}: shape={tuple(attn.shape)}")
    # (B, heads, seq, seq) → head 평균 → 첫 배치
    if attn.dim() == 4:
        a = attn[0].mean(0).float().cpu().numpy()
    else:
        a = attn[0].float().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(a, cmap="hot", aspect="auto")
    ax.set_title(f"Self-attention map ({key})\nbrighter = stronger attention", fontsize=11)
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out = FIG_DIR / "attention_map.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def _plot_variate_structure(batch, D):
    """attention을 못 빼면, 대신 variate_id 구성을 시각화 (대체용)."""
    var_id = batch["variate_id"][0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(var_id, ".", ms=2)
    ax.set_title("variate_id over packed sequence\n(각 패치가 어느 변수에 속하는지)", fontsize=11)
    ax.set_xlabel("packed position"); ax.set_ylabel("variate_id")
    fig.tight_layout()
    out = FIG_DIR / "variate_structure.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out} (attention 대체)")


# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    arr, func_ids, _ = load_wide_array(DATA_STEP4 / f"timeseries_norm_{FREQ}.parquet")
    D = arr.shape[0]

    finetune = build_model(D, device)
    batch = make_batch(arr, D, finetune, device)

    # representation → t-SNE
    rep = extract_representation(finetune, batch)
    plot_embeddings(rep, batch, D)

    # attention map
    plot_attention(finetune, batch, D)

    log.info(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
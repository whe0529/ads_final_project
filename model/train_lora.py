"""
model/train_lora.py
───────────────────
Moirai + 단일 LoRA 어댑터 fine-tuning (uni2ts 2.0 / PyTorch Lightning).

검증된 흐름 (diagnose_transform.py 로 확인 완료):
  wide array
    → {"freq","window","target"} dict
    → MoiraiFinetune.train_transform  (patchify + masking + id 부여)
    → PackCollate                     (배치 묶기, sample_id 자동 추가)
    → MoiraiFinetune (NLL loss + optimizer 내장)에 LoRA 주입
    → Lightning Trainer.fit()

설계:
  - Moirai backbone freeze, LoRA 어댑터 1개만 학습.
  - 함수 = 변수(variate). 510개 변수(170함수×3채널)를 multivariate로 입력.
  - patch 변환/손실은 uni2ts 검증 코드를 그대로 사용 (직접 구현 X).
"""

import os
import sys
import math
import logging
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
import lightning as L

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP4, CKPT_DIR, FREQ,
    CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE, NUM_VARIATES,
    MOIRAI_HF_ID,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, SEED, NUM_WORKERS,
)
from model.dataset import load_wide_array
from model.moirai_lora import inject_lora, save_lora_adapter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. Dataset — wide array에서 윈도우를 잘라 transform 적용
# ════════════════════════════════════════════════════════════════
class FaaSFinetuneDataset(TorchDataset):
    """
    wide array (D, T)에서 슬라이딩 윈도우를 만들고,
    uni2ts transform을 적용해 patch batch dict를 반환합니다.

    transform 입력 형식 (diagnose로 검증):
        {"freq": str, "window": int, "target": [1d array per variate]}
    """
    def __init__(self, arr, transform, context_length, prediction_length,
                 stride, split, val_ratio=0.1, test_ratio=0.1):
        self.transform = transform
        self.ctx  = context_length
        self.pred = prediction_length
        self.win  = context_length + prediction_length
        self.freq = FREQ

        D, T = arr.shape
        test_start = int(T * (1 - test_ratio))
        val_start  = int(T * (1 - test_ratio - val_ratio))

        if split == "train":
            self.data = arr[:, :val_start]
        elif split == "val":
            lo = max(0, val_start - context_length)
            self.data = arr[:, lo:test_start]
        elif split == "test":
            lo = max(0, test_start - context_length)
            self.data = arr[:, lo:]
        else:
            raise ValueError(split)

        self.D = D
        Tn = self.data.shape[1]
        self.starts = list(range(0, max(1, Tn - self.win + 1), stride)) or [0]
        log.info(f"[{split}] data={self.data.shape} windows={len(self.starts)}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s   = self.starts[idx]
        win = self.data[:, s:s + self.win]                  # (D, win)
        if win.shape[1] < self.win:                         # 좌측 zero-pad
            pad = np.zeros((self.D, self.win - win.shape[1]), dtype=np.float32)
            win = np.concatenate([pad, win], axis=1)

        data_entry = {
            "freq":   self.freq,
            "window": 0,                                    # 이미 윈도우를 잘라줬으므로 0
            "target": [win[i].astype(np.float32).copy() for i in range(self.D)],
        }
        return self.transform(data_entry)


# ════════════════════════════════════════════════════════════════
# 2. 메인 학습
# ════════════════════════════════════════════════════════════════
def main(args):
    L.seed_everything(args.seed)
    log.info(f"CUDA available: {torch.cuda.is_available()}")

    # ── 데이터 로드 ──────────────────────────────────────────────
    arr, func_ids, _ = load_wide_array(DATA_STEP4 / f"timeseries_norm_{args.freq}.parquet")
    D = arr.shape[0]
    log.info(f"Variate dim D = {D} ({len(func_ids)} funcs × {NUM_VARIATES} channels)")

    # ── MoiraiFinetune 생성 (transform/loss/optim 내장) ─────────
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
    from uni2ts.data.loader import PackCollate

    module = MoiraiModule.from_pretrained(MOIRAI_HF_ID)

    # 학습 스텝 수 추정 (스케줄러용) — 우선 대략 잡고 아래서 보정
    finetune = MoiraiFinetune(
        min_patches        = 2,
        min_mask_ratio     = 0.15,
        max_mask_ratio     = 0.5,
        max_dim            = max(D + 1, 128),     # 변수 수 이상
        num_training_steps = 10_000,              # 아래서 실제 값으로 갱신
        num_warmup_steps   = 0,
        module             = module,
        context_length     = args.context_length,
        prediction_length  = args.prediction_length,
        patch_size         = args.patch_size,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
    )
    log.info(f"MoiraiFinetune ready. patch_sizes={module.patch_sizes}")

    # ── LoRA 주입 (backbone freeze + 어댑터 1개) ────────────────
    # finetune.module 에 직접 주입
    finetune.module = inject_lora(
        finetune.module, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )

    # ── transform 가져오기 (train / val) ────────────────────────
    train_tf = finetune.train_transform_map["_"](
        distance          = args.prediction_length,
        prediction_length = args.prediction_length,
        context_length    = args.context_length,
        patch_size        = args.patch_size,
    )
    val_tf = finetune.train_transform_map["_"](      # train transform 재사용 (동일 구조)
        distance          = args.prediction_length,
        prediction_length = args.prediction_length,
        context_length    = args.context_length,
        patch_size        = args.patch_size,
    )

    # ── Dataset / DataLoader ────────────────────────────────────
    train_ds = FaaSFinetuneDataset(arr, train_tf, args.context_length,
                                   args.prediction_length, args.stride, "train")
    val_ds   = FaaSFinetuneDataset(arr, val_tf, args.context_length,
                                   args.prediction_length, args.prediction_length, "val")

    # packing 후 시퀀스 최대 길이 (diagnose 출력 기준 여유있게)
    patches_per_var = math.ceil(args.context_length / args.patch_size) + \
                      math.ceil(args.prediction_length / args.patch_size)
    max_len = D * patches_per_var + 64

    collate = PackCollate(
        max_length   = max_len,
        seq_fields   = finetune.seq_fields,
        target_field = "target",
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    # 실제 학습 스텝 수로 스케줄러 보정
    total_steps = len(train_loader) * args.num_epochs
    finetune.hparams.num_training_steps = total_steps
    finetune.hparams.num_warmup_steps   = int(total_steps * 0.05)
    log.info(f"total_steps={total_steps}, max_len(packing)={max_len}")

    # ── Lightning Trainer ───────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath    = str(ckpt_dir),
        filename   = "moirai_lora_best",
        monitor    = "val/PackedNLLLoss",
        mode       = "min",
        save_top_k = 1,
    )
    lr_cb = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs        = args.num_epochs,
        accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
        devices           = 1,
        precision         = "bf16-mixed",
        gradient_clip_val = 1.0,
        callbacks         = [ckpt_cb, lr_cb],
        log_every_n_steps = args.log_every,
        default_root_dir  = str(ckpt_dir),
    )

    log.info("Start training ...")
    trainer.fit(finetune, train_loader, val_loader)

    # ── LoRA 어댑터만 저장 ──────────────────────────────────────
    save_lora_adapter(finetune.module, ckpt_dir / "best_adapter")
    log.info(f"Done. Adapter saved → {ckpt_dir/'best_adapter'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq",              default=FREQ)
    p.add_argument("--context_length",    type=int, default=CONTEXT_LENGTH)
    p.add_argument("--prediction_length", type=int, default=PREDICTION_LENGTH)
    p.add_argument("--patch_size",        type=int, default=PATCH_SIZE)
    p.add_argument("--stride",            type=int, default=PREDICTION_LENGTH)
    p.add_argument("--batch_size",        type=int, default=BATCH_SIZE)
    p.add_argument("--lr",                type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay",      type=float, default=WEIGHT_DECAY)
    p.add_argument("--num_epochs",        type=int, default=NUM_EPOCHS)
    p.add_argument("--num_workers",       type=int, default=NUM_WORKERS)
    p.add_argument("--seed",              type=int, default=SEED)
    p.add_argument("--lora_r",            type=int, default=LORA_R)
    p.add_argument("--lora_alpha",        type=int, default=LORA_ALPHA)
    p.add_argument("--lora_dropout",      type=float, default=LORA_DROPOUT)
    p.add_argument("--ckpt_dir",          default=str(CKPT_DIR))
    p.add_argument("--log_every",         type=int, default=10)
    main(p.parse_args())

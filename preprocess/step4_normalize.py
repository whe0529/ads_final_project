"""
preprocess/step4_normalize.py
──────────────────────────────
[STEP 4]  정규화 + Patching 검증

RobustScaler (함수별 독립, train 80% 구간으로 fit)
  - 호출이 없는 타임스텝(값=0)은 변환 제외
  - 역변환 시: concurrency → round + clip(≥0), IAT/duration → clip(≥0)

Patching 검증
  - context_length % patch_size == 0 확인
  - 512 = 8×64 = 16×32 = 32×16 = 64×8 — 모든 Moirai patch size 호환
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP3, DATA_STEP4,
    COL_FUNC, FREQ,
    FEATURE_COLS, PATCH_SIZES, CONTEXT_LENGTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── RobustScaler ─────────────────────────────────────────────────
@dataclass
class RobustScaler:
    """채널별(3채널) 독립 RobustScaler."""
    func_id : int
    medians : np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    iqrs    : np.ndarray = field(default_factory=lambda: np.ones(3,  dtype=np.float32))

    def fit(self, X: np.ndarray) -> "RobustScaler":
        """X: (T, 3). 값이 0보다 큰 타임스텝만으로 fit."""
        for c in range(3):
            vals = X[X[:, c] > 0, c]
            if len(vals) < 2:
                continue
            self.medians[c] = float(np.median(vals))
            q75, q25        = np.percentile(vals, [75, 25])
            self.iqrs[c]    = float(max(q75 - q25, 1e-8))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """0이 아닌 값만 변환. shape 유지: (T, 3)."""
        X    = X.astype(np.float32).copy()
        mask = X > 0
        for c in range(3):
            idx       = mask[:, c]
            X[idx, c] = (X[idx, c] - self.medians[c]) / self.iqrs[c]
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """예측값 → 원래 스케일 복원. shape: (T, 3) 또는 (3,)."""
        X = X.astype(np.float32).copy()
        for c in range(3):
            X[..., c] = X[..., c] * self.iqrs[c] + self.medians[c]
        # 물리적 제약
        X[..., 0] = np.clip(X[..., 0], 0, None)                            # IAT ≥ 0
        X[..., 1] = np.clip(X[..., 1], 0, None)                            # duration ≥ 0
        X[..., 2] = np.clip(np.round(X[..., 2]), 0, None).astype(np.float32)  # concurrency: 정수 ≥ 0
        return X


# ─── 정규화 ───────────────────────────────────────────────────────
def normalize_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, RobustScaler]]:
    scalers    = {}
    norm_parts = []

    for fid, group in df.groupby(COL_FUNC):
        group   = group.sort_values("timestamp").copy()
        X       = group[FEATURE_COLS].values       # (T, 3)
        n_train = int(len(X) * 0.8)

        scaler  = RobustScaler(func_id=int(fid))
        scaler.fit(X[:n_train])
        scalers[int(fid)] = scaler

        group[FEATURE_COLS] = scaler.transform(X)
        norm_parts.append(group)

    log.info(f"Normalized {len(scalers):,} functions")
    return pd.concat(norm_parts, ignore_index=True), scalers


# ─── Patching 검증 ────────────────────────────────────────────────
def verify_patching(context_len: int = CONTEXT_LENGTH) -> int:
    log.info(f"Patching compatibility (context_length={context_len}):")
    compatible = []
    for ps in PATCH_SIZES:
        ok   = context_len % ps == 0
        n    = context_len // ps
        mark = "✓" if ok else f"✗  remainder={context_len % ps}"
        log.info(f"  patch_size={ps:3d}  n_patches={n:4d}  {mark}")
        if ok:
            compatible.append(ps)

    recommended = max(compatible) if compatible else PATCH_SIZES[0]
    log.info(f"  → Recommended patch_size: {recommended}")
    return recommended


# ─── 메인 ─────────────────────────────────────────────────────────
def main(args):
    load_dir = Path(args.load_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(load_dir / f"timeseries_{args.freq}.parquet")
    log.info(f"Loaded {len(df):,} rows, {df[COL_FUNC].nunique():,} functions")

    df_norm, scalers = normalize_all(df)

    # 저장
    ts_out = save_dir / f"timeseries_norm_{args.freq}.parquet"
    sc_out = save_dir / "scalers.pkl"

    df_norm.to_parquet(ts_out, index=False)
    with open(sc_out, "wb") as f:
        pickle.dump(scalers, f)

    log.info(f"Saved timeseries → {ts_out}")
    log.info(f"Saved scalers    → {sc_out}  ({len(scalers)} scalers)")

    verify_patching(args.context_length)

    # 역변환 동작 확인
    sample_fid = list(scalers.keys())[0]
    sc         = scalers[sample_fid]
    dummy      = np.array([[1.0, 0.5, 2.3], [0.0, 0.0, 0.0]], dtype=np.float32)
    restored   = sc.inverse_transform(dummy)
    log.info(f"Inverse transform test (func_id={sample_fid}):")
    log.info(f"  normalized  {dummy[0]}")
    log.info(f"  restored    {restored[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir",       default=str(DATA_STEP3))
    parser.add_argument("--save_dir",       default=str(DATA_STEP4))
    parser.add_argument("--freq",           default=FREQ)
    parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH)
    main(parser.parse_args())

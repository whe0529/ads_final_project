"""
preprocess/step3_build_timeseries.py
─────────────────────────────────────
[STEP 3]  균일 그리드 시계열 변환 (Slurm Array Job 지원)

event-based 호출 로그 → freq 단위 균일 그리드 시계열

집계 전략 (bin 내 복수 호출):
  inter_arrival_sec  첫 번째 호출의 IAT
  duration_sec       평균 (mean)
  concurrency        최대 (max)

호출 없는 타임스텝 → 0으로 fill

출력 컬럼:
  timestamp | inter_arrival_sec | duration_sec | concurrency | app_func_id_encoded
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP2, DATA_STEP3,
    COL_FUNC, COL_DATE,
    FREQ, MIN_INVOCATIONS, FEATURE_COLS,
    CONTEXT_LENGTH, PREDICTION_LENGTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 단일 함수 → 균일 그리드 ──────────────────────────────────────
def to_uniform_grid(group: pd.DataFrame, freq: str = FREQ) -> pd.DataFrame:
    """
    단일 함수의 이벤트 로그를 freq 단위 균일 그리드로 변환.
    호출이 없는 타임스텝은 0으로 채웁니다.
    """
    group = group.set_index(COL_DATE).sort_index()

    agg = group.resample(freq).agg(
        inter_arrival_sec = ("inter_arrival_sec", "first"),
        duration_sec      = ("duration_sec",      "mean"),
        concurrency       = ("concurrency",       "max"),
    ).fillna(0.0)

    agg["concurrency"] = agg["concurrency"].clip(lower=0).astype(np.int32)

    # 결측 타임스텝 채우기 (reindex)
    full_idx  = pd.date_range(agg.index.min(), agg.index.max(), freq=freq)
    agg       = agg.reindex(full_idx, fill_value=0.0)
    agg.index.name = "timestamp"

    return agg.reset_index()


# ─── train / val / test 분할 ─────────────────────────────────────
def split_ts(ts: pd.DataFrame,
             val_ratio: float  = 0.1,
             test_ratio: float = 0.1,
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """시간순 분할 (랜덤 split 금지)."""
    n          = len(ts)
    test_start = int(n * (1 - test_ratio))
    val_start  = int(n * (1 - test_ratio - val_ratio))
    return (
        ts.iloc[:val_start].copy(),
        ts.iloc[val_start:test_start].copy(),
        ts.iloc[test_start:].copy(),
    )


# ─── GluonTS / Moirai 입력 레코드 생성 ───────────────────────────
def make_moirai_record(fid: int,
                       ts:  pd.DataFrame,
                       split_name: str = "train",
                       freq: str = FREQ,
                       ctx_len: int = CONTEXT_LENGTH,
                      ) -> dict:
    """
    target shape: (3, T)
      [0] inter_arrival_sec
      [1] duration_sec
      [2] concurrency
    """
    train_ts, val_ts, test_ts = split_ts(ts)

    if split_name == "train":
        data = train_ts
    elif split_name == "val":
        ctx  = train_ts.iloc[-min(ctx_len, len(train_ts)):]
        data = pd.concat([ctx, val_ts], ignore_index=True)
    elif split_name == "test":
        combined = pd.concat([train_ts, val_ts], ignore_index=True)
        ctx      = combined.iloc[-min(ctx_len, len(combined)):]
        data     = pd.concat([ctx, test_ts], ignore_index=True)
    else:
        raise ValueError(f"Unknown split: {split_name}")

    target = data[FEATURE_COLS].values.T.astype(np.float32)  # (3, T)

    return {
        "start"  : pd.Period(data["timestamp"].iloc[0], freq=freq),
        "target" : target,
        "item_id": str(fid),
    }


# ─── 청크 처리 ────────────────────────────────────────────────────
def process_chunk(df: pd.DataFrame,
                  summary: pd.DataFrame,
                  func_ids: list,
                  task_id: int,
                  n_chunks: int,
                  save_dir: Path,
                  freq: str,
                  min_inv: int) -> None:

    # 최소 호출 수 필터
    valid_funcs = set(
        summary.loc[summary["num_invocations"] >= min_inv, COL_FUNC].tolist()
    )
    func_ids = [f for f in func_ids if f in valid_funcs]

    # 청크 분할
    chunk_size  = max(1, len(func_ids) // n_chunks)
    start_idx   = task_id * chunk_size
    end_idx     = start_idx + chunk_size if task_id < n_chunks - 1 else len(func_ids)
    chunk_funcs = func_ids[start_idx:end_idx]

    log.info(
        f"Array task {task_id}/{n_chunks-1} | "
        f"{len(chunk_funcs)} functions (idx {start_idx}~{end_idx-1})"
    )

    chunk_df = df[df[COL_FUNC].isin(chunk_funcs)].copy()
    all_ts   = []

    for fid in chunk_funcs:
        group = chunk_df[chunk_df[COL_FUNC] == fid].copy()
        if len(group) < 2:
            continue
        ts       = to_uniform_grid(group, freq)
        ts[COL_FUNC] = fid
        all_ts.append(ts)

    if not all_ts:
        log.warning(f"Chunk {task_id}: no valid functions")
        return

    result = pd.concat(all_ts, ignore_index=True)
    out = save_dir / "chunks" / f"chunk_{task_id:04d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    log.info(f"Chunk {task_id} saved → {out}  ({len(result):,} rows, {len(all_ts)} funcs)")


# ─── 메인 ─────────────────────────────────────────────────────────
def main(args):
    load_dir = Path(args.load_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df      = pd.read_parquet(load_dir / "trace_features.parquet")
    summary = pd.read_csv(load_dir / "function_summary.csv")
    log.info(f"Loaded {len(df):,} rows, {df[COL_FUNC].nunique():,} functions")

    func_ids = sorted(df[COL_FUNC].unique().tolist())

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

    if task_id >= 0:
        process_chunk(df, summary, func_ids, task_id, args.n_chunks,
                      save_dir, args.freq, args.min_invocations)
    else:
        # 단일 실행 모드
        log.info("Single-node mode")
        valid = set(
            summary.loc[summary["num_invocations"] >= args.min_invocations, COL_FUNC]
        )
        log.info(f"Valid functions: {len(valid)}")

        all_ts = []
        for fid, group in df[df[COL_FUNC].isin(valid)].groupby(COL_FUNC):
            ts          = to_uniform_grid(group, args.freq)
            ts[COL_FUNC]= fid
            all_ts.append(ts)

        result = pd.concat(all_ts, ignore_index=True)
        out    = save_dir / f"timeseries_{args.freq}.parquet"
        result.to_parquet(out, index=False)
        log.info(f"Saved {len(all_ts)} functions, {len(result):,} rows → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir",        default=str(DATA_STEP2))
    parser.add_argument("--save_dir",        default=str(DATA_STEP3))
    parser.add_argument("--freq",            default=FREQ)
    parser.add_argument("--min_invocations", type=int, default=MIN_INVOCATIONS)
    parser.add_argument("--n_chunks",        type=int, default=50)
    main(parser.parse_args())

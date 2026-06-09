"""
preprocess/step2_compute_features.py
─────────────────────────────────────
[STEP 2]  피처 계산 (Slurm Array Job 지원)

계산 피처:
  inter_arrival_sec  현재 → 다음 호출까지의 간격 (초)
  duration_sec       duration ms → 초 변환
  concurrency        실행 구간 겹침 기반 동시 호출 수 (Sweep Line)

Slurm Array Job 동작 방식:
  - 전체 함수를 N_CHUNKS 개 청크로 분할
  - $SLURM_ARRAY_TASK_ID (0-based) 에 해당하는 청크만 처리
  - 각 청크 결과를 chunk_{task_id}.parquet 으로 저장
  - step2_merge.py 가 전체 청크를 병합

단일 실행 (array 없이):
  python preprocess/step2_compute_features.py   # 전체 처리
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
    DATA_STEP1, DATA_STEP2,
    COL_DATE, COL_DUR, COL_FUNC,
    FEATURE_COLS, OUTLIER_Q,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 1. duration 단위 변환 ────────────────────────────────────────
def convert_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_sec"] = df[COL_DUR] / 1000.0
    return df


# ─── 2. inter-arrival time ────────────────────────────────────────
def compute_inter_arrival(df: pd.DataFrame) -> pd.DataFrame:
    """
    함수별 정렬 후 다음 호출까지의 간격(초) 계산.
    마지막 호출은 IAT = NaN → 이후 단계에서 제거.
    """
    df = df.sort_values([COL_FUNC, COL_DATE]).copy()
    df["inter_arrival_sec"] = (
        df.groupby(COL_FUNC)[COL_DATE]
          .diff()
          .dt.total_seconds()
          .shift(-1)
    )
    return df


# ─── 3. concurrency (Sweep Line) ─────────────────────────────────
def _sweep_line(group: pd.DataFrame) -> pd.DataFrame:
    """
    단일 함수 그룹에 대해 Sweep Line으로 concurrency 계산.
    각 호출의 [start, start+duration_sec) 구간에
    동시 실행 중인 호출 수(자기 자신 포함)를 반환.
    """
    group = group.sort_values(COL_DATE).copy()
    n = len(group)

    starts_ns = group[COL_DATE].astype(np.int64).values
    durs_ns   = (group["duration_sec"].values * 1e9).astype(np.int64)
    ends_ns   = starts_ns + durs_ns

    # 이벤트 생성: (time, type, idx)
    # 같은 시각에서 종료(-1)를 시작(+1)보다 먼저 처리
    events = []
    for i in range(n):
        events.append((starts_ns[i], +1, i))
        events.append((ends_ns[i],   -1, i))
    events.sort(key=lambda x: (x[0], x[1]))

    active       = 0
    start_active = {}
    for t, etype, idx in events:
        if etype == +1:
            active += 1
            start_active[idx] = active
        else:
            active -= 1

    group["concurrency"] = [start_active.get(i, 1) for i in range(n)]
    return group


def compute_concurrency(df: pd.DataFrame) -> pd.DataFrame:
    """전체 DataFrame에 대해 함수별로 Sweep Line을 적용합니다."""
    return pd.concat(
        [_sweep_line(g) for _, g in df.groupby(COL_FUNC)],
        ignore_index=True,
    )


# ─── 4. 이상치 clip ───────────────────────────────────────────────
def clip_outliers(df: pd.DataFrame, q: float = OUTLIER_Q) -> pd.DataFrame:
    df = df.copy()
    for col in ["duration_sec", "inter_arrival_sec"]:
        cap = df[col].quantile(q)
        n_clipped = (df[col] > cap).sum()
        df[col] = df[col].clip(upper=cap)
        log.info(f"  clip {col} at {cap:.4f}  ({n_clipped} values clipped)")
    return df


# ─── 5. 함수별 요약 통계 ─────────────────────────────────────────
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(COL_FUNC)
          .agg(
              num_invocations   = (COL_DATE,             "count"),
              mean_duration_sec = ("duration_sec",        "mean"),
              mean_iat_sec      = ("inter_arrival_sec",   "mean"),
              max_concurrency   = ("concurrency",         "max"),
              mean_concurrency  = ("concurrency",         "mean"),
          )
          .reset_index()
    )


# ─── 청크 처리 (Array Job 핵심) ───────────────────────────────────
def process_chunk(df: pd.DataFrame,
                  func_ids: list,
                  task_id: int,
                  n_chunks: int,
                  save_dir: Path) -> None:
    """
    func_ids 를 n_chunks 로 나눠 task_id 번째 청크만 처리 후 저장.
    """
    chunk_size  = max(1, len(func_ids) // n_chunks)
    start_idx   = task_id * chunk_size
    end_idx     = start_idx + chunk_size if task_id < n_chunks - 1 else len(func_ids)
    chunk_funcs = func_ids[start_idx:end_idx]

    log.info(
        f"Array task {task_id}/{n_chunks-1} | "
        f"functions {start_idx}~{end_idx-1} ({len(chunk_funcs)} funcs)"
    )

    chunk_df = df[df[COL_FUNC].isin(chunk_funcs)].copy()

    chunk_df = convert_duration(chunk_df)
    chunk_df = compute_inter_arrival(chunk_df)
    chunk_df = compute_concurrency(chunk_df)
    chunk_df = chunk_df.dropna(subset=["inter_arrival_sec"]).copy()
    chunk_df = clip_outliers(chunk_df)

    out = save_dir / "chunks" / f"chunk_{task_id:04d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_parquet(out, index=False)
    log.info(f"Chunk {task_id} saved → {out}  ({len(chunk_df):,} rows)")


# ─── 메인 ─────────────────────────────────────────────────────────
def main(args):
    load_dir = Path(args.load_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(load_dir / "trace_loaded.parquet")
    log.info(f"Loaded {len(df):,} rows, {df[COL_FUNC].nunique():,} functions")

    func_ids = sorted(df[COL_FUNC].unique().tolist())

    # ── Slurm Array Job 분기 ──────────────────────────────────────
    # $SLURM_ARRAY_TASK_ID 가 설정돼 있으면 청크 모드
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

    if task_id >= 0:
        # Array Job 모드: 지정된 청크만 처리
        process_chunk(df, func_ids, task_id, args.n_chunks, save_dir)

    else:
        # 단일 실행 모드: 전체 처리
        log.info("Running in single-node mode (no SLURM_ARRAY_TASK_ID)")
        df = convert_duration(df)
        df = compute_inter_arrival(df)
        df = compute_concurrency(df)
        df = df.dropna(subset=["inter_arrival_sec"]).copy()
        df = clip_outliers(df)

        summary = summarize(df)
        log.info(f"\nTop 10 functions by invocations:\n"
                 f"{summary.sort_values('num_invocations', ascending=False).head(10).to_string()}")

        df.to_parquet(save_dir / "trace_features.parquet", index=False)
        summary.to_csv(save_dir / "function_summary.csv", index=False)
        log.info(f"Saved → {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir",  default=str(DATA_STEP1))
    parser.add_argument("--save_dir",  default=str(DATA_STEP2))
    parser.add_argument("--n_chunks",  type=int, default=50,
                        help="Number of array job chunks (= --array=0-{n_chunks-1})")
    main(parser.parse_args())

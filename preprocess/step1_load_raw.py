"""
preprocess/step1_load_raw.py
────────────────────────────
[STEP 1]  원본 CSV 로드 및 기본 정제

  - data/raw/*.csv 를 전부 읽어 병합
  - date 파싱, 타입 보정, 유효하지 않은 행 제거
  - data/processed/01_loaded/trace_loaded.parquet 저장

Slurm 실행:
  sbatch scripts/step1_load_raw.sh
  또는 파이프라인 전체: bash scripts/submit_pipeline.sh
"""

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 sys.path에 추가 (Slurm에서 PYTHONPATH 설정 전 안전장치)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_RAW, DATA_STEP1,
    COL_DATE, COL_DUR, COL_WEEKDAY, COL_FUNC,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def load_single(filepath: Path) -> pd.DataFrame:
    log.info(f"  Reading {filepath.name}")
    df = pd.read_csv(filepath)

    # date 파싱 ("2021-01-31 00:00:00" 형식)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    # 타입 보정
    df[COL_DUR]     = pd.to_numeric(df[COL_DUR],     errors="coerce")
    df[COL_WEEKDAY] = pd.to_numeric(df[COL_WEEKDAY], errors="coerce").astype("Int64")
    df[COL_FUNC]    = pd.to_numeric(df[COL_FUNC],    errors="coerce").astype("Int64")

    # 유효하지 않은 행 제거
    before = len(df)
    df = df.dropna(subset=[COL_DATE, COL_DUR, COL_FUNC])
    df = df[df[COL_DUR] > 0]
    dropped = before - len(df)
    if dropped:
        log.warning(f"    Dropped {dropped} invalid rows")

    return df.sort_values([COL_FUNC, COL_DATE]).reset_index(drop=True)


def load_all(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    log.info(f"Found {len(csv_files)} CSV file(s) in {raw_dir}")

    df = pd.concat([load_single(f) for f in csv_files], ignore_index=True)
    df = df.sort_values([COL_FUNC, COL_DATE]).reset_index(drop=True)

    log.info(
        f"Merged → {len(df):,} rows | "
        f"{df[COL_FUNC].nunique():,} functions | "
        f"{df[COL_DATE].min()} ~ {df[COL_DATE].max()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────
def main(args):
    raw_dir  = Path(args.raw_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(raw_dir)

    out = save_dir / "trace_loaded.parquet"
    df.to_parquet(out, index=False)
    log.info(f"Saved → {out}")

    # Slurm 로그용 간단 요약
    log.info("── dtypes ──")
    for col, dtype in df.dtypes.items():
        log.info(f"  {col}: {dtype}")
    log.info("── describe ──")
    log.info(f"\n{df[[COL_DUR]].describe().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",  default=str(DATA_RAW))
    parser.add_argument("--save_dir", default=str(DATA_STEP1))
    main(parser.parse_args())

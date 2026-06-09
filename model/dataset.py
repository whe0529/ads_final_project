"""
model/dataset.py
────────────────
전처리 산출물(step4)을 Moirai 학습용 윈도우 샘플로 변환하는 PyTorch Dataset.

핵심 개념 — 함수 = 변수(variate):
  Moirai는 multivariate 입력을 받습니다. 우리는 "여러 함수를 같은 시간축에
  변수로 쌓아" 하나의 multivariate 시계열로 모델에 넣습니다.
  이렇게 해야 Any-variate attention이 함수 간 연쇄 호출 관계를 학습합니다.

  각 함수는 3개 채널(inter_arrival_sec, duration_sec, concurrency)을 가지므로,
  V개 함수를 묶으면 변수 차원은 V*3 이 됩니다.

윈도우 구성:
  [-------- context_length --------][-- prediction_length --]
   과거 관측치 (모델이 봄)            미래 (모델이 예측)

  하나의 multivariate 시계열에서 슬라이딩 윈도우로 여러 샘플을 만듭니다.

반환 텐서 (한 샘플):
  target          : (V*3, context+pred)   전체 시계열 값
  observed_mask   : (V*3, context+pred)   관측 여부 (1=관측, 0=결측/패딩)
  past/future 분리는 collate 또는 모델 내부에서 처리.
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_STEP4, COL_FUNC, FREQ, FEATURE_COLS,
    CONTEXT_LENGTH, PREDICTION_LENGTH,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def load_wide_array(parquet_path: Path,
                    feature_cols: list[str] = FEATURE_COLS,
                   ) -> tuple[np.ndarray, list[int], pd.DatetimeIndex]:
    """
    long-format parquet (timestamp, features..., app_func_id)을
    wide multivariate 배열로 변환합니다.

    Returns:
        arr        : (V*C, T)  V=함수 수, C=채널 수(3), T=타임스텝
        func_ids   : 길이 V 리스트 (변수 순서)
        time_index : 길이 T DatetimeIndex (공통 시간축)
    """
    df = pd.read_parquet(parquet_path)

    # 모든 함수가 공유하는 공통 시간축 구성
    # (함수마다 시작/끝이 다를 수 있으므로 합집합으로 reindex)
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    time_index = pd.date_range(t_min, t_max, freq=FREQ)
    T = len(time_index)

    func_ids = sorted(df[COL_FUNC].unique().tolist())
    C        = len(feature_cols)
    V        = len(func_ids)

    arr = np.zeros((V * C, T), dtype=np.float32)

    for vi, fid in enumerate(func_ids):
        sub = df[df[COL_FUNC] == fid].set_index("timestamp")
        sub = sub.reindex(time_index, fill_value=0.0)
        for ci, col in enumerate(feature_cols):
            arr[vi * C + ci, :] = sub[col].values.astype(np.float32)

    log.info(f"Wide array: {arr.shape} (V={V} funcs × C={C} channels, T={T})")
    return arr, func_ids, time_index


# ─────────────────────────────────────────────────────────────────
class FaaSWindowDataset(Dataset):
    """
    wide multivariate 배열에서 슬라이딩 윈도우 샘플을 생성합니다.

    Args:
        arr            : (D, T)  D=V*C 변수, T 타임스텝
        context_length : 과거 길이
        prediction_length : 예측 길이
        stride         : 윈도우 이동 간격 (작을수록 샘플 多)
        split          : "train" | "val" | "test"  — 시간순 구간 선택
    """

    def __init__(self,
                 arr: np.ndarray,
                 context_length: int = CONTEXT_LENGTH,
                 prediction_length: int = PREDICTION_LENGTH,
                 stride: int = None,
                 split: str = "train",
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1):
        super().__init__()
        self.ctx  = context_length
        self.pred = prediction_length
        self.win  = context_length + prediction_length
        self.stride = stride or max(1, prediction_length)

        D, T = arr.shape

        # 시간순 split 경계
        test_start = int(T * (1 - test_ratio))
        val_start  = int(T * (1 - test_ratio - val_ratio))

        if split == "train":
            self.data = arr[:, :val_start]
        elif split == "val":
            # context가 train 끝에서 이어지도록 약간 겹쳐서 가져옴
            lo = max(0, val_start - context_length)
            self.data = arr[:, lo:test_start]
        elif split == "test":
            lo = max(0, test_start - context_length)
            self.data = arr[:, lo:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.D = D
        self._build_indices()
        log.info(f"[{split}] data={self.data.shape}  windows={len(self.starts)}")

    def _build_indices(self):
        T = self.data.shape[1]
        self.starts = list(range(0, max(1, T - self.win + 1), self.stride))
        if not self.starts:           # 시계열이 너무 짧으면 한 윈도우라도
            self.starts = [0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s   = self.starts[idx]
        e   = s + self.win
        win = self.data[:, s:e]                          # (D, win)

        # 시계열이 win보다 짧으면 좌측 zero-pad
        if win.shape[1] < self.win:
            pad = np.zeros((self.D, self.win - win.shape[1]), dtype=np.float32)
            win = np.concatenate([pad, win], axis=1)

        target        = torch.from_numpy(win).float()    # (D, win)
        # observed_mask: 값이 0이 아니거나(호출 발생) padding이 아닌 위치
        observed_mask = (target != 0).float()

        return {
            "target":        target,                      # (D, ctx+pred)
            "observed_mask": observed_mask,               # (D, ctx+pred)
            "context_length":    self.ctx,
            "prediction_length": self.pred,
        }


# ─────────────────────────────────────────────────────────────────
def collate_fn(batch: list[dict]) -> dict:
    """
    배치 내 샘플의 target/mask를 스택합니다.
    (모든 샘플의 D, win이 동일하므로 단순 stack)
    """
    return {
        "target":        torch.stack([b["target"]        for b in batch]),  # (B, D, win)
        "observed_mask": torch.stack([b["observed_mask"] for b in batch]),  # (B, D, win)
        "context_length":    batch[0]["context_length"],
        "prediction_length": batch[0]["prediction_length"],
    }


# ─────────────────────────────────────────────────────────────────
def load_scalers(scaler_path: Path = None) -> dict:
    """
    step4에서 저장한 함수별 RobustScaler를 로드합니다.

    pickle은 저장 당시의 클래스 모듈 경로를 기억합니다. step4를 단독 실행
    (python preprocess/step4_normalize.py)했다면 클래스가 '__main__.RobustScaler'
    로 저장됐을 수 있고, 모듈 import로 실행했다면
    'preprocess.step4_normalize.RobustScaler' 로 저장됐을 수 있습니다.
    어느 경우든 풀 수 있도록 __main__ 네임스페이스에 클래스를 등록해 둡니다.
    """
    import sys as _sys
    from preprocess.step4_normalize import RobustScaler as _RS

    # '__main__.RobustScaler' 로 저장된 경우 대비
    main_mod = _sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "RobustScaler"):
        setattr(main_mod, "RobustScaler", _RS)

    scaler_path = scaler_path or (DATA_STEP4 / "scalers.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s  %(message)s")

    arr, func_ids, tidx = load_wide_array(DATA_STEP4 / f"timeseries_norm_{FREQ}.parquet")

    ds = FaaSWindowDataset(arr, split="train")
    sample = ds[0]
    print("target:",        sample["target"].shape)
    print("observed_mask:", sample["observed_mask"].shape)
    print("num windows:",   len(ds))
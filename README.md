# FaaS Cold Start — Preprocessing Pipeline (Slurm)

Azure Functions Invocation Trace 2021 데이터를 Moirai 시계열 예측 모델 입력 형식으로
전처리하는 Slurm 기반 파이프라인입니다.

## 목표

함수별로 세 가지 시계열을 예측해 Cold Start를 사전 완화합니다.

| 예측 변수 | 의미 | 컨테이너 결정 |
|-----------|------|---------------|
| `inter_arrival_sec` | 다음 호출까지 간격 | **언제** 띄울지 |
| `duration_sec` | 실행 지속 시간 | **얼마나** 유지할지 |
| `concurrency` | 동시 호출 수 | **몇 개** 띄울지 |

## 입력 데이터 스키마

```
date                 datetime   "2021-01-31 00:00:00"
duration             float      호출 실행 시간 (ms)
weekday              int        0=월 ~ 6=일
app_func_id_encoded  int        함수 식별자 (정수 인코딩)
```

`data/raw/` 아래에 CSV 파일을 넣으면 됩니다 (여러 파일이면 자동 병합).

## 디렉터리 구조

```
faas_cold_start/
├── config/
│   └── settings.py                  # 공통 상수 (컬럼명, FREQ, 경로 등)
├── preprocess/
│   ├── step1_load_raw.py            # CSV 로드 & 정제
│   ├── step2_compute_features.py    # IAT / duration / concurrency
│   ├── step3_build_timeseries.py    # 균일 그리드 변환
│   └── step4_normalize.py           # RobustScaler + patching 검증
├── scripts/
│   ├── step1_load_raw.sh            # 각 단계 SBATCH 스크립트
│   ├── step2_compute_features.sh
│   ├── step3_build_timeseries.sh
│   ├── step4_normalize.sh
│   └── submit_pipeline.sh           # ★ 전체 파이프라인 dependency 체인 제출
├── data/
│   ├── raw/                         # 원본 CSV (사용자가 넣음)
│   └── processed/                   # 단계별 산출물
├── logs/
└── environment.yml
```

## 실행 방법

### 1. 환경 구성

```bash
conda env create -f environment.yml
conda activate faas_env
```

> 각 `.sh` 스크립트 상단의 `source activate faas_env` 를 환경에 맞게 수정하세요.

### 2. 전체 파이프라인 제출 (권장)

```bash
bash scripts/submit_pipeline.sh
```

각 단계가 Slurm job dependency로 자동 연결됩니다:

```
step1 → step2 → step3 → step4
```

각 단계는 선행 job이 **성공(afterok)** 해야만 실행됩니다.
중간 단계가 실패하면 이후 단계는 자동으로 취소돼요.

### 3. 개별 단계 실행

```bash
sbatch scripts/step1_load_raw.sh
sbatch scripts/step2_compute_features.sh
```

### 4. 단일 노드(디버깅용, Slurm 없이)

```bash
export PYTHONPATH=$(pwd)
python preprocess/step1_load_raw.py
python preprocess/step2_compute_features.py
python preprocess/step3_build_timeseries.py
python preprocess/step4_normalize.py
```

## 처리 방식

모든 함수는 **하나의 job 안에서** `groupby` 루프로 순차 처리합니다.
함수마다 독립적으로:

1. inter-arrival time, duration, concurrency 계산
2. 균일 시간 그리드로 변환 (호출 없는 타임스텝은 0)
3. 함수별 RobustScaler로 정규화

함수별로 시계열을 분리하는 이유: 컨테이너는 함수마다 따로 띄우므로
예측도 함수 단위로 나와야 합니다. 분리된 시계열들은 같은 시간축에 정렬되어
Moirai의 **multivariate(다변량)** 입력 — 각 함수가 하나의 변수(채널) — 이 됩니다.
이를 통해 Moirai의 Any-variate attention이 함수 간 연쇄 호출 관계까지 학습합니다.

## 모니터링

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
tail -f /nas2/data/suaveh97/logs/slurm-<JOBID>.out
```

## 최종 산출물

```
data/processed/04_normalized/
├── timeseries_norm_1min.parquet   # 정규화된 멀티변량 시계열
└── scalers.pkl                    # 함수별 RobustScaler (추론 시 역변환용)
```

다음 단계는 이 산출물로 **Moirai + LoRA fine-tuning** (`model/train_lora.py`).

## 주요 설계 결정

- **Concurrency = 실행 구간 겹침** (Sweep Line). 단순히 "같은 초 시작"이 아니라
  `[start, start+duration)` 이 겹치는 호출 수로 계산 → 실제 필요한 컨테이너 수에 더 정확.
- **시간 해상도 1분** (`FREQ`). Edge/IoT 시나리오면 `10s`/`1s` 로 조정 가능.
- **함수별 독립 RobustScaler**. FaaS trace는 함수마다 스케일이 크게 다르고 outlier가 많음.
- **시간순 train/val/test 분할** (랜덤 금지). 시계열 누수 방지.
- **context_length=512**: Moirai의 모든 patch size(8~128)와 호환.

## Slurm 설정

스크립트의 SBATCH 헤더는 다음 클러스터 설정에 맞춰져 있습니다:

```
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 1-0
#SBATCH -o /nas2/data/suaveh97/logs/slurm-%A.out
```

전처리 단계는 CPU 연산이므로 GPU가 필수는 아니지만, CPU 전용 파티션이 없다면
`batch_grad`에서 GPU 1개를 할당받아 실행합니다. CPU 전용 파티션이 있다면
`--gres`/`--mem-per-gpu` 를 `--cpus-per-task`/`--mem` 으로 바꾸세요.

---

# 모델: Moirai + LoRA

## 구조 — 단일 LoRA 어댑터

Moirai backbone은 **freeze(❄️)** 하고, FaaS trace 도메인 전체에 적응하는
**LoRA 어댑터 1개(🔥)** 만 학습합니다. 함수별 어댑터를 만들지 않습니다.

```
                    ┌─ func 171 (3 channels) ─┐
  [Moirai backbone] │  func 188 (3 channels)  │ ← 함수 = 변수(variate)
   ❄️ frozen        │  func 189 (3 channels)  │   같은 시간축에 쌓아
        +           └─ ...                    ┘   multivariate 입력
  [LoRA adapter ×1] 🔥 ← FaaS 도메인 전체에 학습
```

이유:
- **Zero-Shot 유지**: 신규 함수가 배포돼도 어댑터 재학습 없이 새 변수로 입력만 하면 예측 가능 (제안서 동기 ③).
- **함수 간 종속성 학습**: 여러 함수가 한 모델에 동시 입력돼야 Any-variate attention이 연쇄 호출 관계를 학습 (제안서 동기 ②).
- **관리 용이**: 어댑터 1개만 저장/배포.

## 모델 디렉터리

```
model/
├── dataset.py        # 전처리 산출물 → wide multivariate → 윈도우 Dataset
├── moirai_lora.py    # Moirai 로드 + backbone freeze + LoRA 주입
├── train_lora.py     # 학습 (NLL 손실, LoRA만 업데이트, best 어댑터 저장)
└── inference.py      # 예측 + 역정규화 + 컨테이너 스케줄 생성
eval/
└── metrics.py        # MSE/MAE/MAPE + concurrency under/over-provision 분석
```

## 실행

### 전처리만
```bash
bash scripts/submit_pipeline.sh
```

### 전체 (전처리 → 학습 → 추론 → 평가)
```bash
bash scripts/submit_all.sh
```

체인: `step1 → step2 → step3 → step4 → train → inference → evaluate`

### 개별 (GPU)
```bash
sbatch scripts/train_lora.sh       # checkpoints/best, checkpoints/last
sbatch scripts/inference.sh        # predictions/predictions.npz, container_schedule.csv
sbatch scripts/evaluate.sh         # predictions/metrics.json
```

## 산출물

```
checkpoints/best/           # 최저 val 손실 LoRA 어댑터 (배포용)
predictions/
├── predictions.npz         # 역정규화된 예측/실제값
├── container_schedule.csv  # 함수별 (언제, 몇 개, 얼마나) 컨테이너 결정
└── metrics.json            # MSE/MAE/MAPE + concurrency 정확도
```

`container_schedule.csv` 컬럼:
- `next_invocation_in_sec`: 다음 호출까지 예상 시간 → 컨테이너 **시작 시점**
- `hold_duration_sec`: 예상 실행 시간 → 컨테이너 **유지 시간**
- `num_containers`: 예상 동시 호출 수 → 띄울 컨테이너 **개수**

## 평가 지표

제안서 Metric(MSE/MAE/MAPE)을 채널별로 계산하고,
컨테이너 수(concurrency)는 운영 관점 지표를 추가로 봅니다:
- **under-provision rate**: 예측 < 실제 → Cold Start 발생 위험
- **over-provision rate**: 예측 > 실제 → 자원 낭비

이 둘의 trade-off가 제안서 Expected Contribution의 "Cold Start ↔ 자원 낭비 균형"을 직접 측정합니다.

## ⚠️ uni2ts 버전 주의

uni2ts는 학습 API가 버전마다 달라집니다. `train_lora.py` 의 `_compute_loss()` 와
`moirai_lora.py` 의 `find_lora_target_modules()` 는 버전 차이를 흡수하도록 작성했지만,
설치된 uni2ts 버전에 따라 다음을 확인하세요:

1. `MoiraiForecast` 의 생성자 인자 (`patch_size="auto"` vs 고정값)
2. LoRA 주입 대상 모듈 이름 — 로그의 "LoRA target modules detected" 확인
3. 손실 계산 경로 — `model.loss(...)` 존재 여부에 따라 자동 분기

먼저 작은 설정(small 모델, 적은 epoch)으로 1 step 돌려서 shape/loss가 정상인지 확인 후 본 학습을 권장합니다.

"""
model/moirai_lora.py
────────────────────
Moirai foundation 모델에 단일 LoRA 어댑터를 주입합니다.

설계:
  - Moirai backbone(transformer encoder)은 freeze (❄️).
  - LoRA 어댑터 1개만 학습 (🔥) → FaaS trace 도메인 전체에 적응.
  - 함수별 어댑터를 만들지 않음. 함수 구분은 Moirai의 variate 차원에서 처리.
    → 신규 함수도 같은 어댑터로 Zero-Shot 예측 가능 (제안서 동기 유지).

LoRA 주입 대상:
  transformer의 attention projection (q/k/v/out_proj).
  peft 라이브러리의 LoraConfig + get_peft_model 사용.

참고: uni2ts 버전에 따라 내부 모듈 경로/이름이 다를 수 있어
      target_modules는 런타임에 자동 탐지하는 헬퍼도 제공합니다.
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MOIRAI_HF_ID, MOIRAI_SIZE,
    CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE, NUM_VARIATES,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def build_moirai_forecast(prediction_length: int = PREDICTION_LENGTH,
                          context_length: int = CONTEXT_LENGTH,
                          patch_size=PATCH_SIZE,
                          target_dim: int = NUM_VARIATES,
                          ):
    """
    사전학습된 MoiraiForecast 모듈을 로드합니다.

    target_dim: 한 번에 입력하는 변수 수.
      여기서는 채널 수(3)를 기본으로 두되, 실제로는 함수 묶음 단위로
      train 루프에서 동적으로 설정할 수 있습니다.
    """
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    module = MoiraiModule.from_pretrained(MOIRAI_HF_ID)

    model = MoiraiForecast(
        module             = module,
        prediction_length  = prediction_length,
        context_length     = context_length,
        patch_size         = patch_size,
        num_samples        = 100,        # probabilistic forecast 샘플 수
        target_dim         = target_dim,
        feat_dynamic_real_dim      = 0,
        past_feat_dynamic_real_dim = 0,
    )
    log.info(f"Loaded MoiraiForecast: {MOIRAI_HF_ID} "
             f"(ctx={context_length}, pred={prediction_length}, patch={patch_size})")
    return model


# ─────────────────────────────────────────────────────────────────
def find_lora_target_modules(model: nn.Module,
                             keywords: list[str] = LORA_TARGETS,
                             ) -> list[str]:
    """
    모델 내부에서 LoRA를 붙일 nn.Linear 모듈 이름을 자동 탐지합니다.
    uni2ts 버전에 따라 정확한 이름이 다를 수 있어, keyword 매칭으로 찾습니다.
    """
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for kw in keywords:
                if kw in name:
                    # peft는 보통 leaf 이름(마지막 토큰)을 기대
                    names.add(name.split(".")[-1])
    found = sorted(names)
    log.info(f"LoRA target modules detected: {found}")
    if not found:
        log.warning("No target modules matched keywords. "
                    "Falling back to all nn.Linear leaf names.")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                names.add(name.split(".")[-1])
        found = sorted(names)
    return found


# ─────────────────────────────────────────────────────────────────
def inject_lora(model: nn.Module,
                r: int = LORA_R,
                alpha: int = LORA_ALPHA,
                dropout: float = LORA_DROPOUT,
                target_modules: list[str] = None,
                ) -> nn.Module:
    """
    Moirai backbone을 freeze하고 LoRA 어댑터를 주입합니다.
    peft.get_peft_model 사용.
    """
    from peft import LoraConfig, get_peft_model

    # 1) 전체 freeze
    for p in model.parameters():
        p.requires_grad = False

    # 2) target 모듈 탐지
    if target_modules is None:
        target_modules = find_lora_target_modules(model)

    # 3) LoRA 주입
    lora_cfg = LoraConfig(
        r              = r,
        lora_alpha     = alpha,
        lora_dropout   = dropout,
        target_modules = target_modules,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)

    # 학습 가능 파라미터 비율 출력
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA injected | trainable {trainable:,} / total {total:,} "
             f"({100*trainable/total:.3f}%)")

    return model


# ─────────────────────────────────────────────────────────────────
def save_lora_adapter(model: nn.Module, save_dir: Path) -> None:
    """LoRA 어댑터 가중치만 저장 (backbone 제외 → 용량 작음)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    log.info(f"LoRA adapter saved → {save_dir}")


def load_lora_adapter(base_model: nn.Module, adapter_dir: Path) -> nn.Module:
    """저장된 LoRA 어댑터를 base 모델에 로드합니다."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    log.info(f"LoRA adapter loaded ← {adapter_dir}")
    return model

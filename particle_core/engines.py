"""분할 엔진 Protocol 과 WatershedEngine.

HybridAIEngine(U-Net x YOLO) 은 app 측(engines_ai.py)에서 이 Protocol 을 구현한다.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .params import AnalysisParams, WatershedParams
from .segment import SegmentationResult, segment_watershed


@runtime_checkable
class SegmentationEngine(Protocol):
    name: str

    def prepare(self) -> None:
        """모델 로드 등 1회 준비 작업."""
        ...

    def segment(self, gray_pre: np.ndarray) -> SegmentationResult:
        """전처리된 2D uint8 영상 → 라벨맵."""
        ...


class WatershedEngine:
    """고전 watershed 엔진 (상태 없음)."""

    name = "watershed"

    def __init__(self, ws: WatershedParams) -> None:
        self.ws = ws

    def prepare(self) -> None:
        return None

    def segment(self, gray_pre: np.ndarray) -> SegmentationResult:
        return segment_watershed(gray_pre, self.ws)


def make_engine(params: AnalysisParams) -> SegmentationEngine:
    """engine == 'watershed' 만 처리. 'ai_hybrid' 는 app 측에서 생성해야 하므로 NotImplementedError."""
    eng = (params.engine or "watershed").lower()
    if eng == "watershed":
        return WatershedEngine(params.watershed)
    if eng == "ai_hybrid":
        raise NotImplementedError("ai_hybrid engine is provided by the app layer (engines_ai.HybridAIEngine)")
    raise ValueError(f"Unknown engine: {params.engine}")


__all__ = ["SegmentationEngine", "WatershedEngine", "make_engine"]

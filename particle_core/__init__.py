"""particle_core — SEM 입자 분석 헤드리스 코어 패키지.

PyQt 의존 없이 분할(segmentation) → 특성 추출 → 필터 → 통계 → Excel/PPT 를 수행한다.
matplotlib 는 반드시 Agg 백엔드로 고정한다 (서버/스레드 환경).
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # 다른 어떤 import 보다 먼저

from .params import (  # noqa: E402
    AfterFilterParams,
    AnalysisParams,
    HybridEngineParams,
    MeasureParams,
    OutputParams,
    ScaleParams,
    SemParams,
    SpatialParams,
    WatershedParams,
)
from .segment import SegmentationResult, segment_watershed  # noqa: E402
from .engines import SegmentationEngine, WatershedEngine, make_engine  # noqa: E402
from .pipeline import BatchInput, BatchResult, ImageResult, run_batch  # noqa: E402

logging.getLogger("particle_core").addHandler(logging.NullHandler())

__all__ = [
    "AfterFilterParams",
    "AnalysisParams",
    "HybridEngineParams",
    "MeasureParams",
    "OutputParams",
    "ScaleParams",
    "SemParams",
    "SpatialParams",
    "WatershedParams",
    "SegmentationResult",
    "segment_watershed",
    "SegmentationEngine",
    "WatershedEngine",
    "make_engine",
    "BatchInput",
    "BatchResult",
    "ImageResult",
    "run_batch",
]

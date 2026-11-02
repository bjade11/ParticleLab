"""튜너 API — 단일 이미지 전처리 미리보기 / 분할 미리보기."""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from particle_core.params import AnalysisParams  # noqa: E402
from particle_core.io_utils import to_gray_u8, pixel_to_micro_from_name  # noqa: E402
from particle_core.preprocess import apply_sem_preprocessing, preview_preprocessing  # noqa: E402
from particle_core.engines import make_engine  # noqa: E402
from particle_core.measure import measure_particles  # noqa: E402
from particle_core.filters import apply_after_filter  # noqa: E402
from particle_core.stats import calculate_d_values  # noqa: E402
from particle_core import viz  # noqa: E402

from ..engines_ai import HybridAIEngine  # noqa: E402

router = APIRouter(prefix="/api/tuner", tags=["tuner"])

PARTICLE_COLS = [
    "Label", "x", "y", "Diameter (μm)", "Circularity", "Aspect Ratio",
    "Area (μm^2)", "Solidity", "Mean Intensity",
]


def parse_params(raw: str) -> AnalysisParams:
    try:
        d = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"params JSON 파싱 실패: {e}")
    try:
        return AnalysisParams.from_dict(d)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"params 검증 실패: {e}")


def read_gray_upload(upload: UploadFile) -> np.ndarray:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(400, f"빈 파일: {upload.filename}")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, f"이미지 디코딩 실패: {upload.filename}")
    return to_gray_u8(img)


def build_engine(params: AnalysisParams):
    if params.engine == "ai_hybrid":
        eng = HybridAIEngine(params.hybrid)
    else:
        eng = make_engine(params)
    return eng


def resolve_scale(params: AnalysisParams, filename: str) -> float:
    sc = params.scale
    if sc.mode == "auto":
        return float(pixel_to_micro_from_name(filename, sc.auto_default))
    return float(sc.pixel_to_micro)


def _records(df, cols: list[str], limit: int = 500) -> list[dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    use = [c for c in cols if c in df.columns]
    out = df[use].head(limit).copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    recs = out.to_dict(orient="records")
    clean = []
    for r in recs:
        clean.append({k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if isinstance(v, np.generic) else v))
                      for k, v in r.items()})
    return clean


def _d_values_safe(df) -> dict:
    try:
        if df is None or len(df) == 0:
            return {"n": 0}
        d = calculate_d_values(df)
        return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in d.items()
                if not isinstance(v, (np.ndarray, list))}
    except Exception as e:  # noqa: BLE001
        return {"n": int(len(df)) if df is not None else 0, "error": str(e)}


def _gray_to_png(gray: np.ndarray | None) -> str | None:
    if gray is None:
        return None
    return viz.encode_png_data_url(gray)


def _float_map_to_png(m: np.ndarray | None) -> str | None:
    if m is None:
        return None
    m = np.nan_to_num(m.astype(np.float32))
    mx = float(m.max()) if m.size else 0.0
    u8 = (m / mx * 255.0).astype(np.uint8) if mx > 0 else np.zeros(m.shape, np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
    return viz.encode_png_data_url(color)


def _markers_to_png(markers: np.ndarray | None, base_gray: np.ndarray) -> str | None:
    if markers is None:
        return None
    bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(markers > 0)
    for y, x in zip(ys, xs):
        cv2.circle(bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
    return viz.encode_png_data_url(bgr)


@router.post("/preprocess")
async def tuner_preprocess(
    image: UploadFile = File(...),
    params: str = Form("{}"),
):
    p = parse_params(params)
    gray = read_gray_upload(image)
    blur = tuple(p.watershed.blur_kernel) if p.engine == "watershed" else (1, 1)
    pre = preview_preprocessing(gray, p.sem, blur)
    return {"preview_png": viz.encode_png_data_url(pre), "shape": [int(pre.shape[0]), int(pre.shape[1])]}


@router.post("/segment")
async def tuner_segment(
    image: UploadFile = File(...),
    params: str = Form("{}"),
):
    p = parse_params(params)
    gray = read_gray_upload(image)
    filename = image.filename or "image.png"
    p2m = resolve_scale(p, filename)
    timing: dict[str, float] = {}

    t0 = time.time()
    gray_pre = apply_sem_preprocessing(gray, p.sem)
    timing["preprocess"] = (time.time() - t0) * 1000

    t0 = time.time()
    try:
        engine = build_engine(p)
        engine.prepare()
        seg = engine.segment(gray_pre)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"분할 실패 ({engine.__class__.__name__ if 'engine' in dir() else p.engine}): {e}")
    timing["segment"] = (time.time() - t0) * 1000

    t0 = time.time()
    df_before, contours_map = measure_particles(seg.labels, gray_pre, p2m, p.measure)
    df_after = apply_after_filter(df_before, p.after_filter)
    timing["measure"] = (time.time() - t0) * 1000

    labeled_before = viz.draw_labeled_image(df_before, seg.labels, gray_pre, contours_map)
    labeled_after = viz.draw_labeled_image(df_after, seg.labels, gray_pre, contours_map)
    contour_before = viz.draw_contour_only_image(df_before, seg.labels, gray_pre, contours_map)

    views = {
        "original": viz.encode_png_data_url(gray),
        "preprocessed": viz.encode_png_data_url(gray_pre),
        "binary": _gray_to_png((seg.binary.astype(np.uint8) * 255) if seg.binary is not None else None),
        "distance": _float_map_to_png(seg.distance),
        "markers": _markers_to_png(seg.markers, gray_pre),
        "labels": viz.encode_png_data_url(viz.colorize_labels(seg.labels)),
        "labeled_before": viz.encode_png_data_url(labeled_before),
        "labeled_after": viz.encode_png_data_url(labeled_after),
        "contour_before": viz.encode_png_data_url(contour_before),
    }
    if p.engine == "ai_hybrid" and getattr(engine, "last_prob", None) is not None:
        views["distance"] = _float_map_to_png(engine.last_prob)  # AI: U-Net 확률맵을 거리맵 슬롯에

    meta = getattr(engine, "last_meta", {}) or {}
    return {
        "engine": p.engine,
        "pixel_to_micro": p2m,
        "shape": [int(gray_pre.shape[0]), int(gray_pre.shape[1])],
        "n_before": int(len(df_before)),
        "n_after": int(len(df_after)),
        "n_labels": int(seg.n_labels),
        "engine_meta": meta,
        "views": views,
        "particles_before": _records(df_before, PARTICLE_COLS),
        "particles_after": _records(df_after, PARTICLE_COLS),
        "d_values_before": _d_values_safe(df_before),
        "d_values_after": _d_values_safe(df_after),
        "timing_ms": {k: round(v, 1) for k, v in timing.items()},
    }

"""라벨맵 → 입자 특성 DataFrame (원본 analyze_particles 620~745 포팅).

버그 수정: contours_map 을 반드시 채운다 (labels==label 의 최대 cv2 윤곽선, CHAIN_APPROX_SIMPLE).
원본에서 죽어 있던 circularity/aspect 필터는 제거 (실제 Before→After 필터는 filters.apply_after_filter).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from .params import MeasureParams

logger = logging.getLogger("particle_core")

COLUMNS: List[str] = [
    "Label", "Area (px^2)", "Area (μm^2)", "Perimeter (px)", "Perimeter (μm)", "Circularity",
    "Major Axis (px)", "Major Axis (μm)", "Minor Axis (px)", "Minor Axis (μm)", "Aspect Ratio", "Diameter (μm)",
    "Area_Ratio", "Perimeter_Ratio", "Radius_Ratio", "x", "y", "Mean Intensity", "Mean Curvature", "Solidity",
    "Equivalent_Diameter_Pixel", "Image_Width", "Image_Height",
]


def empty_particles_df() -> pd.DataFrame:
    """컬럼만 있는 빈 입자 DataFrame."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS})


# ----------------------------------------------------------------------------- 윤곽선 / 곡률
def extract_contour(labels: np.ndarray, lab: int, bbox: Optional[Tuple[int, int, int, int]] = None
                    ) -> Optional[np.ndarray]:
    """labels==lab 영역의 가장 큰 외곽 윤곽선 (N,1,2) int32. bbox=(minr,minc,maxr,maxc) 로 가속."""
    if bbox is not None:
        minr, minc, maxr, maxc = bbox
        # 경계 1px 패딩: 영상 가장자리에 닿는 입자도 닫힌 윤곽선이 되도록
        sub = (labels[minr:maxr, minc:maxc] == lab).astype(np.uint8)
        sub = np.pad(sub, 1)
        off = np.array([minc - 1, minr - 1], dtype=np.int32)
    else:
        sub = (labels == lab).astype(np.uint8)
        off = np.zeros(2, dtype=np.int32)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea) if len(contours) > 1 else contours[0]
    return (best + off).astype(np.int32)


def mean_curvature(contour: np.ndarray) -> float:
    """윤곽선 평균 곡률 (원본 mean_curvature 의 벡터화 버전)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 5:
        return 0.0
    v1 = pts - np.roll(pts, 1, axis=0)        # p_curr - p_prev
    v2 = np.roll(pts, -1, axis=0) - pts       # p_next - p_curr
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    ok = (l1 > 0) & (l2 > 0)
    if not ok.any():
        return 0.0
    v1n = v1[ok] / l1[ok, None]
    v2n = v2[ok] / l2[ok, None]
    cross = np.abs(v1n[:, 0] * v2n[:, 1] - v1n[:, 1] * v2n[:, 0])
    avg_len = (l1[ok] + l2[ok]) / 2.0
    good = avg_len > 0
    if not good.any():
        return 0.0
    return float(np.mean(cross[good] / avg_len[good]))


# ----------------------------------------------------------------------------- 원형성 지표
def calculate_inscribed_circle(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    """윤곽선 내접 최대원 (거리변환 기반)."""
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([x, y], dtype=np.int32)], 0, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return np.array([max_loc[0] + x, max_loc[1] + y], dtype=np.float64), float(max_val)


def calculate_circularity_metrics(contour: np.ndarray, pixel_to_micro: float) -> Dict[str, float]:
    """Circularity / Area_Ratio / Perimeter_Ratio / Radius_Ratio (원본 calculate_circularity_metrics)."""
    area = cv2.contourArea(contour) * (pixel_to_micro ** 2)
    perimeter = cv2.arcLength(contour, True) * pixel_to_micro
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    pts = contour.reshape(-1, 2).astype(np.float32)
    _, outer_r = cv2.minEnclosingCircle(pts)
    _, inner_r = calculate_inscribed_circle(contour)
    outer_area = np.pi * (outer_r * pixel_to_micro) ** 2
    outer_perim = 2 * np.pi * outer_r * pixel_to_micro
    return {
        "Circularity": float(circularity),
        "Area_Ratio": float(area / outer_area) if outer_area > 0 else 0.0,
        "Perimeter_Ratio": float(perimeter / outer_perim) if outer_perim > 0 else 0.0,
        "Radius_Ratio": float(inner_r / outer_r) if outer_r > 0 else 0.0,
    }


def _mean_intensity(gray: np.ndarray, contour: np.ndarray) -> float:
    """윤곽선 내부(채움) 평균 밝기 — bbox 크롭으로 계산."""
    x, y, w, h = cv2.boundingRect(contour)
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w + 1), min(H, y + h + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([x0, y0], dtype=np.int32)], -1, 255, -1)
    return float(cv2.mean(gray[y0:y1, x0:x1], mask=mask)[0])


# ----------------------------------------------------------------------------- 메인
def measure_particles(labels: np.ndarray, gray_for_intensity: np.ndarray, pixel_to_micro: float,
                      mp: MeasureParams) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """라벨맵에서 입자별 특성을 계산한다.

    Returns (df, contours_map). df 컬럼은 COLUMNS. contours_map[label] = (N,1,2) int32.
    gray_for_intensity 는 labels 와 같은 크기의 2D uint8.
    """
    if labels is None or labels.ndim != 2:
        raise ValueError("measure_particles: labels must be 2D")
    if gray_for_intensity is None or gray_for_intensity.shape[:2] != labels.shape[:2]:
        raise ValueError("measure_particles: gray_for_intensity must match labels shape")
    p2m = float(pixel_to_micro)
    img_h, img_w = labels.shape
    labels = labels.astype(np.int32, copy=False)

    rows: List[dict] = []
    contours_map: Dict[int, np.ndarray] = {}
    curvature_list: List[float] = []
    pending: List[tuple] = []

    for region in regionprops(labels):
        minr, minc, maxr, maxc = region.bbox
        if mp.filter_border_particles and (minr <= 1 or minc <= 1 or maxr >= img_h - 1 or maxc >= img_w - 1):
            continue
        contour = extract_contour(labels, int(region.label), region.bbox)
        if contour is None or len(contour) < int(mp.min_contour_points):
            continue
        curv = mean_curvature(contour) if len(contour) >= 5 else None
        if curv is not None:
            curvature_list.append(curv)
        pending.append((region, contour, curv))

    # 곡률 percentile 필터 / 직선 윤곽선 필터 임계값
    curv_thr = 0.0
    if mp.curvature_percentile and mp.curvature_percentile > 0 and curvature_list:
        curv_thr = float(np.percentile(curvature_list, mp.curvature_percentile))
    linear_cut = 0.0
    if mp.exclude_linear_contour and curvature_list and mp.linear_contour_percentile > 0:
        linear_cut = float(np.percentile(curvature_list, mp.linear_contour_percentile))

    for region, contour, curv in pending:
        if curv is not None and curv_thr > 0 and curv < curv_thr:
            continue
        if mp.exclude_linear_contour and curv is not None and curv < linear_cut:
            continue
        area = float(region.area)
        perimeter = float(region.perimeter)
        major = float(region.axis_major_length)
        minor = float(region.axis_minor_length)
        circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        diameter = float(np.sqrt(4 * area * (p2m ** 2) / np.pi))
        aspect = major / minor if minor != 0 else 0.0
        cy, cx = region.centroid
        metrics = {"Area_Ratio": 0.0, "Perimeter_Ratio": 0.0, "Radius_Ratio": 0.0}
        mean_int = float("nan")
        if len(contour) >= 5:
            try:
                metrics = calculate_circularity_metrics(contour, p2m)
            except Exception as e:  # noqa: BLE001
                logger.debug("circularity metrics failed for label %s: %s", region.label, e)
            mean_int = _mean_intensity(gray_for_intensity, contour)
        lab = int(region.label)
        contours_map[lab] = contour
        rows.append({
            "Label": lab,
            "Area (px^2)": area,
            "Area (μm^2)": area * (p2m ** 2),
            "Perimeter (px)": perimeter,
            "Perimeter (μm)": perimeter * p2m,
            "Circularity": float(circ),
            "Major Axis (px)": major,
            "Major Axis (μm)": major * p2m,
            "Minor Axis (px)": minor,
            "Minor Axis (μm)": minor * p2m,
            "Aspect Ratio": float(aspect),
            "Diameter (μm)": diameter,
            "Area_Ratio": metrics["Area_Ratio"],
            "Perimeter_Ratio": metrics["Perimeter_Ratio"],
            "Radius_Ratio": metrics["Radius_Ratio"],
            "x": float(cx),
            "y": float(cy),
            "Mean Intensity": mean_int,
            "Mean Curvature": float(curv) if curv is not None else float("nan"),
            "Solidity": float(region.solidity),
            "Equivalent_Diameter_Pixel": float(region.equivalent_diameter_area),
            "Image_Width": int(img_w),
            "Image_Height": int(img_h),
        })

    if not rows:
        return empty_particles_df(), {}
    df = pd.DataFrame(rows, columns=COLUMNS)

    # 직경(μm) 하한
    if mp.min_diameter and mp.min_diameter > 0:
        df = df[df["Diameter (μm)"] >= float(mp.min_diameter)]

    # 어두운 입자 필터: 밝기 하위 n% 제외
    if mp.exclude_dark_particle and not df.empty:
        vals = df["Mean Intensity"].dropna()
        if not vals.empty:
            thr = float(np.percentile(vals, mp.dark_particle_percentile))
            df = df[df["Mean Intensity"] > thr]

    df = df.reset_index(drop=True)
    keep = set(int(v) for v in df["Label"].tolist())
    contours_map = {k: v for k, v in contours_map.items() if k in keep}
    logger.debug("measure_particles: %d particles kept", len(df))
    return df, contours_map


__all__ = ["COLUMNS", "empty_particles_df", "measure_particles", "extract_contour", "mean_curvature",
           "calculate_circularity_metrics", "calculate_inscribed_circle"]

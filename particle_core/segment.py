"""Watershed 기반 분할 (원본 analyze_particles 429~599 포팅).

엣지강화 → GaussianBlur(1회) → Otsu(자동 반전) → 모폴로지 → EDT → 마커 → watershed
→ 작은 객체 제거(min_size 실제 적용) → 재라벨.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log, peak_local_max
from skimage.measure import label as sk_label
from skimage.segmentation import watershed

from .params import WatershedParams

logger = logging.getLogger("particle_core")


@dataclass
class SegmentationResult:
    labels: np.ndarray                  # int32 (H,W), 0=배경
    binary: Optional[np.ndarray]        # bool, watershed 만
    distance: Optional[np.ndarray]      # float, watershed 만
    markers: Optional[np.ndarray]       # int32, watershed 만
    n_labels: int


def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _remove_small_labels(labels: np.ndarray, min_size: int) -> np.ndarray:
    """면적 < min_size 인 라벨을 0 으로. (skimage.remove_small_objects(min_size) 와 동일 의미)"""
    if min_size <= 1 or labels.max() == 0:
        return labels
    counts = np.bincount(labels.ravel())
    small = counts < int(min_size)
    small[0] = False
    out = labels.copy()
    out[small[labels]] = 0
    return out


def segment_watershed(gray_pre: np.ndarray, ws: WatershedParams) -> SegmentationResult:
    """전처리된 2D uint8 영상을 watershed 로 분할한다."""
    if gray_pre is None or gray_pre.ndim != 2:
        raise ValueError("segment_watershed: gray_pre must be a 2D array")
    img = gray_pre
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 엣지 강화 ---
    if ws.enhance_edges:
        k = _odd(ws.edge.kernel)
        method = (ws.edge.method or "laplacian").lower()
        if method == "sobelx":
            edges = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        elif method == "sobely":
            edges = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        elif method == "scharr":
            edges = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        else:
            edges = cv2.Laplacian(img, cv2.CV_64F, ksize=k)
        img = cv2.convertScaleAbs(img.astype(np.float64) + float(ws.edge.weight) * edges)

    # --- 블러 1회 + Otsu ---
    kx, ky = _odd(ws.blur_kernel[0]), _odd(ws.blur_kernel[1])
    blurred = cv2.GaussianBlur(img, (kx, ky), 0) if (kx > 0 and ky > 0) else img
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 전경이 배경보다 어두우면 반전 (밝은 입자 가정)
    fg = img[binary == 255]
    bg = img[binary == 0]
    if fg.size > 0 and bg.size > 0 and fg.mean() < bg.mean():
        binary = cv2.bitwise_not(binary)

    # --- 모폴로지 (open → close → erode → dilate) ---
    m = ws.morphology
    if m is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((m.open.kernel, m.open.kernel), np.uint8),
                                  iterations=max(1, int(m.open.iterations)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((m.close.kernel, m.close.kernel), np.uint8),
                                  iterations=max(1, int(m.close.iterations)))
        binary = cv2.erode(binary, np.ones((m.erode.kernel, m.erode.kernel), np.uint8),
                           iterations=max(1, int(m.erode.iterations)))
        binary = cv2.dilate(binary, np.ones((m.dilate.kernel, m.dilate.kernel), np.uint8),
                            iterations=max(1, int(m.dilate.iterations)))

    binary_bool = binary > 0
    distance = ndi.distance_transform_edt(binary_bool)

    # --- 마커 ---
    mk = ws.marker
    method = ws.marker_method
    if method == "peak_local_max":
        coords = peak_local_max(distance, min_distance=max(1, int(mk.peak_min_distance)),
                                labels=binary_bool.astype(np.int32))
        local_max = np.zeros(distance.shape, dtype=bool)
        if len(coords) > 0:
            local_max[tuple(coords.T)] = True
        markers, _ = ndi.label(local_max)
    elif method == "dilate_threshold":
        thr = float(mk.distance_threshold) * float(distance.max())
        seed = (distance > thr).astype(np.uint8)
        seed = cv2.dilate(seed, np.ones((int(mk.dilate_kernel), int(mk.dilate_kernel)), np.uint8),
                          iterations=max(1, int(mk.dilate_iter)))
        markers, _ = ndi.label(seed)
    elif method == "erosion":
        eroded = cv2.erode(binary_bool.astype(np.uint8),
                           np.ones((int(mk.erosion_kernel), int(mk.erosion_kernel)), np.uint8),
                           iterations=max(1, int(mk.erosion_iter)))
        markers, _ = ndi.label(eroded)
    elif method == "blob_detection":
        blobs = blob_log(distance, min_sigma=float(mk.blob_min_sigma), max_sigma=float(mk.blob_max_sigma),
                         num_sigma=int(mk.blob_num_sigma), threshold=float(mk.blob_threshold))
        seed = np.zeros(distance.shape, dtype=np.uint8)
        for b in blobs:
            y, x = int(round(b[0])), int(round(b[1]))
            if 0 <= y < seed.shape[0] and 0 <= x < seed.shape[1]:
                seed[y, x] = 1
        markers, _ = ndi.label(seed)
    else:
        raise ValueError(f"Unknown marker_method: {method}")

    markers = markers.astype(np.int32)

    # --- watershed → 작은 객체 제거 → 재라벨 ---
    labels = watershed(-distance, markers, mask=binary_bool)
    labels = _remove_small_labels(labels.astype(np.int32), int(ws.min_size))
    labels = sk_label(labels, connectivity=1).astype(np.int32)  # 0 유지, 1..N 으로 압축
    n = int(labels.max())
    logger.debug("segment_watershed: %d labels (markers=%d)", n, int(markers.max()))
    return SegmentationResult(labels=labels, binary=binary_bool, distance=distance.astype(np.float32),
                              markers=markers, n_labels=n)


__all__ = ["SegmentationResult", "segment_watershed"]

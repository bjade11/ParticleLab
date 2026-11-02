"""SEM 전처리: crop → brightness/contrast → gamma → CLAHE → unsharp mask.

원본과 달리 블러는 여기서 적용하지 않는다 (segment_watershed 가 1회 적용). 크롭도 여기서 1회만.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .io_utils import crop_bottom
from .params import SemParams


def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=float(contrast), beta=int(brightness))


def adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    gamma = max(float(gamma), 1e-6)
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)


def clarity_enhance(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(max(1, int(tile)), max(1, int(tile))))
    return clahe.apply(img)


def unsharp_mask(img: np.ndarray, radius: float = 1.5, amount: float = 0.6) -> np.ndarray:
    if amount <= 0 or radius <= 0:
        return img
    sigma = float(radius)
    k = int(2 * round(3 * sigma) + 1)
    blur = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def apply_sem_preprocessing(gray_u8: np.ndarray, sem: SemParams) -> np.ndarray:
    """크롭 포함 SEM 전처리. 블러는 포함하지 않는다. 입력은 2D uint8."""
    img = crop_bottom(gray_u8, sem.bottom_crop_ratio).copy()
    if int(sem.brightness) != 0 or float(sem.contrast) != 1.0:
        img = adjust_brightness_contrast(img, int(sem.brightness), float(sem.contrast))
    if float(sem.gamma) != 1.0:
        img = adjust_gamma(img, float(sem.gamma))
    if sem.use_clahe:
        img = clarity_enhance(img, sem.clahe_clip, sem.clahe_tile)
    if sem.use_usm:
        img = unsharp_mask(img, sem.usm_radius, sem.usm_amount)
    return img


def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def preview_preprocessing(gray_u8: np.ndarray, sem: SemParams, blur_kernel: Sequence[int]) -> np.ndarray:
    """튜너 미리보기용: 전처리 + GaussianBlur(blur_kernel)."""
    img = apply_sem_preprocessing(gray_u8, sem)
    kx, ky = _odd(blur_kernel[0]), _odd(blur_kernel[1])
    if kx > 0 and ky > 0:
        img = cv2.GaussianBlur(img, (kx, ky), 0)
    return img


__all__ = ["apply_sem_preprocessing", "preview_preprocessing"]

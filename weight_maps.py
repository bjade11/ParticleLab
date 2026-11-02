"""Weight loss maps — Bals & Epple (RSC Adv. 2023, eqn. 1) 구현 + 사용자 정의.

논문 식:
    w(x) = w0 * exp(-(d1(x) + d2(x))^2 / (2 * sigma^2)) + gamma * I_c(x) + 1

  - d1(x), d2(x): 배경 픽셀 x 에서 가장 가까운 두 입자 경계까지의 거리
  - w0, sigma, gamma: 조절 가능한 파라미터
  - I_c(x): 픽셀 x 의 정규화된 강도 (0~1)
  - 첫째 항만 쓰면 distance-based (SE 이미지)
  - 둘째 항을 추가하면 intensity-based (STEM 이미지)
  - foreground 픽셀의 weight 는 1 로 고정

논문 권장값: w0=10, sigma=7, gamma=2 (intensity 모드).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import ndimage as ndi


@dataclass
class WeightMapConfig:
    mode: str = "none"  # "none" | "distance" | "intensity" | "custom"
    w0: float = 10.0
    sigma: float = 7.0
    gamma: float = 2.0  # intensity 모드일 때만 사용
    custom_code: str = ""  # 사용자 정의 Python 코드 (mode='custom')


def _two_nearest_particle_distances(mask_bin: np.ndarray) -> np.ndarray:
    """배경 픽셀마다 가장 가까운 두 인스턴스의 거리 합 (d1 + d2) 계산.

    구현: 각 instance label 에 대해 그 외 영역의 거리변환을 모두 구해 stack 한 뒤,
    배경 픽셀에서 가장 작은 두 값을 합산.

    인스턴스가 1개뿐이거나 mask 가 비어있으면 0 반환 (가중치는 사실상 1로 수렴).
    """
    labeled, n_inst = ndi.label(mask_bin)
    if n_inst < 2:
        # 인스턴스가 0 또는 1 개 → d1+d2 정의 불가, 0 처리
        return np.zeros(mask_bin.shape, dtype=np.float32)

    # 각 인스턴스 외부 거리변환을 stack
    h, w = mask_bin.shape
    # 메모리 절약: 각 인스턴스마다 거리 → 작은 두 값만 유지
    best1 = np.full((h, w), np.inf, dtype=np.float32)
    best2 = np.full((h, w), np.inf, dtype=np.float32)
    for i in range(1, n_inst + 1):
        d = ndi.distance_transform_edt(labeled != i).astype(np.float32)
        # best1, best2 갱신
        m1 = d < best1
        # 새로 더 작아진 곳: best2 ← best1, best1 ← d
        prev_best1 = best1.copy()
        best1 = np.where(m1, d, best1)
        best2 = np.where(m1, prev_best1, np.where(d < best2, d, best2))
    # 배경 픽셀에서만 의미 있음 (foreground 는 weight=1 로 덮어쓸 거라 0 이어도 무방)
    return best1 + best2


def compute_weight_map(
    mask_bin: np.ndarray,
    image_uint8: Optional[np.ndarray] = None,
    cfg: WeightMapConfig = WeightMapConfig(),
) -> np.ndarray:
    """weight map (H, W) float32. foreground=1, background=논문 식."""
    h, w = mask_bin.shape
    out = np.ones((h, w), dtype=np.float32)
    if cfg.mode == "none":
        return out
    if cfg.mode == "custom":
        return _compute_custom(mask_bin, image_uint8, cfg)

    # background 픽셀 영역
    bg = ~mask_bin

    # 거리 항: w0 * exp(-(d1+d2)^2 / (2*sigma^2))
    d_sum = _two_nearest_particle_distances(mask_bin)
    dist_term = cfg.w0 * np.exp(-(d_sum ** 2) / (2.0 * cfg.sigma ** 2 + 1e-9))

    # intensity 항: gamma * I_c(x)
    if cfg.mode == "intensity" and image_uint8 is not None:
        ic = image_uint8.astype(np.float32) / 255.0
        # 권장: 너무 단단히 빛을 weight에 묶지 않도록 0~1 범위
        intensity_term = cfg.gamma * ic
    else:
        intensity_term = 0.0

    weights = dist_term + intensity_term + 1.0
    out[bg] = weights[bg]  # background 만 갱신, foreground 는 1
    return out


def _compute_custom(
    mask_bin: np.ndarray,
    image_uint8: Optional[np.ndarray],
    cfg: WeightMapConfig,
) -> np.ndarray:
    """사용자가 입력한 Python 코드로 weight map 계산.

    제공되는 변수:
      - mask: H x W bool (입자=True, 배경=False)
      - image: H x W float32 (0~1)  (없으면 zeros)
      - np: numpy
      - ndi: scipy.ndimage
      - w0, sigma, gamma: 사이드바 슬라이더 값
    제공되어야 할 결과:
      - w: H x W float32  (양수 권장; foreground/background 모두 정의)

    예시 (논문 식 + 추가 boost):
      d = ndi.distance_transform_edt(~mask)
      w = w0 * np.exp(-(d**2) / (2*sigma**2)) + 1
      w = np.maximum(w, 1.0)
    """
    code = cfg.custom_code.strip()
    if not code:
        return np.ones(mask_bin.shape, dtype=np.float32)

    img = (image_uint8.astype(np.float32) / 255.0
           if image_uint8 is not None
           else np.zeros(mask_bin.shape, dtype=np.float32))
    g_locals = {
        "mask": mask_bin.astype(bool),
        "image": img,
        "np": np,
        "ndi": ndi,
        "w0": cfg.w0,
        "sigma": cfg.sigma,
        "gamma": cfg.gamma,
        "w": None,
    }
    exec(code, {"__builtins__": __builtins__}, g_locals)
    w = g_locals.get("w")
    if w is None:
        raise ValueError(
            "Custom weight code must assign a 2D numpy array to variable `w`."
        )
    w = np.asarray(w, dtype=np.float32)
    if w.shape != mask_bin.shape:
        raise ValueError(
            f"Custom `w` shape {w.shape} != mask shape {mask_bin.shape}."
        )
    return w

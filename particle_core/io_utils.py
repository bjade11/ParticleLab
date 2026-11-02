"""유니코드(한글) 경로 안전 이미지 IO 및 파일명 유틸."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger("particle_core")

PathLike = Union[str, Path]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# 1k 기준 μm/px: 1.8k 에서 30μm/272px 실측 → base_1k = (30/272)*1.8 = 27/136
_BASE_1K = 27.0 / 136.0


def imread_unicode(path: PathLike, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """한글 경로 지원 이미지 로드. 실패 시 None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception as e:  # noqa: BLE001
        logger.warning("imread_unicode 실패 %s: %s", path, e)
        return None


def imwrite_unicode(path: PathLike, img: np.ndarray, jpeg_quality: int = 95) -> bool:
    """한글 경로 지원 이미지 저장 (cv2.imencode → tofile). 상위 폴더 자동 생성."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower() or ".png"
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("imwrite_unicode 실패 %s: %s", path, e)
        return False


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """임의 dtype/채널 이미지를 2D uint8 그레이스케일로 변환."""
    if img is None:
        raise ValueError("to_gray_u8: img is None")
    out = img
    if out.ndim == 3:
        if out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2GRAY)
        elif out.shape[2] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        else:
            out = out[:, :, 0]
    if out.dtype == np.uint16:
        maxv = int(out.max()) if out.max() > 0 else 65535
        out = (out.astype(np.float32) / maxv * 255.0).clip(0, 255).astype(np.uint8)
    elif out.dtype != np.uint8:
        out = cv2.normalize(out.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(out)


def natural_key(s: object) -> list:
    """Windows 탐색기식 자연 정렬 키."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def sample_name_of(filename: PathLike) -> str:
    """stem 에서 첫 '-' 앞, 없으면 첫 '_' 앞, 없으면 stem 전체."""
    stem = Path(str(filename)).stem
    if "-" in stem:
        return stem.split("-")[0]
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def pixel_to_micro_from_name(filename: PathLike, default: float = 0.7) -> float:
    """파일명의 배율 (예: '5.0k') 로부터 μm/px 계산: (27/136)/mag. 실패 시 default."""
    name = Path(str(filename)).name.lower()
    m = re.search(r"(\d+(?:\.\d+)?)k", name)
    if not m:
        return float(default)
    try:
        mag = float(m.group(1))
    except ValueError:
        return float(default)
    if mag > 0:
        return _BASE_1K / mag
    return float(default)


def list_images(folder: PathLike) -> List[Path]:
    """폴더 내 이미지 파일 목록 (비재귀, 자연 정렬)."""
    d = Path(folder)
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def crop_bottom(gray: np.ndarray, ratio: float) -> np.ndarray:
    """하단 ratio 비율만큼 잘라낸다 (스케일바 제거). 0 이하면 그대로."""
    ratio = float(ratio or 0.0)
    if ratio <= 0:
        return gray
    h = gray.shape[0]
    new_h = int(h * (1.0 - ratio))
    return gray[: max(new_h, 1), ...]


__all__ = [
    "IMAGE_EXTS", "imread_unicode", "imwrite_unicode", "to_gray_u8", "natural_key",
    "sample_name_of", "pixel_to_micro_from_name", "list_images", "crop_bottom",
]

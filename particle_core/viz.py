"""시각화: 라벨/윤곽선 이미지, 비교 이미지, 히스토그램, 요약 카드, 정규분포 플롯, 라벨 컬러맵."""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .io_utils import imwrite_unicode, natural_key

logger = logging.getLogger("particle_core")


def _to_bgr(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr.copy()


def draw_labeled_image(df: pd.DataFrame, labels: np.ndarray, base_gray: np.ndarray,
                       contours_map: Dict[int, np.ndarray], seed: int = 42) -> np.ndarray:
    """윤곽선 + 라벨 번호를 그린 BGR 이미지."""
    from .measure import extract_contour

    out = _to_bgr(base_gray)
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        lab = int(row["Label"])
        contour = contours_map.get(lab)
        if contour is None:
            contour = extract_contour(labels, lab)
            if contour is None:
                continue
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.drawContours(out, [contour], -1, color, 2)
        if pd.notna(row.get("x")) and pd.notna(row.get("y")):
            cx, cy = float(row["x"]), float(row["y"])
        else:
            cx, cy = float(np.mean(contour[:, 0, 0])), float(np.mean(contour[:, 0, 1]))
        pos = (max(0, min(int(cx), w - 10)), max(0, min(int(cy), h - 10)))
        cv2.putText(out, str(lab), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(lab), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_contour_only_image(df: pd.DataFrame, labels: np.ndarray, base_gray: np.ndarray,
                            contours_map: Dict[int, np.ndarray], seed: int = 42) -> np.ndarray:
    """윤곽선만 그린 BGR 이미지."""
    from .measure import extract_contour

    out = _to_bgr(base_gray)
    rng = np.random.default_rng(seed)
    if df is None or df.empty:
        return out
    for lab in df["Label"].tolist():
        lab = int(lab)
        contour = contours_map.get(lab)
        if contour is None:
            contour = extract_contour(labels, lab)
            if contour is None:
                continue
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.drawContours(out, [contour], -1, color, 2)
    return out


def save_comparison_image(img_before: np.ndarray, img_after: np.ndarray, path: str | Path, line_width: int = 5,
                          jpeg_quality: int = 95) -> None:
    """Before | 흰 선 | After 가로 결합 이미지 저장."""
    a, b = _to_bgr(img_before), _to_bgr(img_after)
    h = max(a.shape[0], b.shape[0])
    if a.shape[0] < h:
        a = np.vstack([a, np.zeros((h - a.shape[0], a.shape[1], 3), np.uint8)])
    if b.shape[0] < h:
        b = np.vstack([b, np.zeros((h - b.shape[0], b.shape[1], 3), np.uint8)])
    line = np.full((h, int(line_width), 3), 255, np.uint8)
    imwrite_unicode(path, np.concatenate([a, line, b], axis=1), jpeg_quality=jpeg_quality)


def save_histogram_comparison(before_values: Iterable[float], after_values: Iterable[float], feature_name: str,
                              path: str | Path, title: Optional[str] = None, tag: str = "After") -> None:
    """Before/After 밀도 히스토그램 저장."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bv = pd.to_numeric(pd.Series(list(before_values)), errors="coerce").dropna().to_numpy()
    av = pd.to_numeric(pd.Series(list(after_values)), errors="coerce").dropna().to_numpy()
    fig = plt.figure()
    try:
        ax = fig.gca()
        if bv.size > 0:
            ax.hist(bv, bins=30, alpha=0.5, label="Before", color="gray", density=True)
        if av.size > 0:
            ax.hist(av, bins=30, alpha=0.7, label=tag, color="skyblue", density=True)
        ax.set_title(title or f"{feature_name} Histogram Comparison ({tag})")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        if bv.size > 0 or av.size > 0:
            ax.legend()
        fig.savefig(str(path), bbox_inches="tight")
    finally:
        plt.close(fig)


def _load_font(font_path: Optional[str | Path], size: int = 20) -> ImageFont.ImageFont:
    if font_path and Path(font_path).is_file():
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception as e:  # noqa: BLE001
            logger.warning("폰트 로드 실패 %s: %s", font_path, e)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 구버전 PIL
        return ImageFont.load_default()


def save_summary_card(sample: str, stats: dict, path: str | Path, tag: str = "After",
                      font_path: Optional[str | Path] = None) -> None:
    """D값/통계 요약 카드 (PIL) 저장. stats 키: Particle Count, Mean, Median, Mode, Std, C.V., D10, D50, D90, Span."""
    def fmt(key: str, f: str = "{:.5f}", default: str = "N/A") -> str:
        v = stats.get(key)
        if v is None:
            return default
        if isinstance(v, (float, np.floating)):
            return f.format(float(v)) if np.isfinite(v) else default
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return default

    ratio = "N/A"
    med, mean = stats.get("Median"), stats.get("Mean")
    if med is not None and mean is not None and np.isfinite(med) and np.isfinite(mean) and med != 0:
        ratio = f"{float(mean) / float(med):.4f}"

    lines = [f"Sample: {sample} ({tag})"]
    if "Particle Count" in stats:
        lines.append(f"Particle Count: {stats['Particle Count']}")
    lines += [
        f"Mean: {fmt('Mean')}", f"Median: {fmt('Median')}", f"Mean/Median: {ratio}", f"Mode: {fmt('Mode')}",
        f"Std Dev: {fmt('Std')}", f"C.V.: {fmt('C.V.', '{:.4f}')}%", f"D10: {fmt('D10')}", f"D50: {fmt('D50')}",
        f"D90: {fmt('D90')}", f"Span: {fmt('Span', '{:.6f}')}",
    ]
    width, height = 480, 32 * len(lines) + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(font_path, 20)
    for i, line in enumerate(lines):
        draw.text((20, 10 + i * 32), line, font=font, fill="black")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(p), "JPEG", quality=95)


_NORMAL_TARGETS = [
    ("Circularity Stats", "Circularity", ("Mean_Circularity_Before", "Std_Circularity_Before", "Mean_Circularity_After", "Std_Circularity_After")),
    ("Aspect Ratio Stats", "Aspect Ratio", ("Mean_AspectRatio_Before", "Std_AspectRatio_Before", "Mean_AspectRatio_After", "Std_AspectRatio_After")),
    ("Area Ratio Stats", "Area Ratio", ("Mean_AreaRatio_Before", "Std_AreaRatio_Before", "Mean_AreaRatio_After", "Std_AreaRatio_After")),
    ("Perimeter Ratio Stats", "Perimeter Ratio", ("Mean_PerimeterRatio_Before", "Std_PerimeterRatio_Before", "Mean_PerimeterRatio_After", "Std_PerimeterRatio_After")),
    ("Radius Ratio Stats", "Radius Ratio", ("Mean_RadiusRatio_Before", "Std_RadiusRatio_Before", "Mean_RadiusRatio_After", "Std_RadiusRatio_After")),
    ("Summary", "D50", ("D50_Before", "Std_Before", "D50_After", "Std_After")),
    ("Solidity Stats", "Solidity", ("Mean_Solidity_Before", "Std_Solidity_Before", "Mean_Solidity_After", "Std_Solidity_After")),
]


def _plot_normals(means, stds, labels, feature: str, title: str, save_path: Path) -> bool:
    pts = [(l, float(m), float(s)) for l, m, s in zip(labels, means, stds)
           if np.isfinite(m) and np.isfinite(s) and s > 0]
    if not pts:
        return False
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:  # noqa: BLE001
        plt.style.use("default")
    fig = plt.figure(figsize=(12, 8))
    try:
        ax = fig.gca()
        cmap = plt.get_cmap("viridis")
        max_s = max(s for _, _, s in pts)
        x_min = min(m - 4 * max_s for _, m, _ in pts)
        x_max = max(m + 4 * max_s for _, m, _ in pts)
        x = np.linspace(x_min, x_max, 1000)
        n = len(pts)
        for i, (lab, m, s) in enumerate(pts):
            y = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-((x - m) ** 2) / (2 * s ** 2))
            ax.plot(x, y, color=cmap(i / max(n - 1, 1)), label=str(lab), linewidth=2)
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(str(save_path))
        return True
    finally:
        plt.close(fig)
        plt.style.use("default")


def save_normal_distribution_plots(excel_path: str | Path, out_dir: str | Path) -> List[str]:
    """Excel 통계 시트에서 샘플별 정규분포 곡선을 `out_dir/normal_dist/` 에 저장. 생성된 파일 경로 목록 반환."""
    norm_dir = Path(out_dir) / "normal_dist"
    norm_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    try:
        xls = pd.ExcelFile(str(excel_path))
    except Exception as e:  # noqa: BLE001
        logger.warning("정규분포: Excel 로드 실패 %s", e)
        return written
    for sheet, feature, (mb, sb, ma, sa) in _NORMAL_TARGETS:
        if sheet not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=sheet)
        if "Sample" not in df.columns:
            continue
        labels = df["Sample"].astype(str).tolist()
        for tag, mcol, scol in (("before", mb, sb), ("after", ma, sa)):
            if mcol not in df.columns or scol not in df.columns:
                continue
            z = sorted(zip(labels, df[mcol].to_numpy(dtype=float), df[scol].to_numpy(dtype=float)),
                       key=lambda t: natural_key(t[0]))
            l, m, s = zip(*z) if z else ([], [], [])
            p = norm_dir / f"{feature}_{tag}.png"
            title = f"{feature} Normal Distribution ({'Before' if tag == 'before' else 'After'} Filtering)"
            if _plot_normals(m, s, l, feature, title, p):
                written.append(str(p))
    return written


def encode_png_data_url(img_bgr_or_gray: np.ndarray) -> str:
    """ndarray → 'data:image/png;base64,...'"""
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    if not ok:
        raise ValueError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def colorize_labels(labels: np.ndarray, seed: int = 7) -> np.ndarray:
    """라벨맵 → 라벨별 랜덤 색 BGR (0=검정)."""
    n = int(labels.max()) if labels.size else 0
    rng = np.random.default_rng(seed)
    lut = rng.integers(40, 255, size=(n + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels.astype(np.int64)]


__all__ = [
    "draw_labeled_image", "draw_contour_only_image", "save_comparison_image", "save_histogram_comparison",
    "save_summary_card", "save_normal_distribution_plots", "encode_png_data_url", "colorize_labels",
]

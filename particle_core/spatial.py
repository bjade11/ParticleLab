"""공간 분석: 최근접 이웃 거리(NND)/응집도, DBSCAN 클러스터링, 경계선 거리/접촉.

원본 particle_analyzer_threads.py (NND/DBSCAN) 와 particle_analysis_functions.py (boundary) 포팅.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .params import SpatialParams

logger = logging.getLogger("particle_core")


def _ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ============================================================================= NND / 응집도
def calculate_nearest_neighbor_distances(df: pd.DataFrame, pixel_to_micro: float) -> pd.DataFrame:
    """각 입자의 최근접 이웃 거리(μm) 를 'Nearest_Neighbor_Distance' 컬럼으로 추가한 복사본 반환."""
    out = df.copy()
    out["Nearest_Neighbor_Distance"] = np.nan
    if "x" not in out.columns or "y" not in out.columns or len(out) < 2:
        return out
    valid = out[["x", "y"]].notna().all(axis=1)
    if valid.sum() < 2:
        return out
    coords = out.loc[valid, ["x", "y"]].to_numpy(dtype=float)
    try:
        tree = cKDTree(coords)
        dist, _ = tree.query(coords, k=2)
        out.loc[valid, "Nearest_Neighbor_Distance"] = dist[:, 1] * float(pixel_to_micro)
    except Exception as e:  # noqa: BLE001
        logger.warning("NND 계산 실패: %s", e)
    return out


def _zero_agglom(n: int) -> dict:
    return {"Mean_Nearest_Distance": 0, "Std_Nearest_Distance": 0, "Min_Nearest_Distance": 0,
            "Max_Nearest_Distance": 0, "Median_Nearest_Distance": 0, "Density": 0, "Total_Particles": int(n)}


def calculate_agglomeration_metrics(df: pd.DataFrame) -> dict:
    """NND 기반 응집도 통계. 'Nearest_Neighbor_Distance' 가 없으면 px 단위로 자동 계산."""
    n = len(df) if isinstance(df, pd.DataFrame) else 0
    if df is None or "x" not in df.columns or "y" not in df.columns or n == 0:
        return _zero_agglom(n)
    if "Nearest_Neighbor_Distance" not in df.columns:
        df = calculate_nearest_neighbor_distances(df, 1.0)
    nnd = pd.to_numeric(df["Nearest_Neighbor_Distance"], errors="coerce").dropna()
    x_range = float(df["x"].max() - df["x"].min())
    y_range = float(df["y"].max() - df["y"].min())
    total_area = x_range * y_range
    density = n / total_area if total_area > 0 else 0.0
    if nnd.empty:
        m = _zero_agglom(n)
        m["Density"] = density
        return m
    return {
        "Mean_Nearest_Distance": float(nnd.mean()),
        "Std_Nearest_Distance": float(nnd.std()) if len(nnd) > 1 else 0.0,
        "Min_Nearest_Distance": float(nnd.min()),
        "Max_Nearest_Distance": float(nnd.max()),
        "Median_Nearest_Distance": float(nnd.median()),
        "Density": float(density),
        "Total_Particles": int(n),
    }


def visualize_agglomeration(df: pd.DataFrame, pixel_to_micro: float, save_path: str | Path) -> None:
    """입자 위치 산점도 (색=NND, 크기=면적) 저장."""
    _ensure_dir(save_path)
    fig = plt.figure(figsize=(12, 10))
    try:
        need = {"x", "y", "Nearest_Neighbor_Distance", "Area (μm^2)"}
        ax = fig.gca()
        if not need.issubset(df.columns) or len(df) == 0:
            ax.text(0.5, 0.5, "No data for agglomeration visualization", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
        else:
            area = df["Area (μm^2)"].clip(lower=0.1)
            sizes = np.maximum(area / 10.0, 20)
            sc = ax.scatter(df["x"] * pixel_to_micro, df["y"] * pixel_to_micro,
                            c=df["Nearest_Neighbor_Distance"], cmap="viridis", s=sizes, alpha=0.7,
                            edgecolors="black", linewidths=0.5)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Nearest Neighbor Distance (um)", fontsize=12)
            ax.set_xlabel("X Position (um)", fontsize=12)
            ax.set_ylabel("Y Position (um)", fontsize=12)
            ax.set_title("Particle Distribution and Agglomeration", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.3)
            for s in ax.spines.values():
                s.set_linewidth(1.5)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


# ============================================================================= DBSCAN
def analyze_clusters_with_dbscan(df: pd.DataFrame, pixel_to_micro: float, sp: SpatialParams
                                 ) -> Tuple[pd.DataFrame, dict]:
    """DBSCAN 클러스터링. 'cluster_id' 컬럼이 추가된 복사본과 통계 dict 반환.

    stats: {n_clusters, noise_pct, eps, min_samples, total}
    """
    from sklearn.cluster import DBSCAN  # 지연 import (scikit-learn 미설치 환경에서 패키지 import 는 가능하도록)

    out = df.copy()
    n = len(out)
    stats = {"n_clusters": 0, "noise_pct": 0.0, "eps": float("nan"), "min_samples": 0, "total": int(n)}
    if n == 0 or "x" not in out.columns or "y" not in out.columns:
        out["cluster_id"] = np.array([], dtype=int) if n == 0 else -1
        return out, stats
    coords = out[["x", "y"]].to_numpy(dtype=float)
    coords = np.nan_to_num(coords)

    if sp.dbscan_eps_auto:
        if "Nearest_Neighbor_Distance" in out.columns and out["Nearest_Neighbor_Distance"].notna().any():
            # 원본: NND(μm) 평균 * factor — 좌표는 px 이므로 px 로 환산
            mean_nnd_um = float(out["Nearest_Neighbor_Distance"].mean())
            eps = (mean_nnd_um / float(pixel_to_micro) if pixel_to_micro > 0 else mean_nnd_um) * float(sp.dbscan_eps_factor)
        else:
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            eps = float(min(x_range, y_range)) * 0.08
    else:
        eps = float(sp.dbscan_eps_value)
    if not np.isfinite(eps) or eps <= 0:
        eps = 1.0

    if sp.dbscan_min_samples_auto:
        min_samples = max(2, int(n * float(sp.dbscan_min_samples_factor)))
    else:
        min_samples = max(1, int(sp.dbscan_min_samples_value))

    try:
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    except Exception as e:  # noqa: BLE001
        logger.warning("DBSCAN 실패: %s", e)
        clusters = np.full(n, -1, dtype=int)
    out["cluster_id"] = clusters
    n_clusters = len(set(clusters.tolist())) - (1 if -1 in clusters else 0)
    stats.update({
        "n_clusters": int(n_clusters),
        "noise_pct": float((clusters == -1).mean() * 100.0) if n > 0 else 0.0,
        "eps": float(eps),
        "min_samples": int(min_samples),
    })
    logger.debug("DBSCAN: eps=%.2f min_samples=%d clusters=%d", eps, min_samples, n_clusters)
    return out, stats


def visualize_dbscan_clusters(df: pd.DataFrame, save_path: str | Path) -> None:
    """DBSCAN 결과 산점도 저장 (cluster_id 없으면 전부 노이즈 처리)."""
    _ensure_dir(save_path)
    d = df if "cluster_id" in df.columns else df.assign(cluster_id=-1)
    fig = plt.figure(figsize=(12, 10))
    try:
        ax = fig.gca()
        if len(d) == 0 or "x" not in d.columns:
            ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        else:
            uniq = sorted(int(c) for c in d["cluster_id"].unique())
            noise = d["cluster_id"] == -1
            if noise.any():
                ax.scatter(d.loc[noise, "x"], d.loc[noise, "y"], c="gray", s=10, alpha=0.3, label="Noise")
            cmap = plt.get_cmap("tab10")
            for cid in uniq:
                if cid == -1:
                    continue
                m = d["cluster_id"] == cid
                ax.scatter(d.loc[m, "x"], d.loc[m, "y"], color=cmap(cid % 10), s=50, alpha=0.7,
                           label=f"Cluster {cid}")
                cx, cy = float(d.loc[m, "x"].mean()), float(d.loc[m, "y"].mean())
                ax.scatter(cx, cy, c="black", marker="x", s=100)
                ax.text(cx, cy, str(cid), fontsize=12)
            if len(uniq) <= 11:
                ax.legend(loc="upper right")
            ax.invert_yaxis()
        ax.set_title("DBSCAN Clustering Results", fontsize=14)
        ax.set_xlabel("X Position (px)", fontsize=12)
        ax.set_ylabel("Y Position (px)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


# ============================================================================= 경계선 거리 / 접촉
def _contour_min_dist(c1: np.ndarray, c2: np.ndarray) -> float:
    """두 윤곽선 간 최단 거리(px): 각 윤곽선 점에서 상대 폴리곤까지 거리의 최소 (원본 방식)."""
    best = float("inf")
    for pt in c1.reshape(-1, 2):
        d = abs(cv2.pointPolygonTest(c2, (float(pt[0]), float(pt[1])), True))
        if d < best:
            best = d
    for pt in c2.reshape(-1, 2):
        d = abs(cv2.pointPolygonTest(c1, (float(pt[0]), float(pt[1])), True))
        if d < best:
            best = d
    return best


def calculate_boundary_distances(df: pd.DataFrame, contours_map: Dict[int, np.ndarray], contact_threshold: float,
                                 pixel_to_micro: float, labeled_mask: Optional[np.ndarray]
                                 ) -> Tuple[pd.DataFrame, List[tuple]]:
    """입자 윤곽선 간 최단 거리와 접촉 수를 계산.

    Returns (df 복사본 + 'Boundary_Distance'(μm), 'Contact_Count'), contact_pairs [(l1, l2, dist_μm)].
    외접원 기반 하한으로 쌍을 가지치기하므로 결과는 원본 완전탐색과 동일하되 훨씬 빠르다.
    """
    from .measure import extract_contour

    out = df.copy()
    out["Boundary_Distance"] = np.nan
    out["Contact_Count"] = 0
    if len(out) < 2:
        return out, []

    cmap: Dict[int, np.ndarray] = dict(contours_map or {})
    labels_list = [int(v) for v in out["Label"].tolist()]
    for lab in labels_list:
        if lab not in cmap and labeled_mask is not None:
            c = extract_contour(labeled_mask, lab)
            if c is not None:
                cmap[lab] = c
    valid = [lab for lab in labels_list if lab in cmap]
    if len(valid) < 2:
        return out, []

    # 외접원 (중심, 반지름) → 거리 하한 = |c1-c2| - r1 - r2
    centers = np.zeros((len(valid), 2), dtype=np.float64)
    radii = np.zeros(len(valid), dtype=np.float64)
    for i, lab in enumerate(valid):
        (cx, cy), r = cv2.minEnclosingCircle(cmap[lab].reshape(-1, 2).astype(np.float32))
        centers[i] = (cx, cy)
        radii[i] = r

    thr = float(contact_threshold)
    p2m = float(pixel_to_micro)
    bd: Dict[int, float] = {}
    cc: Dict[int, int] = {}
    pairs: List[tuple] = []
    pair_cache: Dict[Tuple[int, int], float] = {}

    for i, l1 in enumerate(valid):
        cd = np.linalg.norm(centers - centers[i], axis=1)
        lower = cd - radii - radii[i]
        order = np.argsort(lower)
        min_d = float("inf")
        n_contact = 0
        for j in order:
            if j == i:
                continue
            lb = float(lower[j])
            if lb > min_d and lb > thr:
                break  # 이후 쌍은 더 가까울 수 없음
            l2 = valid[j]
            key = (min(l1, l2), max(l1, l2))
            d = pair_cache.get(key)
            if d is None:
                d = _contour_min_dist(cmap[l1], cmap[l2])
                pair_cache[key] = d
            if d < min_d:
                min_d = d
            if d <= thr:
                n_contact += 1
                if l1 < l2:
                    pairs.append((l1, l2, d * p2m))
        bd[l1] = min_d * p2m if np.isfinite(min_d) else np.nan
        cc[l1] = n_contact

    out["Boundary_Distance"] = out["Label"].map(bd)
    out["Contact_Count"] = out["Label"].map(cc).fillna(0).astype(int)
    return out, pairs


def _zero_boundary(n: int) -> dict:
    return {"Mean_Boundary_Distance": 0, "Std_Boundary_Distance": 0, "Min_Boundary_Distance": 0,
            "Max_Boundary_Distance": 0, "Median_Boundary_Distance": 0, "Contact_Ratio": 0,
            "Average_Contacts_Per_Particle": 0, "Total_Particles": int(n)}


def calculate_boundary_metrics(df: pd.DataFrame) -> dict:
    """경계선 거리 기반 응집 통계."""
    n = len(df)
    if "Boundary_Distance" not in df.columns:
        return _zero_boundary(n)
    bdist = pd.to_numeric(df["Boundary_Distance"], errors="coerce").dropna()
    if bdist.empty:
        return _zero_boundary(n)
    cc = pd.to_numeric(df.get("Contact_Count", pd.Series(dtype=float)), errors="coerce").fillna(0)
    with_contact = int((cc > 0).sum())
    return {
        "Mean_Boundary_Distance": float(bdist.mean()),
        "Std_Boundary_Distance": float(bdist.std()) if len(bdist) > 1 else 0.0,
        "Min_Boundary_Distance": float(bdist.min()),
        "Max_Boundary_Distance": float(bdist.max()),
        "Median_Boundary_Distance": float(bdist.median()),
        "Contact_Ratio": with_contact / n if n > 0 else 0.0,
        "Average_Contacts_Per_Particle": float(cc.mean()) if n > 0 else 0.0,
        "Total_Particles": int(n),
    }


def visualize_boundary_contacts(df: pd.DataFrame, contours_map: Dict[int, np.ndarray], contact_pairs: List[tuple],
                                img_shape: Tuple[int, int], save_path: str | Path, pixel_to_micro: float,
                                contact_threshold: float, base_gray: Optional[np.ndarray],
                                labeled_mask: Optional[np.ndarray]) -> None:
    """경계 접촉 시각화 저장. base_gray 는 labels 와 같은 크기 (원본 크기 불일치 버그 수정)."""
    from .io_utils import imwrite_unicode

    h, w = int(img_shape[0]), int(img_shape[1])
    if base_gray is not None and base_gray.shape[:2] == (h, w):
        base = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR) if base_gray.ndim == 2 else base_gray.copy()
    else:
        if base_gray is not None:
            logger.warning("visualize_boundary_contacts: base_gray shape %s != labels %s, using white",
                           base_gray.shape[:2], (h, w))
        base = np.full((h, w, 3), 255, dtype=np.uint8)

    legend_h = 200
    img = np.full((h + legend_h, w, 3), 255, dtype=np.uint8)
    img[:h] = base
    cv2.line(img, (0, h), (w, h), (0, 0, 0), 1)

    coord: Dict[int, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("x")) and pd.notna(row.get("y")):
            coord[int(row["Label"])] = (int(row["x"]), int(row["y"]))

    rng = np.random.default_rng(42)
    for lab, contour in contours_map.items():
        color = tuple(int(v) for v in rng.integers(100, 255, size=3))
        cv2.drawContours(img, [contour], -1, color, 2)
        if lab not in coord:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                x, y, bw, bh = cv2.boundingRect(contour)
                cx, cy = x + bw // 2, y + bh // 2
            coord[lab] = (max(0, min(cx, w - 1)), max(0, min(cy, h - 1)))

    contact_labels = set()
    for l1, l2, _ in contact_pairs:
        contact_labels.add(l1)
        contact_labels.add(l2)

    p2m = float(pixel_to_micro) if pixel_to_micro > 0 else 1.0
    pixel_pairs = sorted(((l1, l2, d / p2m) for l1, l2, d in contact_pairs), key=lambda t: t[2])
    close = [p for p in pixel_pairs if p[2] <= contact_threshold]
    show = (close if len(close) >= 5 else pixel_pairs)[:100]
    colors = [(0, 0, 255), (0, 128, 255), (0, 255, 0)]
    for l1, l2, dpx in show:
        if l1 not in coord or l2 not in coord:
            continue
        if dpx <= 0.5:
            color, t = colors[0], 2
        elif dpx <= 1.5:
            color, t = colors[1], 1
        else:
            color, t = colors[2], 1
        cv2.line(img, coord[l1], coord[l2], color, t)
    for lab in contact_labels:
        if lab in coord:
            cv2.circle(img, coord[lab], 4, (0, 165, 255), -1)
    for lab, c in coord.items():
        if lab not in contact_labels:
            cv2.circle(img, c, 3, (0, 0, 255), -1)

    # 범례
    ly = h + 40
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Contact Distance (pixel):", (20, ly), f, 0.7, (0, 0, 0), 1)
    lw = max(1, w // 3)
    for i, (color, text) in enumerate(zip(colors, ["0 px", "1 px", "2 px"])):
        rx, ry = 20 + i * lw, ly + 20
        cv2.rectangle(img, (rx, ry), (rx + 30, ry + 15), color, -1)
        cv2.putText(img, text, (rx + 40, ry + 12), f, 0.7, (0, 0, 0), 1)
    y1, y2 = ly + 70, ly + 110
    cv2.putText(img, "Orange dots:", (20, y1), f, 0.7, (0, 0, 0), 1)
    cv2.circle(img, (170, y1 - 5), 4, (0, 165, 255), -1)
    cv2.putText(img, "Particles with contacts", (190, y1), f, 0.7, (0, 0, 0), 1)
    cv2.putText(img, "Red dots:", (20, y2), f, 0.7, (0, 0, 0), 1)
    cv2.circle(img, (170, y2 - 5), 3, (0, 0, 255), -1)
    cv2.putText(img, "Isolated particles", (190, y2), f, 0.7, (0, 0, 0), 1)
    iy = y2 + 40
    cv2.putText(img, f"Total particles: {len(df)}", (20, iy), f, 0.7, (0, 0, 0), 1)
    cv2.putText(img, f"Contact threshold: {float(contact_threshold):.1f} px", (w // 2, iy), f, 0.7, (0, 0, 0), 1)

    imwrite_unicode(save_path, img)


__all__ = [
    "calculate_nearest_neighbor_distances", "calculate_agglomeration_metrics", "visualize_agglomeration",
    "analyze_clusters_with_dbscan", "visualize_dbscan_clusters",
    "calculate_boundary_distances", "calculate_boundary_metrics", "visualize_boundary_contacts",
]

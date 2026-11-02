"""통계 유틸: D값, 분포 통계, Summary 행, DataFrame 정리."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# 분포 통계 시트 대상 (컬럼명 → 시트명). 순서는 Excel 시트 순서와 동일.
DIST_FEATURES: List[tuple] = [
    ("Circularity", "Circularity Stats"),
    ("Aspect Ratio", "Aspect Ratio Stats"),
    ("Perimeter (μm)", "Perimeter Stats"),
    ("Area_Ratio", "Area Ratio Stats"),
    ("Perimeter_Ratio", "Perimeter Ratio Stats"),
    ("Radius_Ratio", "Radius Ratio Stats"),
    ("Solidity", "Solidity Stats"),
]


def _nan_dvalues(n: int = 0) -> Dict[str, float]:
    return {"d10": np.nan, "d50": np.nan, "d90": np.nan, "span": np.nan, "mean": np.nan,
            "std": np.nan, "mode": np.nan, "cv": np.nan, "n": int(n)}


def calculate_d_values(df: pd.DataFrame) -> Dict[str, float]:
    """'Diameter (μm)' 분포의 d10/d50/d90/span/mean/std/mode/cv/n. 데이터 없으면 NaN."""
    if df is None or df.empty or "Diameter (μm)" not in df.columns:
        return _nan_dvalues(0)
    d = pd.to_numeric(df["Diameter (μm)"], errors="coerce").dropna()
    if d.empty:
        return _nan_dvalues(0)
    d10, d50, d90 = (float(v) for v in np.percentile(d, [10, 50, 90]))
    span = (d90 - d10) / d50 if d50 != 0 else np.nan
    mean = float(d.mean())
    std = float(d.std()) if len(d) > 1 else 0.0
    cv = (std / mean) * 100.0 if mean != 0 else np.nan
    try:
        hist, edges = np.histogram(d, bins=30)
        i = int(np.argmax(hist))
        mode = float((edges[i] + edges[i + 1]) / 2.0)
    except Exception:  # noqa: BLE001
        mode = np.nan
    return {"d10": d10, "d50": d50, "d90": d90, "span": float(span), "mean": mean, "std": std,
            "mode": mode, "cv": float(cv), "n": int(len(d))}


def collect_distribution_statistics(df: pd.DataFrame, feature: str) -> Dict[str, float]:
    """Mean/Std/Min/Max/Median/25th/75th (원본 threads.collect_distribution_statistics)."""
    zero = {"Mean": 0, "Std": 0, "Min": 0, "Max": 0, "Median": 0, "25th": 0, "75th": 0}
    if df is None or feature not in df.columns:
        return dict(zero)
    s = pd.to_numeric(df[feature], errors="coerce").dropna()
    if s.empty:
        return dict(zero)
    return {
        "Mean": float(s.mean()), "Std": float(s.std()) if len(s) > 1 else 0.0,
        "Min": float(s.min()), "Max": float(s.max()), "Median": float(s.median()),
        "25th": float(s.quantile(0.25)), "75th": float(s.quantile(0.75)),
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """중복 제거 → NaN 50% 이상 행 제거 → Sample NaN 제거 → 숫자 NaN 0 → 인덱스 리셋 (threads.py 버전)."""
    if df is None or df.empty:
        return df
    try:
        out = df.drop_duplicates()
        out = out.dropna(thresh=len(out.columns) // 2)
        if "Sample" in out.columns:
            out = out.dropna(subset=["Sample"])
        out = out.copy()
        num_cols = out.select_dtypes(include=["number"]).columns
        for c in num_cols:
            out[c] = out[c].fillna(0)
        return out.reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return df


def summary_row(sample: str, df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Summary 시트 1행 (원본 summary_entry_s). NaN 은 0.0 으로."""
    b = calculate_d_values(df_before)
    a = calculate_d_values(df_after)
    row = {
        "Sample": str(sample),
        "Total_Particles_Before": int(len(df_before)),
        "D10_Before": b["d10"], "D50_Before": b["d50"], "D90_Before": b["d90"], "Span_Before": b["span"],
        "Mean_Before": b["mean"], "Std_Before": b["std"], "Mode_Before": b["mode"], "CV_Before": b["cv"],
        "Total_Particles_After": int(len(df_after)),
        "D10_After": a["d10"], "D50_After": a["d50"], "D90_After": a["d90"], "Span_After": a["span"],
        "Mean_After": a["mean"], "Std_After": a["std"], "Mode_After": a["mode"], "CV_After": a["cv"],
    }
    for k, v in row.items():
        if isinstance(v, float) and not np.isfinite(v):
            row[k] = 0.0
    return row


def feature_key(feature: str) -> str:
    """'Aspect Ratio' → 'AspectRatio', 'Perimeter (μm)' → 'Perimeter', 'Area_Ratio' → 'AreaRatio'."""
    return feature.replace(" (μm)", "").replace("_", "").replace(" ", "")


def distribution_stats_row(sample: str, df_before: pd.DataFrame, df_after: pd.DataFrame, feature: str) -> dict:
    """'<X> Stats' 시트 1행: Mean_<K>_Before, Std_<K>_Before, '<K> Min (Before)', ... (원본 명명 유지)."""
    k = feature_key(feature)
    row: dict = {"Sample": str(sample)}
    for tag, df in (("Before", df_before), ("After", df_after)):
        st = collect_distribution_statistics(df, feature)
        for name, val in st.items():
            if name == "Mean":
                row[f"Mean_{k}_{tag}"] = val
            elif name == "Std":
                row[f"Std_{k}_{tag}"] = val
            else:
                row[f"{k} {name} ({tag})"] = val
    return row


def stats_for_card(df: pd.DataFrame) -> dict:
    """요약 카드용 dict: D10/D50/D90/Span/Mean/Std/Mode/C.V./Median/Particle Count."""
    d = calculate_d_values(df)
    return {"D10": d["d10"], "D50": d["d50"], "D90": d["d90"], "Span": d["span"], "Mean": d["mean"],
            "Std": d["std"], "Mode": d["mode"], "C.V.": d["cv"], "Median": d["d50"],
            "Particle Count": int(len(df))}


__all__ = ["DIST_FEATURES", "calculate_d_values", "collect_distribution_statistics", "clean_dataframe",
           "summary_row", "feature_key", "distribution_stats_row", "stats_for_card"]

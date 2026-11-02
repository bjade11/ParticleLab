"""Before → After 필터 (원본 particle_analyzer_threads.py 572-591 과 동일 의미)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import AfterFilterParams


def apply_after_filter(df_before: pd.DataFrame, af: AfterFilterParams) -> pd.DataFrame:
    """직경 percentile 범위 + Circularity 하한 + Aspect Ratio 상한 필터. 빈 입력이면 빈 복사본 반환."""
    if df_before is None or df_before.empty or "Diameter (μm)" not in df_before.columns:
        return df_before.iloc[0:0].copy() if df_before is not None else pd.DataFrame()

    d = df_before["Diameter (μm)"].dropna()
    if d.empty:
        return df_before.iloc[0:0].copy()

    d_low = float(np.percentile(d, float(af.diameter_lower_percentile)))
    d_high = float(np.percentile(d, 100.0 - float(af.diameter_upper_percentile)))

    mask = (df_before["Diameter (μm)"] >= d_low) & (df_before["Diameter (μm)"] <= d_high)

    if getattr(af, "enable_circularity", True):
        circ = df_before["Circularity"].dropna()
        if (af.circularity_mode or "value") == "percentile" and not circ.empty:
            circ_th = float(np.percentile(circ, float(af.circularity_percentile)))
        else:
            circ_th = float(af.circularity_value)
        mask &= df_before["Circularity"] >= circ_th

    if getattr(af, "enable_aspect_ratio", True):
        ar = df_before["Aspect Ratio"].dropna()
        if (af.aspect_ratio_mode or "value") == "percentile" and not ar.empty:
            ar_th = float(np.percentile(ar, 100.0 - float(af.aspect_ratio_percentile)))
        else:
            ar_th = float(af.aspect_ratio_value)
        mask &= df_before["Aspect Ratio"] <= ar_th

    return df_before[mask].copy()


__all__ = ["apply_after_filter"]

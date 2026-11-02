"""Excel / PPT 리포트 생성."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .io_utils import natural_key, sample_name_of

logger = logging.getLogger("particle_core")

# 통계 시트 고정 순서 (DESIGN §2.1)
STATS_SHEET_ORDER: List[str] = [
    "Summary", "Circularity Stats", "Aspect Ratio Stats", "Perimeter Stats", "Area Ratio Stats",
    "Perimeter Ratio Stats", "Radius Ratio Stats", "Solidity Stats", "Agglomeration Stats", "DBSCAN Stats",
    "Boundary Stats",
]

_INVALID = set('[]:*?/\\')


def _safe_sheet_name(name: str, used: set) -> str:
    """31자 제한 + 금지문자 치환 + 중복 시 _2, _3 접미."""
    base = "".join("_" if ch in _INVALID else ch for ch in str(name)).strip("'") or "Sheet"
    base = base[:31]
    cand, i = base, 2
    while cand.lower() in used:
        suffix = f"_{i}"
        cand = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(cand.lower())
    return cand


def _sort_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    if "Sample" in df.columns and not df.empty:
        df = df.copy()
        df["Sample"] = df["Sample"].astype(str)
        order = sorted(range(len(df)), key=lambda i: natural_key(df["Sample"].iloc[i]))
        return df.iloc[order].reset_index(drop=True)
    return df


def write_excel(path: str | Path, sample_sheets: Dict[str, pd.DataFrame], stats_sheets: Dict[str, pd.DataFrame]) -> None:
    """샘플 시트(자연 정렬) → 통계 시트(STATS_SHEET_ORDER, 그 외는 뒤에) 순으로 Excel 작성."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    used: set = set()
    with pd.ExcelWriter(str(p), engine="openpyxl") as writer:
        wrote_any = False
        for sample in sorted(sample_sheets.keys(), key=natural_key):
            df = sample_sheets[sample]
            if df is None:
                continue
            df.to_excel(writer, sheet_name=_safe_sheet_name(sample, used), index=False)
            wrote_any = True
        ordered = [k for k in STATS_SHEET_ORDER if k in stats_sheets] + \
                  [k for k in stats_sheets if k not in STATS_SHEET_ORDER]
        for key in ordered:
            df = stats_sheets[key]
            if df is None or df.empty:
                continue
            _sort_by_sample(df).to_excel(writer, sheet_name=_safe_sheet_name(key, used), index=False)
            wrote_any = True
        if not wrote_any:
            pd.DataFrame({"info": ["no results"]}).to_excel(writer, sheet_name="Summary", index=False)


def _comparison_stem(fname: str) -> str:
    return fname[: -len("_comparison.jpg")]


def create_summary_ppt(comparison_dir: str | Path, summary_dir: str | Path, out_path: str | Path,
                       comparison_img_width: float = 6.0, summary_img_width: float = 1.9) -> None:
    """샘플별 1슬라이드: 제목 + Before/After 요약 카드 + 비교 이미지 최대 3장.

    비교 이미지 매칭은 prefix 가 아니라 sample_name_of(stem) == sample (정확 매칭) 으로 한다.
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt

    cdir, sdir = Path(comparison_dir), Path(summary_dir)
    if not sdir.is_dir():
        raise FileNotFoundError(f"summary dir not found: {sdir}")
    samples = set()
    for f in sdir.iterdir():
        n = f.name
        if n.endswith("_Before_summary.jpg"):
            samples.add(n[: -len("_Before_summary.jpg")])
        elif n.endswith("_After_summary.jpg"):
            samples.add(n[: -len("_After_summary.jpg")])
    if not samples:
        raise ValueError("No summary images found")

    comp_files = [f.name for f in cdir.iterdir() if f.name.endswith("_comparison.jpg")] if cdir.is_dir() else []

    prs = Presentation()
    layout = prs.slide_layouts[6]
    for sample in sorted(samples, key=natural_key):
        slide = prs.slides.add_slide(layout)
        tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(6), Inches(0.5))
        tb.text_frame.text = sample
        tb.text_frame.paragraphs[0].font.size = Pt(32)
        tb.text_frame.paragraphs[0].font.bold = True

        before = sdir / f"{sample}_Before_summary.jpg"
        after = sdir / f"{sample}_After_summary.jpg"
        if before.is_file():
            slide.shapes.add_picture(str(before), Inches(6.2), Inches(1.0), width=Inches(summary_img_width))
        if after.is_file():
            slide.shapes.add_picture(str(after), Inches(6.2 + summary_img_width), Inches(1.0),
                                     width=Inches(summary_img_width))

        mine = [f for f in comp_files if sample_name_of(_comparison_stem(f) + ".jpg") == sample]
        for idx, fname in enumerate(sorted(mine, key=natural_key)[:3]):
            y = 1.0 + idx * (comparison_img_width * 0.36)
            try:
                slide.shapes.add_picture(str(cdir / fname), Inches(0), Inches(y), width=Inches(comparison_img_width))
            except Exception as e:  # noqa: BLE001
                logger.warning("PPT 비교 이미지 추가 실패 %s: %s", fname, e)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))


__all__ = ["STATS_SHEET_ORDER", "write_excel", "create_summary_ppt"]

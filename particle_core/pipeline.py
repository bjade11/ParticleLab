"""배치 파이프라인 (원본 ParticleAnalysisThread.run 포팅, 버그 수정 포함).

이미지 1장: 로드 → 전처리(크롭 1회) → 엔진 분할 → 특성 → Before/After → 이미지 출력 → 공간분석
샘플 단위: concat → 샘플 전체 After 필터 → 시트/히스토그램/요약카드/통계 행
배치 끝: Excel → 정규분포 플롯 → PPT
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import spatial, stats, viz
from .engines import SegmentationEngine
from .filters import apply_after_filter
from .io_utils import imread_unicode, imwrite_unicode, natural_key, pixel_to_micro_from_name, sample_name_of, to_gray_u8
from .measure import measure_particles
from .params import AnalysisParams
from .preprocess import apply_sem_preprocessing
from .report import create_summary_ppt, write_excel

logger = logging.getLogger("particle_core")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT = PROJECT_ROOT / "arial.ttf"

ProgressCB = Callable[[int, str], None]
CancelCheck = Callable[[], bool]


@dataclass
class BatchInput:
    images: List[Path]
    output_dir: Path


@dataclass
class ImageResult:
    filename: str
    sample: str
    pixel_to_micro: float
    n_before: int
    n_after: int
    files: Dict[str, str] = field(default_factory=dict)


@dataclass
class BatchResult:
    output_dir: str
    excel_path: str
    pptx_path: Optional[str]
    images: List[ImageResult] = field(default_factory=list)
    samples: List[dict] = field(default_factory=list)
    sample_files: Dict[str, Dict[str, str]] = field(default_factory=dict)
    extra_files: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BatchCancelled(Exception):
    """cancel_check 가 True 를 반환했을 때 내부 제어용."""


def _rel(out: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(out.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _avg_row(sample: str, rows: List[dict]) -> dict:
    """이미지별 통계 행들의 평균 (ImageName='ALL_IMAGES_AVG')."""
    out = {"Sample": sample, "ImageName": "ALL_IMAGES_AVG"}
    keys = [k for k in rows[0].keys() if k not in ("Sample", "ImageName")]
    for k in keys:
        vals = [r[k] for r in rows if k in r and pd.notna(r[k]) and isinstance(r[k], (int, float, np.integer, np.floating))]
        out[k] = float(np.mean(vals)) if vals else 0
    return out


def _resolve_p2m(filename: str, params: AnalysisParams) -> float:
    sc = params.scale
    if (sc.mode or "auto").lower() == "auto":
        return float(pixel_to_micro_from_name(filename, sc.auto_default))
    return float(sc.pixel_to_micro)


# ----------------------------------------------------------------------------- 이미지 1장
def _process_image(path: Path, sample: str, out: Path, params: AnalysisParams, engine: SegmentationEngine,
                   warnings: List[str]) -> tuple:
    """Returns (ImageResult, df_before, df_after, boundary_row|None, agglom_row|None, dbscan_row|None)."""
    filename = path.name
    stem = path.stem
    q = int(params.output.jpeg_quality)
    p2m = _resolve_p2m(filename, params)

    raw = imread_unicode(path)
    if raw is None:
        raise ValueError(f"이미지 로드 실패: {path}")
    gray = to_gray_u8(raw)
    gray_pre = apply_sem_preprocessing(gray, params.sem)          # 크롭 1회, 블러 없음
    seg = engine.segment(gray_pre)                                 # watershed 는 내부에서 블러 1회
    labels = seg.labels
    if labels.shape[:2] != gray_pre.shape[:2]:
        raise ValueError(f"engine returned labels {labels.shape} != preprocessed image {gray_pre.shape}")

    df_before, contours = measure_particles(labels, gray_pre, p2m, params.measure)
    df_after = apply_after_filter(df_before, params.after_filter)
    df_before["ImageName"] = filename
    df_before["Filter"] = "Before"
    df_after["ImageName"] = filename
    df_after["Filter"] = "After"
    if df_before.empty:
        warnings.append(f"{filename}: 입자가 검출되지 않았습니다.")

    after_labels = set(int(v) for v in df_after["Label"].tolist())
    contours_after = {k: v for k, v in contours.items() if k in after_labels}

    files: Dict[str, str] = {}

    # --- 라벨/윤곽선/비교 이미지 ---
    img_b = viz.draw_labeled_image(df_before, labels, gray_pre, contours)
    img_a = viz.draw_labeled_image(df_after, labels, gray_pre, contours_after)
    p = out / "labeled_images" / "before" / f"{stem}_before_labeled.jpg"
    imwrite_unicode(p, img_b, q); files["labeled_before"] = _rel(out, p)
    p = out / "labeled_images" / "after" / f"{stem}_after_labeled.jpg"
    imwrite_unicode(p, img_a, q); files["labeled_after"] = _rel(out, p)
    if params.output.make_contour_only:
        p = out / "labeled_images" / "before_contour_only" / f"{stem}_before_contour.jpg"
        imwrite_unicode(p, viz.draw_contour_only_image(df_before, labels, gray_pre, contours), q)
        files["contour_before"] = _rel(out, p)
        p = out / "labeled_images" / "after_contour_only" / f"{stem}_after_contour.jpg"
        imwrite_unicode(p, viz.draw_contour_only_image(df_after, labels, gray_pre, contours_after), q)
        files["contour_after"] = _rel(out, p)
    p = out / "comparison_images" / f"{stem}_comparison.jpg"
    viz.save_comparison_image(img_b, img_a, p, jpeg_quality=q)
    files["comparison"] = _rel(out, p)

    sp = params.spatial
    boundary_row = agglom_row = dbscan_row = None

    # --- 경계선 거리/접촉 ---
    if sp.enable_boundary:
        try:
            thr = float(sp.boundary_contact_threshold)
            b_df, b_pairs = spatial.calculate_boundary_distances(df_before, contours, thr, p2m, labels)
            a_df, a_pairs = spatial.calculate_boundary_distances(df_after, contours_after, thr, p2m, labels)
            bm, am = spatial.calculate_boundary_metrics(b_df), spatial.calculate_boundary_metrics(a_df)
            p = out / "boundary_analysis" / f"{stem}_before_boundary.jpg"
            spatial.visualize_boundary_contacts(b_df, contours, b_pairs, labels.shape, p, p2m, thr, gray_pre, labels)
            files["boundary_before"] = _rel(out, p)
            p = out / "boundary_analysis" / f"{stem}_after_boundary.jpg"
            spatial.visualize_boundary_contacts(a_df, contours_after, a_pairs, labels.shape, p, p2m, thr, gray_pre, labels)
            files["boundary_after"] = _rel(out, p)
            boundary_row = {"Sample": sample, "ImageName": filename,
                            **{f"{k}_Before": v for k, v in bm.items()},
                            **{f"{k}_After": v for k, v in am.items()}}
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{filename}: boundary 분석 실패: {e}")
            logger.debug(traceback.format_exc())

    # --- 최근접 이웃 / 응집도 ---
    df_b_nn = df_a_nn = None
    if sp.enable_nnd:
        try:
            df_b_nn = spatial.calculate_nearest_neighbor_distances(df_before, p2m)
            df_a_nn = spatial.calculate_nearest_neighbor_distances(df_after, p2m)
            p = out / "agglomeration_visualization" / f"{stem}_before_agglomeration.png"
            spatial.visualize_agglomeration(df_b_nn, p2m, p); files["aggl_before"] = _rel(out, p)
            p = out / "agglomeration_visualization" / f"{stem}_after_agglomeration.png"
            spatial.visualize_agglomeration(df_a_nn, p2m, p); files["aggl_after"] = _rel(out, p)
            bm, am = spatial.calculate_agglomeration_metrics(df_b_nn), spatial.calculate_agglomeration_metrics(df_a_nn)
            agglom_row = {
                "Sample": sample, "ImageName": filename,
                "Total_Particles_Before": bm["Total_Particles"], "Mean_Distance_Before": bm["Mean_Nearest_Distance"],
                "Std_Distance_Before": bm["Std_Nearest_Distance"], "Median_Distance_Before": bm["Median_Nearest_Distance"],
                "Density_Before": bm["Density"],
                "Total_Particles_After": am["Total_Particles"], "Mean_Distance_After": am["Mean_Nearest_Distance"],
                "Std_Distance_After": am["Std_Nearest_Distance"], "Median_Distance_After": am["Median_Nearest_Distance"],
                "Density_After": am["Density"],
            }
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{filename}: NND 분석 실패: {e}")
            logger.debug(traceback.format_exc())

    # --- DBSCAN ---
    if sp.enable_dbscan:
        try:
            src_b = df_b_nn if df_b_nn is not None else df_before
            src_a = df_a_nn if df_a_nn is not None else df_after
            cb_df, cb = spatial.analyze_clusters_with_dbscan(src_b, p2m, sp)
            ca_df, ca = spatial.analyze_clusters_with_dbscan(src_a, p2m, sp)
            p = out / "dbscan_visualization" / f"{stem}_before_dbscan.png"
            spatial.visualize_dbscan_clusters(cb_df, p); files["dbscan_before"] = _rel(out, p)
            p = out / "dbscan_visualization" / f"{stem}_after_dbscan.png"
            spatial.visualize_dbscan_clusters(ca_df, p); files["dbscan_after"] = _rel(out, p)
            dbscan_row = {
                "Sample": sample, "ImageName": filename,
                "Total_Particles_Before": cb["total"], "Clusters_Count_Before": cb["n_clusters"],
                "Noise_Pct_Before": cb["noise_pct"],
                "Total_Particles_After": ca["total"], "Clusters_Count_After": ca["n_clusters"],
                "Noise_Pct_After": ca["noise_pct"],
            }
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{filename}: DBSCAN 분석 실패: {e}")
            logger.debug(traceback.format_exc())

    res = ImageResult(filename=filename, sample=sample, pixel_to_micro=p2m,
                      n_before=int(len(df_before)), n_after=int(len(df_after)), files=files)
    return res, df_before, df_after, boundary_row, agglom_row, dbscan_row


# ----------------------------------------------------------------------------- 샘플 단위
def _finish_sample(sample: str, befores: List[pd.DataFrame], out: Path, params: AnalysisParams,
                   acc: dict, result: BatchResult) -> None:
    df_b = pd.concat(befores, ignore_index=True)
    df_a = apply_after_filter(df_b, params.after_filter)          # 샘플 전체 기준 After
    df_b = df_b.copy()
    df_b["Filter"] = "Before"
    df_a["Filter"] = "After"
    acc["sample_sheets"][sample] = pd.concat([df_b, df_a], ignore_index=True)

    sfiles: Dict[str, str] = {}
    hdir = out / "histograms" / "After_SampleWide"
    for key, col, fname in (("hist_diameter", "Diameter (μm)", "Diameter"),
                            ("hist_circularity", "Circularity", "Circularity"),
                            ("hist_aspect", "Aspect Ratio", "Aspect_Ratio")):
        p = hdir / f"{sample}_{fname}_SampleWide_hist_comparison.png"
        viz.save_histogram_comparison(df_b[col], df_a[col], f"{fname}_SampleWide", p,
                                      title=f"{fname}_SampleWide Histogram Comparison: {sample} (After_SampleWide)",
                                      tag="After_SampleWide")
        sfiles[key] = _rel(out, p)

    sdir = out / "summary_images"
    font = DEFAULT_FONT if DEFAULT_FONT.is_file() else None
    p = sdir / f"{sample}_Before_summary.jpg"
    viz.save_summary_card(sample, stats.stats_for_card(df_b), p, "Before", font); sfiles["summary_before"] = _rel(out, p)
    p = sdir / f"{sample}_After_summary.jpg"
    viz.save_summary_card(sample, stats.stats_for_card(df_a), p, "After", font); sfiles["summary_after"] = _rel(out, p)
    result.sample_files[sample] = sfiles

    row = stats.summary_row(sample, df_b, df_a)
    acc["summary"].append(row)
    result.samples.append(row)
    for feature, sheet in stats.DIST_FEATURES:
        if feature in df_b.columns:
            acc["dist"].setdefault(sheet, []).append(stats.distribution_stats_row(sample, df_b, df_a, feature))


# ----------------------------------------------------------------------------- 배치
def run_batch(inp: BatchInput, params: AnalysisParams, engine: SegmentationEngine,
              progress_cb: Optional[ProgressCB] = None,
              cancel_check: Optional[CancelCheck] = None) -> BatchResult:
    """이미지 목록을 분석하고 output_dir 에 결과(이미지/Excel/PPT)를 쓴다. 이미지 단위 예외는 warnings 로."""
    out = Path(inp.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    excel_path = out / "particle_analysis_result.xlsx"
    result = BatchResult(output_dir=str(out), excel_path=str(excel_path), pptx_path=None)

    def progress(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(int(max(0, min(100, pct))), msg)
            except Exception:  # noqa: BLE001
                pass

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    engine.prepare()

    # 샘플 그룹핑 (자연 정렬)
    groups: Dict[str, List[Path]] = {}
    for p in inp.images:
        groups.setdefault(sample_name_of(Path(p).name), []).append(Path(p))
    sample_order = sorted(groups.keys(), key=natural_key)
    for s in sample_order:
        groups[s].sort(key=lambda p: natural_key(p.name))
    total = sum(len(v) for v in groups.values())
    if total == 0:
        result.warnings.append("분석할 이미지가 없습니다.")
        write_excel(excel_path, {}, {})
        progress(100, "no images")
        return result

    acc: dict = {"sample_sheets": {}, "summary": [], "dist": {}, "agglom": [], "dbscan": [], "boundary": []}
    done = 0
    was_cancelled = False
    progress(0, f"0/{total} 시작")

    try:
        for sample in sample_order:
            befores: List[pd.DataFrame] = []
            s_boundary: List[dict] = []
            s_agglom: List[dict] = []
            s_dbscan: List[dict] = []
            for path in groups[sample]:
                if cancelled():
                    raise BatchCancelled()
                try:
                    ir, df_b, _df_a, brow, arow, drow = _process_image(path, sample, out, params, engine, result.warnings)
                    result.images.append(ir)
                    befores.append(df_b)
                    if brow: s_boundary.append(brow)
                    if arow: s_agglom.append(arow)
                    if drow: s_dbscan.append(drow)
                except Exception as e:  # noqa: BLE001
                    result.warnings.append(f"{path.name}: {type(e).__name__}: {e}")
                    logger.warning("이미지 처리 실패 %s: %s\n%s", path.name, e, traceback.format_exc())
                done += 1
                progress(int(done / total * 85), f"{done}/{total} {path.name}")

            # 샘플 집계
            if s_agglom:
                acc["agglom"].extend(s_agglom); acc["agglom"].append(_avg_row(sample, s_agglom))
            if s_dbscan:
                acc["dbscan"].extend(s_dbscan); acc["dbscan"].append(_avg_row(sample, s_dbscan))
            if s_boundary:
                acc["boundary"].extend(s_boundary); acc["boundary"].append(_avg_row(sample, s_boundary))
            if befores:
                try:
                    _finish_sample(sample, befores, out, params, acc, result)
                except Exception as e:  # noqa: BLE001
                    result.warnings.append(f"샘플 {sample} 집계 실패: {type(e).__name__}: {e}")
                    logger.warning("샘플 집계 실패 %s: %s\n%s", sample, e, traceback.format_exc())
            else:
                result.warnings.append(f"샘플 {sample}: 유효 이미지가 없습니다.")
    except BatchCancelled:
        was_cancelled = True
        result.warnings.append(f"사용자 취소: {done}/{total} 이미지 처리 후 중단")
        logger.info("batch cancelled after %d/%d images", done, total)

    # --- Excel ---
    progress(90, "Excel 작성")
    stats_sheets: Dict[str, pd.DataFrame] = {}
    if acc["summary"]:
        stats_sheets["Summary"] = pd.DataFrame(acc["summary"])
    for sheet, rows in acc["dist"].items():
        stats_sheets[sheet] = stats.clean_dataframe(pd.DataFrame(rows))
    if acc["agglom"]:
        stats_sheets["Agglomeration Stats"] = stats.clean_dataframe(pd.DataFrame(acc["agglom"]))
    if acc["dbscan"]:
        stats_sheets["DBSCAN Stats"] = stats.clean_dataframe(pd.DataFrame(acc["dbscan"]))
    if acc["boundary"]:
        stats_sheets["Boundary Stats"] = stats.clean_dataframe(pd.DataFrame(acc["boundary"]))
    try:
        write_excel(excel_path, acc["sample_sheets"], stats_sheets)
    except Exception as e:  # noqa: BLE001
        result.warnings.append(f"Excel 작성 실패: {e}")
        logger.error("Excel 작성 실패: %s\n%s", e, traceback.format_exc())

    if was_cancelled:
        progress(100, "취소됨")
        return result

    # --- 정규분포 ---
    if params.output.make_normal_dist and excel_path.is_file():
        progress(93, "정규분포 플롯")
        try:
            written = viz.save_normal_distribution_plots(excel_path, out)
            result.extra_files["normal_dist"] = [_rel(out, Path(w)) for w in written]
        except Exception as e:  # noqa: BLE001
            result.warnings.append(f"정규분포 플롯 실패: {e}")
            logger.warning("정규분포 플롯 실패: %s\n%s", e, traceback.format_exc())

    # --- PPT ---
    if params.output.make_ppt and result.samples:
        progress(97, "PPT 작성")
        pptx_path = out / "particle_analysis_result.pptx"
        try:
            create_summary_ppt(out / "comparison_images", out / "summary_images", pptx_path)
            result.pptx_path = str(pptx_path)
        except Exception as e:  # noqa: BLE001
            result.warnings.append(f"PPT 작성 실패: {e}")
            logger.warning("PPT 작성 실패: %s\n%s", e, traceback.format_exc())

    progress(100, "완료")
    return result


__all__ = ["BatchInput", "ImageResult", "BatchResult", "run_batch"]

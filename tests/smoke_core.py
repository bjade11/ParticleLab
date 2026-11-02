"""particle_core 스모크 테스트: 합성 이미지 3장 → run_batch → 산출물 검증.

실행: python tests/smoke_core.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import pandas as pd  # noqa: E402

import particle_core  # noqa: E402
from particle_core import AnalysisParams, BatchInput, WatershedEngine, run_batch  # noqa: E402
from particle_core.report import STATS_SHEET_ORDER  # noqa: E402


def make_image(seed: int, w: int = 600, h: int = 500, n_circles: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 30, dtype=np.uint8)
    usable_h = int(h * 0.90)
    for i in range(n_circles):
        r = int(rng.integers(8, 30))
        cx = int(rng.integers(r + 8, w - r - 8))
        cy = int(rng.integers(r + 8, usable_h - r - 8))
        val = int(rng.integers(170, 230))
        cv2.circle(img, (cx, cy), r, val, -1)
    # 가짜 스케일바 스트립 (하단 ~6%) — bottom_crop_ratio 0.074 로 제거되어야 함
    img[int(h * 0.94):, :] = 255
    cv2.rectangle(img, (40, int(h * 0.95)), (200, int(h * 0.97)), 0, -1)
    noise = rng.normal(0, 8, size=img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def main() -> int:
    assert "PyQt5" not in sys.modules, "particle_core must not import PyQt5"
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()

    tmp = Path(tempfile.mkdtemp(prefix="plab_smoke_"))
    in_dir, out_dir = tmp / "input", tmp / "output"
    in_dir.mkdir()
    names = ["S1-01_5.0k.png", "S1-02_5.0k.png", "S2-01_5.0k.png"]
    for i, n in enumerate(names):
        ok, buf = cv2.imencode(".png", make_image(100 + i))
        assert ok
        buf.tofile(str(in_dir / n))

    params = AnalysisParams()
    params.spatial.enable_boundary = True
    params.spatial.enable_nnd = True
    params.spatial.enable_dbscan = True
    params.output.make_ppt = True
    params.output.make_normal_dist = True
    # round-trip 검증
    params2 = AnalysisParams.from_dict(params.to_dict())
    assert params2.to_dict() == params.to_dict()
    assert params2.watershed.blur_kernel == (3, 3)

    engine = WatershedEngine(params.watershed)
    progress_log = []
    t0 = time.perf_counter()
    res = run_batch(BatchInput(images=sorted(in_dir.iterdir()), output_dir=out_dir), params, engine,
                    progress_cb=lambda p, m: progress_log.append((p, m)))
    dt = time.perf_counter() - t0

    # --- 검증 ---
    assert Path(res.excel_path).is_file(), "excel missing"
    with pd.ExcelFile(res.excel_path) as xls:
        sheets = xls.sheet_names
        summary = pd.read_excel(xls, sheet_name="Summary")
        s1 = pd.read_excel(xls, sheet_name="S1")
        aggl = pd.read_excel(xls, sheet_name="Agglomeration Stats")
    expected = ["S1", "S2"] + STATS_SHEET_ORDER
    missing = [s for s in expected if s not in sheets]
    assert not missing, f"missing sheets: {missing} (have {sheets})"
    assert sheets[:2] == ["S1", "S2"], sheets
    assert sheets[2:] == STATS_SHEET_ORDER, sheets

    assert res.pptx_path and Path(res.pptx_path).is_file(), "pptx missing"
    assert len(res.images) == 3, res.images
    for ir in res.images:
        assert ir.n_before > 0, f"{ir.filename}: n_before == 0"
        assert ir.n_after <= ir.n_before
        for cat, rel in ir.files.items():
            assert (out_dir / rel).is_file(), f"{ir.filename} {cat}: {rel} missing"
        assert set(ir.files) >= {"labeled_before", "labeled_after", "contour_before", "contour_after", "comparison",
                                 "boundary_before", "boundary_after", "aggl_before", "aggl_after",
                                 "dbscan_before", "dbscan_after"}, ir.files.keys()
        assert abs(ir.pixel_to_micro - (27 / 136) / 5.0) < 1e-9
    for s, files in res.sample_files.items():
        for k, rel in files.items():
            assert (out_dir / rel).is_file(), f"{s} {k}: {rel} missing"
    assert set(res.sample_files) == {"S1", "S2"}
    assert len(res.samples) == 2 and {r["Sample"] for r in res.samples} == {"S1", "S2"}
    assert res.extra_files.get("normal_dist"), "normal_dist plots missing"
    for rel in res.extra_files["normal_dist"]:
        assert (out_dir / rel).is_file(), rel
    assert progress_log and progress_log[-1][0] == 100

    assert set(s1["Filter"].unique()) == {"Before", "After"}
    assert "ImageName" in s1.columns
    assert (aggl["ImageName"] == "ALL_IMAGES_AVG").sum() == 2

    # 취소 동작: 첫 이미지 후 취소 → Excel 만 작성
    out2 = tmp / "output_cancel"
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    res2 = run_batch(BatchInput(images=sorted(in_dir.iterdir()), output_dir=out2), params, engine, cancel_check=cancel)
    assert Path(res2.excel_path).is_file()
    assert res2.pptx_path is None
    assert any("취소" in w for w in res2.warnings), res2.warnings
    assert len(res2.images) == 1

    # make_engine
    from particle_core import make_engine
    assert make_engine(params).name == "watershed"
    try:
        p_ai = AnalysisParams.from_dict({"engine": "ai_hybrid"})
        make_engine(p_ai)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass

    print("SMOKE OK")
    print(f"  output_dir : {out_dir}")
    print(f"  elapsed    : {dt:.1f}s")
    print(f"  sheets     : {sheets}")
    for ir in res.images:
        print(f"  {ir.filename}: sample={ir.sample} p2m={ir.pixel_to_micro:.4f} before={ir.n_before} after={ir.n_after} files={len(ir.files)}")
    print("  summary    :")
    print(summary[["Sample", "Total_Particles_Before", "Total_Particles_After", "D50_Before", "D50_After"]].to_string(index=False))
    print(f"  normal_dist: {len(res.extra_files['normal_dist'])} files")
    print(f"  warnings   : {res.warnings}")
    print(f"  PyQt5 loaded: {'PyQt5' in sys.modules}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

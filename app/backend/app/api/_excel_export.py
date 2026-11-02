"""YOLO / U-Net / Hybrid 결과 → 엑셀 (.xlsx) 생성 공용 헬퍼.

- make_excel_simple : 단일 metrics (YOLO, U-Net)
- make_excel_3way   : 3-way metrics (Hybrid — YOLO/U-Net_low/Hybrid 비교)
"""
from __future__ import annotations

import base64
import io
from typing import Optional


METRIC_KEYS = ("Precision", "Recall", "Accuracy", "IoU", "F1",
               "mAP@0.5", "ParticleCountError")


def xlsx_data_url(data: bytes) -> str:
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _style_header_cell(cell, fill_hex: str = "1F5FD6") -> None:
    from openpyxl.styles import Font, PatternFill, Alignment
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=fill_hex)
    cell.alignment = Alignment(horizontal="center")


def _style_header_row(ws, n_cols: int, fill_hex: str = "1F5FD6") -> None:
    for c in range(1, n_cols + 1):
        _style_header_cell(ws.cell(row=1, column=c), fill_hex)
    ws.freeze_panes = "A2"


def _autosize(ws, max_width: int = 32) -> None:
    from openpyxl.utils import get_column_letter
    for col in ws.columns:
        length = max(
            (len(str(c.value)) for c in col if c.value is not None),
            default=8,
        )
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(
            length + 2, max_width
        )


def _format_numbers(ws, min_col: int, max_col: int, fmt: str = "0.0000") -> None:
    for row in ws.iter_rows(min_row=2, min_col=min_col, max_col=max_col):
        for cell in row:
            cell.number_format = fmt


# ─────────────── 단일 metrics (YOLO / U-Net) ───────────────


def make_excel_simple(
    title: str,
    params: dict,
    results: list[dict],
    summary: Optional[dict],
    *,
    extra_cols: Optional[list[tuple[str, str]]] = None,
) -> bytes:
    """단일 metrics 결과 → 4 시트 xlsx.

    시트:
        Parameters         : 모든 설정값
        Metrics_per_image  : filename + 7개 지표 + 입자 수 + (옵션) extra_cols
        Summary            : 전체 평균 (n_with_gt 명시)
        Image_diagnostics  : 이미지별 진단 (extra_cols 가 있을 때만)

    extra_cols 예: [("n_detections", "n_detections")] 또는 [("pred_pixels", "pred_pixels")]
    """
    from openpyxl import Workbook

    wb = Workbook()

    # ─── Sheet 1: Parameters ───
    ws = wb.active
    ws.title = "Parameters"
    ws.append(["Parameter", "Value"])
    _style_header_row(ws, 2)
    for k, v in params.items():
        val = str(v) if isinstance(v, bool) else v
        ws.append([str(k), val])
    _autosize(ws)

    # ─── Sheet 2: Metrics_per_image ───
    ws = wb.create_sheet("Metrics_per_image")
    headers = ["filename", *METRIC_KEYS, "n_gt_inst", "n_pred_inst"]
    extra_keys: list[str] = []
    if extra_cols:
        for key, label in extra_cols:
            headers.append(label)
            extra_keys.append(key)
    ws.append(headers)
    _style_header_row(ws, len(headers))
    for r in results:
        m = r.get("metrics")
        if m is None:
            # GT 매칭 안 된 이미지 — 지표 행 자체는 추가하되 NaN 으로
            row = [r.get("filename", ""), *(None for _ in METRIC_KEYS),
                   None, None]
            for k in extra_keys:
                row.append(r.get(k))
            ws.append(row)
            continue
        row = [
            r.get("filename", ""),
            float(m["Precision"]), float(m["Recall"]),
            float(m["Accuracy"]), float(m["IoU"]), float(m["F1"]),
            float(m["mAP@0.5"]), float(m["ParticleCountError"]),
            int(m["n_gt_inst"]), int(m["n_pred_inst"]),
        ]
        for k in extra_keys:
            row.append(r.get(k))
        ws.append(row)
    _format_numbers(ws, min_col=2, max_col=8)
    _autosize(ws)

    # ─── Sheet 3: Summary ───
    ws = wb.create_sheet("Summary")
    headers = ["scope", *METRIC_KEYS, "n_gt_inst (sum)", "n_pred_inst (sum)", "n_with_gt"]
    ws.append(headers)
    _style_header_row(ws, len(headers))
    if summary is not None:
        ws.append([
            title,
            float(summary["Precision"]), float(summary["Recall"]),
            float(summary["Accuracy"]), float(summary["IoU"]), float(summary["F1"]),
            float(summary["mAP@0.5"]), float(summary["ParticleCountError"]),
            int(summary["n_gt_inst"]), int(summary["n_pred_inst"]),
            int(summary["n_with_gt"]),
        ])
        _format_numbers(ws, min_col=2, max_col=8)
    else:
        ws.append(["(no GT — summary unavailable)"])
    _autosize(ws)

    # ─── Sheet 4: Image_diagnostics (있을 때만) ───
    if extra_cols:
        ws = wb.create_sheet("Image_diagnostics")
        diag_headers = ["filename", *[label for _, label in extra_cols]]
        ws.append(diag_headers)
        _style_header_row(ws, len(diag_headers))
        for r in results:
            ws.append([r.get("filename", ""),
                       *[r.get(k) for k, _ in extra_cols]])
        _autosize(ws)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ─────────────── 3-way metrics (Hybrid) ───────────────


def make_excel_3way(params: dict, results: list[dict], summaries: dict) -> bytes:
    """Hybrid 결과 → 4 시트 xlsx.

    시트:
        Parameters         : 알고리즘/추론 설정값
        Metrics_per_image  : (filename × method) 행, 7개 지표
        Summary            : 방법별 이미지 평균 (3행)
        Image_diagnostics  : 이미지별 진단 (K_unet, n_yolo, tau 등)
    """
    from openpyxl import Workbook

    wb = Workbook()

    # ─── Sheet 1: Parameters ───
    ws = wb.active
    ws.title = "Parameters"
    ws.append(["Parameter", "Value"])
    _style_header_row(ws, 2)
    for k, v in params.items():
        val = str(v) if isinstance(v, bool) else v
        ws.append([str(k), val])
    _autosize(ws)

    method_labels = [
        ("metrics_yolo",     "YOLO only"),
        ("metrics_unet_low", f"U-Net ≥ {params.get('low_thresh', 0.30)}"),
        ("metrics_hybrid",   "Hybrid"),
    ]
    summary_labels = [
        ("summary_yolo",     "YOLO only"),
        ("summary_unet_low", f"U-Net ≥ {params.get('low_thresh', 0.30)}"),
        ("summary_hybrid",   "Hybrid"),
    ]

    # ─── Sheet 2: Metrics_per_image ───
    ws = wb.create_sheet("Metrics_per_image")
    headers = ["filename", "method", *METRIC_KEYS, "n_gt_inst", "n_pred_inst"]
    ws.append(headers)
    _style_header_row(ws, len(headers))
    for r in results:
        fname = r.get("filename", "")
        for key, label in method_labels:
            m = r.get(key)
            if m is None:
                continue
            ws.append([
                fname, label,
                float(m["Precision"]), float(m["Recall"]),
                float(m["Accuracy"]), float(m["IoU"]), float(m["F1"]),
                float(m["mAP@0.5"]), float(m["ParticleCountError"]),
                int(m["n_gt_inst"]), int(m["n_pred_inst"]),
            ])
    _format_numbers(ws, min_col=3, max_col=9)
    _autosize(ws)

    # ─── Sheet 3: Summary ───
    ws = wb.create_sheet("Summary")
    headers = ["method", *METRIC_KEYS,
               "n_gt_inst (sum)", "n_pred_inst (sum)", "n_with_gt"]
    ws.append(headers)
    _style_header_row(ws, len(headers))
    for key, label in summary_labels:
        s = summaries.get(key)
        if s is None:
            continue
        ws.append([
            label,
            float(s["Precision"]), float(s["Recall"]),
            float(s["Accuracy"]), float(s["IoU"]), float(s["F1"]),
            float(s["mAP@0.5"]), float(s["ParticleCountError"]),
            int(s["n_gt_inst"]), int(s["n_pred_inst"]),
            int(s["n_with_gt"]),
        ])
    _format_numbers(ws, min_col=2, max_col=8)
    _autosize(ws)

    # ─── Sheet 4: Image_diagnostics ───
    ws = wb.create_sheet("Image_diagnostics")
    headers = ["filename", "n_instances (hybrid)",
               "K_unet_anchor", "n_yolo", "tau (used)", "pixel_csv_rows"]
    ws.append(headers)
    _style_header_row(ws, len(headers))
    for r in results:
        ws.append([
            r.get("filename", ""),
            int(r.get("n_instances", 0)),
            int(r.get("K_unet_anchor", 0)),
            int(r.get("n_yolo", 0)),
            float(r.get("tau", 0.0)),
            int(r.get("pixel_csv_rows", 0)),
        ])
    _format_numbers(ws, min_col=5, max_col=5, fmt="0.00")
    _autosize(ws)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

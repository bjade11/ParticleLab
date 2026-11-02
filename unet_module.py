"""Streamlit page for U-Net / U-Net++ training & evaluation.

논문 (Bals & Epple 2023) 의 weighted BCE 학습 + 같은 평가지표를 그대로 산출하고,
사용자 정의 weight map 코드도 입력받는다. PPT / CSV 내보내기 기능 포함.

진입점: `run_unet_page()`. viewer.py 의 모드 선택 라디오에서 호출된다.
"""
from __future__ import annotations

import io
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from scipy import ndimage as ndi

from evaluate import pixel_confusion, rand_error, safe_div
from unet_models import build_model, pick_device
from unet_trainer import (
    TrainConfig, TrainResult, deserialize_model, predict_with_state,
    serialize_model, train_unet,
)
from weight_maps import WeightMapConfig, compute_weight_map


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------


def _session_dir() -> Path:
    if "unet_tmpdir" not in st.session_state:
        d = Path(tempfile.mkdtemp(prefix="unet_"))
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "masks").mkdir(parents=True, exist_ok=True)
        st.session_state.unet_tmpdir = d
    return st.session_state.unet_tmpdir


def _save_uploads(uploads, dest: Path) -> int:
    n = 0
    for uf in uploads or []:
        ext = Path(uf.name).suffix.lower()
        if ext not in IMG_EXTS:
            continue
        with open(dest / uf.name, "wb") as f:
            f.write(uf.getbuffer())
        n += 1
    return n


def _stem_for_pairing(name: str) -> str:
    """SEM `_image` ↔ mask `_label` 페어링 규칙."""
    s = Path(name).stem
    for suf in ("_image", "_img", "_sem", "_orig"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    for suf in ("_label", "_mask", "_gt", "_labels"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def _build_pairs(images_dir: Path, masks_dir: Path
                 ) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    img_map = {}
    for q in images_dir.iterdir():
        if q.is_file() and q.suffix.lower() in IMG_EXTS:
            img_map.setdefault(_stem_for_pairing(q.name), str(q))
    msk_map = {}
    for q in masks_dir.iterdir():
        if q.is_file() and q.suffix.lower() in IMG_EXTS:
            msk_map.setdefault(_stem_for_pairing(q.name), str(q))
    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    pairs = [(s, img_map[s], msk_map[s]) for s in common]
    only_img = sorted(set(img_map) - set(msk_map))
    only_msk = sorted(set(msk_map) - set(img_map))
    return pairs, only_img, only_msk


# ---------------------------------------------------------------------------
# Metrics (논문 표 1 의 컬럼들)
# ---------------------------------------------------------------------------


def _binary_metrics(gt_bin: np.ndarray, pred_bin: np.ndarray) -> dict:
    tp, fp, fn, tn = pixel_confusion(gt_bin, pred_bin)
    acc = safe_div(tp + tn, tp + fp + fn + tn)
    # rand_error 는 인스턴스 라벨 필요
    gt_inst, _ = ndi.label(gt_bin)
    pr_inst, _ = ndi.label(pred_bin)
    re = rand_error(gt_inst, pr_inst)
    return {
        "Precision": safe_div(tp, tp + fp),
        "Recall":    safe_div(tp, tp + fn),
        "Accuracy":  acc,
        "IoU":       safe_div(tp, tp + fp + fn),
        "F1":        safe_div(2 * tp, 2 * tp + fp + fn),
        "PixelError": 1.0 - acc,
        "RandError": re,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# --- 입자 단위 지표 (논문 2.3절: mAP@0.5, Particle Count Error) ---

def _gt_instances(mask_bin: np.ndarray, min_area: int = 4):
    """connected components → 개별 인스턴스 마스크 리스트."""
    labels, n = ndi.label(mask_bin)
    out = []
    for i in range(1, n + 1):
        m = labels == i
        if int(m.sum()) >= min_area:
            out.append(m)
    return out


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _detection_match(gt_masks, pred_masks, pred_scores: np.ndarray,
                     iou_thr: float = 0.5):
    """한 이미지의 prediction → GT 매칭. score 내림차순 greedy.
    Returns (tp_flags, fp_flags, scores_sorted, n_gt)."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0:
        return (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float32), n_gt)
    order = np.argsort(-pred_scores)
    scores_s = pred_scores[order]
    masks_s = [pred_masks[i] for i in order]
    matched = [False] * n_gt
    tp = np.zeros(n_pred, dtype=bool)
    fp = np.zeros(n_pred, dtype=bool)
    for i, pm in enumerate(masks_s):
        best, bi = 0.0, -1
        for j, gm in enumerate(gt_masks):
            if matched[j]:
                continue
            v = _mask_iou(pm, gm)
            if v > best:
                best, bi = v, j
        if best >= iou_thr and bi >= 0:
            tp[i] = True
            matched[bi] = True
        else:
            fp[i] = True
    return tp, fp, scores_s, n_gt


def _compute_mask_map(records, iou_thr: float = 0.5) -> float:
    """모든 이미지의 detection 을 합쳐 11-point AP. 단일 클래스라 mAP=AP."""
    if not records:
        return 0.0
    all_tp = np.concatenate([r[0].astype(np.float32) for r in records])
    all_fp = np.concatenate([r[1].astype(np.float32) for r in records])
    all_s = np.concatenate([r[2] for r in records])
    n_gt_total = sum(int(r[3]) for r in records)
    if n_gt_total == 0 or all_s.size == 0:
        return 0.0
    order = np.argsort(-all_s)
    tp = all_tp[order]
    fp = all_fp[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    rec = cum_tp / n_gt_total
    pre = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        valid = rec >= r
        ap += float(pre[valid].max()) if valid.any() else 0.0
    return ap / 11.0


def _instance_score_map(prob: np.ndarray, mask: np.ndarray) -> float:
    """입자 단위 confidence: 그 영역의 probability map 평균."""
    if not mask.any():
        return 0.0
    return float(prob[mask].mean())


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _colorize_instance(mask_bin: np.ndarray, seed: int = 42) -> np.ndarray:
    labels, n = ndi.label(mask_bin)
    h, w = mask_bin.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if n == 0:
        return canvas
    rng = np.random.default_rng(seed)
    colors = rng.integers(60, 255, size=(n, 3), dtype=np.uint8)
    for i in range(1, n + 1):
        canvas[labels == i] = colors[i - 1]
    return canvas


def _overlay(image_rgb: np.ndarray, mask_bin: np.ndarray,
             alpha: float = 0.45) -> np.ndarray:
    out = image_rgb.copy()
    if not mask_bin.any():
        return out
    color = np.array([56, 189, 248], dtype=np.uint8)  # sky blue
    out[mask_bin] = (alpha * color + (1 - alpha) * out[mask_bin]).astype(np.uint8)
    # contour
    u8 = mask_bin.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    cv2.drawContours(out, cnts, -1, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def _png_bytes(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        ok, png = cv2.imencode(".png", arr.astype(np.uint8))
    else:
        ok, png = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return png.tobytes() if ok else b""


# ---------------------------------------------------------------------------
# PPT export
# ---------------------------------------------------------------------------


def _build_ppt(records: List[dict], settings: dict, history: list) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Cover
    cover = prs.slides.add_slide(blank)
    tb = cover.shapes.add_textbox(Inches(0.5), Inches(2.0), Inches(12.3), Inches(1.0))
    tf = tb.text_frame
    tf.text = "Particle Lab — U-Net Training Results"
    tf.paragraphs[0].font.size = Pt(36)
    tf.paragraphs[0].font.bold = True

    sub = cover.shapes.add_textbox(Inches(0.5), Inches(3.5), Inches(12.3), Inches(2.5))
    sub.text_frame.text = (
        f"Method: {settings['method']}   |   "
        f"Epochs: {settings['epochs']}   |   "
        f"Image size: {settings['image_size']}   |   "
        f"Batch: {settings['batch_size']}   |   "
        f"LR: {settings['lr']}\n"
        f"Weight map: {settings['wmap_mode']}   "
        f"(w0={settings['w0']}, σ={settings['sigma']}, γ={settings['gamma']})\n"
        f"Loss: {settings['loss_label']}\n"
        f"Best val loss: {settings['best_val']:.4f}   |   "
        f"Elapsed: {settings['elapsed']:.1f}s   |   "
        f"Device: {settings['device']}"
    )
    for p in sub.text_frame.paragraphs:
        p.font.size = Pt(14)

    def fit(box_w, box_h, iw, ih):
        if iw <= 0 or ih <= 0: return int(box_w), int(box_h)
        ar_b, ar_i = box_w / box_h, iw / ih
        if ar_i >= ar_b:
            return int(box_w), int(box_w / ar_i)
        return int(box_h * ar_i), int(box_h)

    def place(slide, png, x, y, bw, bh, iw, ih):
        if not png: return
        w, h = fit(bw, bh, iw, ih)
        cx = int(x + (bw - w) / 2)
        cy = int(y + (bh - h) / 2)
        slide.shapes.add_picture(io.BytesIO(png), cx, cy, width=w, height=h)

    cell_w, cell_h = Inches(6.3), Inches(2.7)
    pos = [
        (Inches(0.35), Inches(1.55)),
        (Inches(6.75), Inches(1.55)),
        (Inches(0.35), Inches(4.55)),
        (Inches(6.75), Inches(4.55)),
    ]
    labels = ["Original SEM", "GT (binary mask)",
              "U-Net prediction (instance)", "Overlay"]
    keys = ["orig_png", "gt_png", "pred_png", "over_png"]

    for r in records:
        s = prs.slides.add_slide(blank)
        tb = s.shapes.add_textbox(Inches(0.3), Inches(0.15), Inches(12.7), Inches(0.45))
        tf = tb.text_frame
        tf.text = r["stem"]
        tf.paragraphs[0].font.size = Pt(20)
        tf.paragraphs[0].font.bold = True

        tb2 = s.shapes.add_textbox(Inches(0.3), Inches(0.60), Inches(12.7), Inches(0.30))
        tf2 = tb2.text_frame
        tf2.text = (f"Settings:  {settings['method']}  ·  "
                    f"image={settings['image_size']}  ·  "
                    f"weight={settings['wmap_mode']}  ·  "
                    f"loss={settings['loss_label']}  |  "
                    f"GT/Pred={r['n_gt']}/{r['n_pred']}")
        tf2.paragraphs[0].font.size = Pt(12)
        tf2.paragraphs[0].font.bold = True

        if r.get("has_gt"):
            tb3 = s.shapes.add_textbox(Inches(0.3), Inches(0.92), Inches(12.7), Inches(0.30))
            tf3 = tb3.text_frame
            tf3.text = (
                f"픽셀 단위:  Accuracy={r['acc']:.4f}  |  Precision={r['prec']:.3f}  |  "
                f"Recall={r['rec']:.3f}  |  IoU={r['iou']:.3f}  |  "
                f"F1={r['f1']:.3f}  |  Pixel Err={r['pix_err']*100:.2f}%  |  "
                f"Rand Err={r['rand_err']*100:.2f}%"
            )
            tf3.paragraphs[0].font.size = Pt(11)
            tf3.paragraphs[0].font.color.rgb = RGBColor(30, 90, 200)

            # 입자 단위 (논문 2.3절): Count Error + mAP@0.5 + GT/Pred 개수
            ce = r.get("count_err", 0.0)
            mp = r.get("map50", 0.0)
            tb4 = s.shapes.add_textbox(Inches(0.3), Inches(1.22), Inches(12.7), Inches(0.30))
            tf4 = tb4.text_frame
            tf4.text = (
                f"입자 단위:  mAP@0.5={mp:.3f}   |   "
                f"Particle Count Error={ce*100:.2f}%   |   "
                f"총 GT 입자={r.get('n_gt', 0)}   |   "
                f"총 Pred 입자={r.get('n_pred', 0)}"
            )
            tf4.paragraphs[0].font.size = Pt(11)
            tf4.paragraphs[0].font.color.rgb = RGBColor(120, 60, 160)

        iw = r.get("img_w", 0); ih = r.get("img_h", 0)
        for (x, y), lbl, k in zip(pos, labels, keys):
            png = r.get(k, b"")
            place(s, png, x, y, cell_w, cell_h, iw, ih)
            cap = s.shapes.add_textbox(x, y - Inches(0.28), cell_w, Inches(0.25))
            cap.text_frame.text = lbl
            cap.text_frame.paragraphs[0].font.size = Pt(10)
            cap.text_frame.paragraphs[0].font.bold = True

    # Loss curve slide
    if history:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
            epochs = [h["epoch"] for h in history]
            ax.plot(epochs, [h["train_loss"] for h in history], label="train_loss")
            ax.plot(epochs, [h["val_loss"] for h in history], label="val_loss")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss")
            ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_title(f"{settings['method']} training curve")
            buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            s = prs.slides.add_slide(blank)
            tb = s.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(12.7), Inches(0.5))
            tb.text_frame.text = "Training curve"
            tb.text_frame.paragraphs[0].font.size = Pt(20)
            tb.text_frame.paragraphs[0].font.bold = True
            s.shapes.add_picture(buf, Inches(1.5), Inches(1.0),
                                 width=Inches(10.3), height=Inches(5.5))
        except Exception:
            pass

    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def run_unet_page() -> None:
    st.title("Particle Lab — U-Net / U-Net++ Trainer")
    st.caption(
        "Bals & Epple (RSC Adv. 2023) 의 weighted BCE 방식. "
        "이진 마스크 GT 와 SEM 이미지 페어를 학습하고, 동일 지표 (Precision / Recall / "
        "Accuracy / IoU / F1 / Pixel Error / Rand Error) 로 평가한다."
    )

    tmp = _session_dir()
    images_dir = tmp / "images"
    masks_dir = tmp / "masks"

    # ---------------- Sidebar ----------------
    with st.sidebar:
        # 학습된 모델 불러오기 — 학습을 건너뛰고 바로 추론 가능.
        st.header("📁 학습된 모델 (선택)")
        st.caption(
            "이전에 저장한 `.pt` 모델을 불러오면 학습 단계를 건너뛰고 "
            "곧바로 새 이미지로 추론할 수 있다."
        )
        up_model = st.file_uploader(
            "모델(.pt) 불러오기",
            type=["pt", "pth"],
            accept_multiple_files=False,
            key="unet_model_upload",
        )
        if up_model is not None:
            if st.button("이 모델 적용", key="unet_apply_model"):
                try:
                    payload = deserialize_model(up_model.getvalue())
                    fake_cfg = TrainConfig(
                        method=payload["method"],
                        image_size=int(payload["image_size"]),
                    )
                    fake_result = TrainResult(
                        history=payload.get("history", []) or [],
                        best_val=float("nan"),
                        elapsed=0.0,
                        device="(loaded)",
                        state_dict=payload["state_dict"],
                        config=fake_cfg,
                    )
                    saved_settings = payload.get("settings") or {}
                    st.session_state["unet_train_result"] = {
                        "result": fake_result,
                        "pairs": [],  # 학습 페어는 없음
                        "settings": dict(
                            method=payload["method"],
                            image_size=int(payload["image_size"]),
                            epochs=saved_settings.get("epochs", 0),
                            batch_size=saved_settings.get("batch_size", 0),
                            lr=saved_settings.get("lr", 0.0),
                            wmap_mode=saved_settings.get("wmap_mode", "(loaded)"),
                            w0=saved_settings.get("w0", 0.0),
                            sigma=saved_settings.get("sigma", 0.0),
                            gamma=saved_settings.get("gamma", 0.0),
                            loss_label=saved_settings.get("loss_label", "(loaded)"),
                            best_val=saved_settings.get("best_val", float("nan")),
                            elapsed=0.0,
                            device="(loaded)",
                            loaded_from=up_model.name,
                        ),
                    }
                    st.success(
                        f"✅ 모델 로드됨: {payload['method']}, "
                        f"image_size={payload['image_size']}, "
                        f"history={len(fake_result.history)} epochs"
                    )
                except Exception as e:
                    st.error(f"로드 실패: {e}")
        st.divider()

        st.header("📦 데이터")
        st.caption("SEM 이미지 폴더 + 이진 마스크 폴더. "
                   "stem 매칭: `_image` ↔ `_label`/`_mask`/`_gt`.")
        if st.button("🗑 모든 업로드 삭제", key="unet_clear_all"):
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(masks_dir, ignore_errors=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
        ifs = st.file_uploader(
            "SEM 이미지 (다중 / 폴더 가능)",
            type=[e[1:] for e in IMG_EXTS],
            accept_multiple_files=True, key="unet_images",
        )
        _save_uploads(ifs, images_dir)
        n_img = sum(1 for _ in images_dir.iterdir())
        st.caption(f"📁 SEM 이미지 {n_img}개")

        mfs = st.file_uploader(
            "이진 마스크 (다중 / 폴더 가능)",
            type=[e[1:] for e in IMG_EXTS],
            accept_multiple_files=True, key="unet_masks",
        )
        _save_uploads(mfs, masks_dir)
        n_msk = sum(1 for _ in masks_dir.iterdir())
        st.caption(f"📁 마스크 {n_msk}개")

        st.divider()
        st.header("⚙️ 모델 / 학습")
        method = st.selectbox("Architecture", ["U-Net++", "U-Net"], index=0,
                              key="unet_method")
        epochs = st.slider("Epochs", 1, 200, 30, 1, key="unet_epochs")
        image_size = st.select_slider("Image size", [128, 192, 256, 320, 384, 512],
                                      value=256, key="unet_imgsz")
        batch_size = st.slider("Batch size", 1, 16, 4, 1, key="unet_bs")
        lr = st.select_slider("Learning rate",
                              [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3],
                              value=1e-3, key="unet_lr")
        val_frac = st.slider("Validation 비율", 0.0, 0.5, 0.2, 0.05, key="unet_val")
        augment = st.checkbox("Data augmentation (flip)", value=True,
                              key="unet_aug")

        st.divider()
        st.header("🏋️ Loss / Weight map")
        st.caption("논문 식 (1): w(x) = w₀·exp(−(d₁+d₂)²/2σ²) + γ·I_c(x) + 1")
        wmap_mode = st.radio(
            "Weight map mode",
            ["none", "distance", "intensity", "custom"],
            index=2,
            help=(
                "• none: 기본 BCE\n"
                "• distance: SE 이미지용 거리 기반 가중치\n"
                "• intensity: STEM 이미지용 거리+강도 기반 가중치\n"
                "• custom: 직접 코드 입력"
            ),
            key="unet_wmode",
        )
        w0 = st.number_input("w₀", 0.0, 100.0, 10.0, 0.5, key="unet_w0")
        sigma = st.number_input("σ", 0.1, 50.0, 7.0, 0.5, key="unet_sigma")
        gamma = st.number_input("γ (intensity 모드)",
                                0.0, 20.0, 2.0, 0.5, key="unet_gamma")

        custom_code = ""
        if wmap_mode == "custom":
            st.caption(
                "사용 가능한 변수: `mask` (HxW bool), `image` (HxW float, 0~1), "
                "`np`, `ndi`, `w0`, `sigma`, `gamma`. 결과를 변수 `w` 에 할당."
            )
            custom_code = st.text_area(
                "Custom weight map code",
                value=(
                    "# 논문 식과 동일\n"
                    "d = ndi.distance_transform_edt(~mask)\n"
                    "w = w0 * np.exp(-(d**2) / (2*sigma**2)) + gamma * image + 1\n"
                ),
                height=160,
                key="unet_custom_code",
            )

        st.divider()
        st.header("🧮 Loss combination")
        use_dice = st.checkbox("BCE + Dice 혼합", value=True, key="unet_use_dice")
        dice_weight = st.slider("Dice 가중치", 0.0, 1.0, 0.5, 0.05,
                                key="unet_dice_w") if use_dice else 0.0

    # ---------------- Main body ----------------
    pairs, only_img, only_msk = _build_pairs(images_dir, masks_dir)
    if not pairs:
        st.warning("⬅️ 사이드바에서 SEM 이미지와 이진 마스크를 업로드해줘. "
                   "stem 이 매칭되어야 한다 (`_image` ↔ `_label`).")
        if only_img:
            st.caption(f"이미지만 있는 stem ({len(only_img)}): {only_img[:8]}…")
        if only_msk:
            st.caption(f"마스크만 있는 stem ({len(only_msk)}): {only_msk[:8]}…")
        return

    st.success(f"✅ {len(pairs)} 페어 매칭됨")
    with st.expander("페어 샘플 (앞 10개)"):
        st.dataframe(pd.DataFrame(
            [{"stem": s, "image": Path(ip).name, "mask": Path(mp).name}
             for s, ip, mp in pairs[:10]]
        ), use_container_width=True)

    # weight map 미리보기
    st.subheader("🧪 Weight map 미리보기 (첫 페어)")
    cfg = WeightMapConfig(mode=wmap_mode, w0=w0, sigma=sigma, gamma=gamma,
                          custom_code=custom_code)
    try:
        s0, ip0, mp0 = pairs[0]
        img0 = np.asarray(Image.open(ip0).convert("L"))
        msk0 = np.asarray(Image.open(mp0).convert("L")) > 127
        wmap0 = compute_weight_map(msk0, img0, cfg)
        c1, c2, c3 = st.columns(3)
        c1.image(img0, caption=f"SEM ({s0})", use_container_width=True)
        c2.image((msk0.astype(np.uint8) * 255), caption="GT mask",
                 use_container_width=True)
        # weight map 시각화: 0~max → 0~255
        wmin, wmax = float(wmap0.min()), float(wmap0.max())
        rng = max(wmax - wmin, 1e-9)
        wvis = ((wmap0 - wmin) / rng * 255).astype(np.uint8)
        c3.image(wvis, caption=f"Weight map (min={wmin:.2f}, max={wmax:.2f})",
                 use_container_width=True)
    except Exception as e:
        st.error(f"Weight map 미리보기 실패: {e}")

    # ---------------- 학습 ----------------
    st.divider()
    st.subheader("🚀 학습 시작")
    if st.button("Start training", type="primary", key="unet_train_btn"):
        cfg_w = WeightMapConfig(mode=wmap_mode, w0=w0, sigma=sigma, gamma=gamma,
                                custom_code=custom_code)
        train_cfg = TrainConfig(
            method=method, epochs=epochs, batch_size=batch_size,
            lr=float(lr), image_size=image_size, val_fraction=val_frac,
            augment=augment, use_dice=use_dice, dice_weight=dice_weight,
            wmap=cfg_w,
        )

        progress = st.progress(0.0, text="시작…")
        loss_box = st.empty()
        loss_history: list[dict] = []

        def cb(epoch: int, total: int, tl: float, vl: float):
            loss_history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl})
            progress.progress(epoch / total,
                              text=f"epoch {epoch}/{total} — "
                                   f"train={tl:.4f}, val={vl:.4f}")
            try:
                df_h = pd.DataFrame(loss_history).set_index("epoch")
                loss_box.line_chart(df_h)
            except Exception:
                pass

        with st.spinner("학습 중… (시간이 걸릴 수 있음)"):
            result = train_unet(pairs, train_cfg, progress_cb=cb)
        progress.empty()
        st.session_state["unet_train_result"] = {
            "result": result, "pairs": pairs, "settings": dict(
                method=method, epochs=epochs, image_size=image_size,
                batch_size=batch_size, lr=lr, wmap_mode=wmap_mode,
                w0=w0, sigma=sigma, gamma=gamma,
                loss_label=("BCE+Dice" if use_dice else "BCE"),
                best_val=result.best_val, elapsed=result.elapsed,
                device=result.device,
            ),
        }
        st.success(f"✅ 학습 완료 · best val loss = {result.best_val:.4f} "
                   f"({result.elapsed:.1f}s, device={result.device})")

    # ---------------- 평가 + 내보내기 ----------------
    if "unet_train_result" not in st.session_state:
        st.info("학습을 마치거나, 사이드바에서 저장된 모델(.pt)을 불러오면 "
                "평가/내보내기 섹션이 활성화됩니다.")
        return

    bundle = st.session_state["unet_train_result"]
    result: TrainResult = bundle["result"]
    train_pairs = bundle["pairs"]
    settings = bundle["settings"]

    # ---- 모델 저장 ----
    st.divider()
    st.subheader("💾 학습된 모델 저장 / 다운로드")
    st.caption("아래 .pt 파일은 사이드바 '학습된 모델 불러오기' 로 다시 적용 가능.")
    try:
        model_bytes = serialize_model(
            result.state_dict,
            method=settings["method"],
            image_size=settings["image_size"],
            settings=settings,
            history=result.history,
        )
        st.download_button(
            "📥 모델 다운로드 (.pt)",
            data=model_bytes,
            file_name=f"unet_{settings['method'].replace('+', 'p')}_model.pt",
            mime="application/octet-stream",
            key="unet_model_dl",
        )
    except Exception as e:
        st.error(f"모델 직렬화 실패: {e}")

    # ---- 추론 / 평가 / 내보내기 ----
    st.divider()
    st.subheader("📊 새 이미지로 분할 / 평가 + PPT·CSV 내보내기")

    # 추론 데이터 소스: 새 이미지 업로드 vs 학습 페어 그대로
    source = st.radio(
        "추론할 이미지 소스",
        ["새 이미지 업로드 (권장)", "학습에 사용한 페어 그대로"],
        index=0, horizontal=True, key="unet_eval_source",
    )

    if source.startswith("학습"):
        if not train_pairs:
            st.warning("불러온 모델이라 학습 페어가 없어. '새 이미지 업로드' 를 선택해줘.")
            return
        eval_pairs: list[tuple[str, str, str | None]] = [
            (s, ip, mp) for s, ip, mp in train_pairs
        ]
        st.caption(f"✅ 학습 페어 {len(eval_pairs)}개에 대해 추론한다.")
    else:
        # 새 이미지 + (선택) GT 폴더
        inf_dir = tmp / "infer_images"
        inf_dir.mkdir(parents=True, exist_ok=True)
        inf_msk_dir = tmp / "infer_masks"
        inf_msk_dir.mkdir(parents=True, exist_ok=True)

        cols = st.columns(2)
        with cols[0]:
            if st.button("🗑 새 이미지 클리어", key="unet_inf_clear_img"):
                shutil.rmtree(inf_dir, ignore_errors=True)
                inf_dir.mkdir(parents=True, exist_ok=True)
            new_imgs = st.file_uploader(
                "분할할 새 SEM 이미지 (필수)",
                type=[e[1:] for e in IMG_EXTS],
                accept_multiple_files=True, key="unet_inf_imgs",
            )
            _save_uploads(new_imgs, inf_dir)
            n_new = sum(1 for _ in inf_dir.iterdir())
            st.caption(f"📁 새 이미지 {n_new}개")
        with cols[1]:
            if st.button("🗑 새 GT 클리어", key="unet_inf_clear_msk"):
                shutil.rmtree(inf_msk_dir, ignore_errors=True)
                inf_msk_dir.mkdir(parents=True, exist_ok=True)
            new_msks = st.file_uploader(
                "GT 이진 마스크 (선택, 있으면 지표 계산)",
                type=[e[1:] for e in IMG_EXTS],
                accept_multiple_files=True, key="unet_inf_msks",
            )
            _save_uploads(new_msks, inf_msk_dir)
            n_new_m = sum(1 for _ in inf_msk_dir.iterdir())
            st.caption(f"📁 새 GT 마스크 {n_new_m}개")

        # 페어링 — 이미지마다 매칭되는 GT 가 있으면 같이, 없으면 None
        eval_pairs = []
        msk_index: dict[str, str] = {}
        for q in inf_msk_dir.iterdir():
            if q.is_file() and q.suffix.lower() in IMG_EXTS:
                msk_index.setdefault(_stem_for_pairing(q.name), str(q))
        for q in sorted(inf_dir.iterdir()):
            if q.is_file() and q.suffix.lower() in IMG_EXTS:
                stem = _stem_for_pairing(q.name)
                eval_pairs.append((stem, str(q), msk_index.get(stem)))

        if not eval_pairs:
            st.info("⬆️ 위에 새 SEM 이미지를 업로드해줘.")
            return
        n_with_gt = sum(1 for _, _, mp in eval_pairs if mp)
        st.caption(
            f"✅ 추론 대기: {len(eval_pairs)}장 (GT 매칭: {n_with_gt}장)"
        )

    if st.button("Run inference + Export", type="primary",
                 key="unet_eval_btn"):
        records: list[dict] = []
        agg = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "rand_sum": 0.0, "n": 0,
               "n_gt_total": 0, "n_pred_total": 0, "count_err_sum": 0.0}
        det_records: list = []  # 입자 단위 mAP 계산용
        progress = st.progress(0.0, text="추론 시작…")
        for k, (stem, ip, mp) in enumerate(eval_pairs, start=1):
            img = np.asarray(Image.open(ip).convert("L"))
            prob, binary = predict_with_state(
                result.state_dict, settings["method"], img,
                image_size=settings["image_size"],
            )
            pr_bin = binary > 0
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            mask_canvas = _colorize_instance(pr_bin)
            over = _overlay(img_rgb, pr_bin, alpha=0.45)
            n_pred = len(_gt_instances(pr_bin))

            rec = {
                "stem": stem,
                "img_w": int(img.shape[1]), "img_h": int(img.shape[0]),
                "n_pred": n_pred,
                "orig_png": _png_bytes(img),
                "pred_png": _png_bytes(mask_canvas),
                "over_png": _png_bytes(over),
            }

            if mp:
                # GT 있음 → 픽셀 + 입자 단위 지표 모두 산출
                gt = np.asarray(Image.open(mp).convert("L")) > 127
                m = _binary_metrics(gt, pr_bin)
                agg["TP"] += m["TP"]; agg["FP"] += m["FP"]
                agg["FN"] += m["FN"]; agg["TN"] += m["TN"]
                agg["rand_sum"] += m["RandError"]; agg["n"] += 1

                gt_insts = _gt_instances(gt)
                pr_insts = _gt_instances(pr_bin)
                n_gt = len(gt_insts)
                count_err = abs(n_pred - n_gt) / max(n_gt, 1)
                pr_scores = np.array(
                    [_instance_score_map(prob, mm) for mm in pr_insts],
                    dtype=np.float32,
                )
                tp_d, fp_d, scores_d, ngt_d = _detection_match(
                    gt_insts, pr_insts, pr_scores, iou_thr=0.5
                )
                det_records.append((tp_d, fp_d, scores_d, ngt_d))
                map50_single = _compute_mask_map(
                    [(tp_d, fp_d, scores_d, ngt_d)], iou_thr=0.5
                )
                agg["n_gt_total"] += n_gt
                agg["n_pred_total"] += n_pred
                agg["count_err_sum"] += count_err

                rec.update({
                    "has_gt": True,
                    "n_gt": n_gt,
                    "count_err": count_err,
                    "map50": map50_single,
                    "acc": m["Accuracy"], "prec": m["Precision"],
                    "rec": m["Recall"], "iou": m["IoU"], "f1": m["F1"],
                    "pix_err": m["PixelError"], "rand_err": m["RandError"],
                    "gt_png": _png_bytes((gt.astype(np.uint8) * 255)),
                })
            else:
                rec.update({
                    "has_gt": False,
                    "n_gt": 0,
                    "gt_png": b"",
                })
            records.append(rec)

            txt = f"({k}/{len(eval_pairs)}) {stem}"
            if rec.get("has_gt"):
                txt += f" — Acc={rec['acc']:.4f}"
            else:
                txt += f" — n_pred={n_pred}"
            progress.progress(k / len(eval_pairs), text=txt)
        progress.empty()

        if agg["n"] == 0:
            st.info("GT 가 매칭된 이미지가 없어 지표 합산은 생략. 추론 시각화만 PPT/CSV 에 들어감.")
            map50_total = 0.0; mean_count_err = 0.0
            TP = FP = FN = TN = 0; acc_total = 0.0
        else:
            # 폴더 합산 — 픽셀 단위
            TP, FP, FN, TN = agg["TP"], agg["FP"], agg["FN"], agg["TN"]
            acc_total = safe_div(TP + TN, TP + FP + FN + TN)
            st.markdown("**픽셀 단위 (전체 합산, GT 매칭된 이미지만)**")
            c = st.columns(4)
            c[0].metric("Precision", f"{safe_div(TP, TP+FP):.3f}")
            c[1].metric("Recall",    f"{safe_div(TP, TP+FN):.3f}")
            c[2].metric("Accuracy",  f"{acc_total:.4f}")
            c[3].metric("IoU",       f"{safe_div(TP, TP+FP+FN):.3f}")
            c = st.columns(4)
            c[0].metric("F1",         f"{safe_div(2*TP, 2*TP+FP+FN):.3f}")
            c[1].metric("Pixel Err",  f"{(1-acc_total)*100:.2f} %")
            c[2].metric("Rand Err",   f"{agg['rand_sum']/max(agg['n'],1)*100:.2f} %")
            c[3].metric("Images",     f"{agg['n']}")

            # 폴더 합산 — 입자 단위 (논문 2.3절)
            map50_total = _compute_mask_map(det_records, iou_thr=0.5)
            mean_count_err = agg["count_err_sum"] / max(agg["n"], 1)
            st.markdown("**입자 단위 (논문 2.3절, GT 매칭된 이미지만)**")
            c = st.columns(4)
            c[0].metric("mAP@0.5", f"{map50_total:.3f}")
            c[1].metric("Particle Count Error (avg)",
                        f"{mean_count_err*100:.2f} %",
                        help="이미지별 |N_pred − N_gt| / N_gt 의 평균")
            c[2].metric("총 GT 입자",   f"{agg['n_gt_total']}")
            c[3].metric("총 Pred 입자", f"{agg['n_pred_total']}")

        # CSV — GT 있는 행에는 모든 컬럼, 없는 행은 metric 칸 비움
        rows_csv: list[dict] = []
        for r in records:
            base = {
                "stem": r["stem"],
                "n_gt": r.get("n_gt", "") if r.get("has_gt") else "",
                "n_pred": r["n_pred"],
            }
            if r.get("has_gt"):
                base.update({
                    "ParticleCountError": round(r["count_err"], 4),
                    "mAP@0.5": round(r["map50"], 4),
                    "Precision": round(r["prec"], 4),
                    "Recall": round(r["rec"], 4),
                    "Accuracy": round(r["acc"], 4),
                    "IoU": round(r["iou"], 4),
                    "F1": round(r["f1"], 4),
                    "PixelError": round(r["pix_err"], 4),
                    "RandError": round(r["rand_err"], 4),
                })
            else:
                for k_ in ("ParticleCountError", "mAP@0.5", "Precision",
                           "Recall", "Accuracy", "IoU", "F1",
                           "PixelError", "RandError"):
                    base[k_] = ""
            rows_csv.append(base)

        df = pd.DataFrame(rows_csv)

        # 폴더 합산 행 — GT 매칭된 이미지가 있을 때만 의미 있음
        if agg["n"] > 0:
            total_row = {
                "stem": "__TOTAL__",
                "n_gt": agg["n_gt_total"],
                "n_pred": agg["n_pred_total"],
                "ParticleCountError": round(mean_count_err, 4),
                "mAP@0.5": round(map50_total, 4),
                "Precision": round(safe_div(TP, TP + FP), 4),
                "Recall":    round(safe_div(TP, TP + FN), 4),
                "Accuracy":  round(acc_total, 4),
                "IoU":       round(safe_div(TP, TP + FP + FN), 4),
                "F1":        round(safe_div(2 * TP, 2 * TP + FP + FN), 4),
                "PixelError": round(1 - acc_total, 4),
                "RandError": round(agg["rand_sum"] / max(agg["n"], 1), 4),
            }
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📊 Summary CSV 다운로드",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"unet_{settings['method']}_results.csv",
            mime="text/csv",
        )

        # PPT
        with st.spinner("PPT 생성 중…"):
            ppt_bytes = _build_ppt(records, settings, result.history)
        st.download_button(
            "📥 PPT 다운로드",
            data=ppt_bytes,
            file_name=f"unet_{settings['method']}_results.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

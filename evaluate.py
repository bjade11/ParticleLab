"""Compute all requested metrics by comparing predictions to ground-truth labels.

Pixel-level (binarized particle vs background):
  TP, FP, FN, TN, Precision, Recall, Accuracy, IoU, F1, Pixel Error, Rand Error

Detection-level (per-instance bbox/mask matching):
  Precision, Recall, AP, mAP, mAP@0.5, mAP@0.5:0.95
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ----------------------------- IO helpers ------------------------------------

def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_yolo_seg_label(label_path: Path, w: int, h: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Parse YOLOv8/v11 polygon labels. Returns (instance_map, polygons, classes, scores).

    If a confidence field is appended (predictions), it is parsed as score; else 1.0.
    """
    inst = np.zeros((h, w), dtype=np.uint16)
    polys: list[np.ndarray] = []
    cls_list: list[int] = []
    score_list: list[float] = []

    if not label_path.exists() or label_path.stat().st_size == 0:
        return inst, polys, np.array(cls_list, dtype=int), np.array(score_list, dtype=float)

    for i, line in enumerate(label_path.read_text().strip().splitlines(), start=1):
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]

        score = 1.0
        if len(nums) % 2 == 1:
            score = float(nums[-1])
            nums = nums[:-1]

        coords = np.array(nums, dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        poly = coords.astype(np.int32)

        cv2.fillPoly(inst, [poly], int(i))
        polys.append(poly)
        cls_list.append(cls)
        score_list.append(score)

    return inst, polys, np.array(cls_list, dtype=int), np.array(score_list, dtype=float)


def polygons_to_bboxes(polys: list[np.ndarray]) -> np.ndarray:
    if not polys:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.zeros((len(polys), 4), dtype=np.float32)
    for i, p in enumerate(polys):
        x = p[:, 0]; y = p[:, 1]
        out[i] = [x.min(), y.min(), x.max(), y.max()]
    return out


def instance_map_to_masks(inst: np.ndarray) -> np.ndarray:
    ids = np.unique(inst)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros((0, *inst.shape), dtype=bool)
    return np.stack([(inst == i) for i in ids], axis=0)


# --------------------------- Pixel-level metrics -----------------------------

def pixel_confusion(gt_bin: np.ndarray, pr_bin: np.ndarray) -> tuple[int, int, int, int]:
    gt = gt_bin.astype(bool); pr = pr_bin.astype(bool)
    tp = int(np.logical_and(gt, pr).sum())
    fp = int(np.logical_and(~gt, pr).sum())
    fn = int(np.logical_and(gt, ~pr).sum())
    tn = int(np.logical_and(~gt, ~pr).sum())
    return tp, fp, fn, tn


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def rand_error(gt_inst: np.ndarray, pr_inst: np.ndarray) -> float:
    """Adjusted Rand-style error: 1 - (a + d) / C(n,2) where a = pairs together
    in both, d = pairs separated in both. Computed via contingency table.
    """
    gt = gt_inst.flatten().astype(np.int64)
    pr = pr_inst.flatten().astype(np.int64)
    n = gt.size
    if n < 2:
        return 0.0

    pair_id = gt * (pr.max() + 1) + pr
    _, counts = np.unique(pair_id, return_counts=True)
    sum_nij2 = float((counts * (counts - 1) // 2).sum())

    _, gt_counts = np.unique(gt, return_counts=True)
    _, pr_counts = np.unique(pr, return_counts=True)
    sum_a2 = float((gt_counts * (gt_counts - 1) // 2).sum())
    sum_b2 = float((pr_counts * (pr_counts - 1) // 2).sum())
    total_pairs = float(n * (n - 1) // 2)

    a = sum_nij2
    b = sum_a2 - sum_nij2
    c = sum_b2 - sum_nij2
    d = total_pairs - a - b - c
    rand_index = (a + d) / total_pairs if total_pairs > 0 else 1.0
    return 1.0 - rand_index


# ---------------------- Detection-level (mAP / AP) ---------------------------

def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0).astype(np.float32)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point COCO-style AP."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    rec_thr = np.linspace(0, 1, 101)
    inds = np.searchsorted(mrec, rec_thr, side="left")
    inds = np.clip(inds, 0, mpre.size - 1)
    return float(mpre[inds].mean())


def detection_pr_at_iou(matches: list[tuple[float, int, int]], n_gt: int,
                        iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """matches: list of (score, iou_with_best_unmatched_gt, class) sorted by score desc.

    Here we precompute per-image matching at the threshold and pass tp/fp arrays in.
    Returning (precision_curve, recall_curve).
    """
    raise NotImplementedError  # not used directly, see compute_map


def compute_map(per_image: list[dict], iou_thrs: np.ndarray) -> dict:
    """Compute mAP across IoU thresholds and per-class AP.

    per_image = [{
        'gt_boxes': (M,4), 'gt_classes': (M,),
        'pred_boxes': (N,4), 'pred_classes': (N,), 'pred_scores': (N,)
    }, ...]
    """
    classes = sorted(set(
        c for d in per_image
        for c in list(d["gt_classes"].tolist()) + list(d["pred_classes"].tolist())
    ))
    if not classes:
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "per_class": {}}

    per_class: dict[int, dict] = {}

    for cls in classes:
        ap_per_thr: list[float] = []
        for thr in iou_thrs:
            scores_all: list[float] = []
            tp_all: list[int] = []
            fp_all: list[int] = []
            n_gt_total = 0

            for d in per_image:
                gt_mask = d["gt_classes"] == cls
                pr_mask = d["pred_classes"] == cls
                gt_boxes = d["gt_boxes"][gt_mask]
                pr_boxes = d["pred_boxes"][pr_mask]
                pr_scores = d["pred_scores"][pr_mask]

                n_gt_total += len(gt_boxes)
                if len(pr_boxes) == 0:
                    continue

                order = np.argsort(-pr_scores)
                pr_boxes = pr_boxes[order]
                pr_scores = pr_scores[order]

                if len(gt_boxes) == 0:
                    scores_all.extend(pr_scores.tolist())
                    tp_all.extend([0] * len(pr_boxes))
                    fp_all.extend([1] * len(pr_boxes))
                    continue

                ious = box_iou_matrix(pr_boxes, gt_boxes)
                gt_used = np.zeros(len(gt_boxes), dtype=bool)
                for i in range(len(pr_boxes)):
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= thr and not gt_used[j]:
                        tp_all.append(1); fp_all.append(0)
                        gt_used[j] = True
                    else:
                        tp_all.append(0); fp_all.append(1)
                    scores_all.append(float(pr_scores[i]))

            if n_gt_total == 0:
                ap_per_thr.append(float("nan"))
                continue
            if not scores_all:
                ap_per_thr.append(0.0)
                continue

            order = np.argsort(-np.array(scores_all))
            tp_arr = np.array(tp_all)[order]
            fp_arr = np.array(fp_all)[order]
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            recall = tp_cum / n_gt_total
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            ap_per_thr.append(compute_ap(recall, precision))

        valid = [v for v in ap_per_thr if not np.isnan(v)]
        per_class[cls] = {
            "AP@0.5": ap_per_thr[0] if iou_thrs[0] == 0.5 and not np.isnan(ap_per_thr[0]) else float("nan"),
            "AP@0.5:0.95": float(np.mean(valid)) if valid else float("nan"),
            "AP_per_iou": ap_per_thr,
        }

    valid_50 = [v["AP@0.5"] for v in per_class.values() if not np.isnan(v["AP@0.5"])]
    valid_avg = [v["AP@0.5:0.95"] for v in per_class.values() if not np.isnan(v["AP@0.5:0.95"])]
    return {
        "mAP@0.5": float(np.mean(valid_50)) if valid_50 else 0.0,
        "mAP@0.5:0.95": float(np.mean(valid_avg)) if valid_avg else 0.0,
        "per_class": per_class,
        "iou_thrs": iou_thrs.tolist(),
    }


def detection_pr_iou_summary(per_image: list[dict], iou_thr: float = 0.5) -> dict:
    """Aggregate Precision/Recall/IoU across all detections at a given IoU threshold."""
    tp = fp = fn = 0
    iou_sum = 0.0
    iou_n = 0
    for d in per_image:
        gt = d["gt_boxes"]; pr = d["pred_boxes"]
        scores = d["pred_scores"]
        if len(pr) and len(gt):
            order = np.argsort(-scores)
            pr_sorted = pr[order]
            ious = box_iou_matrix(pr_sorted, gt)
            gt_used = np.zeros(len(gt), dtype=bool)
            for i in range(len(pr_sorted)):
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= iou_thr and not gt_used[j]:
                    tp += 1
                    gt_used[j] = True
                    iou_sum += float(ious[i, j]); iou_n += 1
                else:
                    fp += 1
            fn += int((~gt_used).sum())
        else:
            fp += len(pr)
            fn += len(gt)
    return {
        "det_TP": tp, "det_FP": fp, "det_FN": fn,
        "det_Precision": safe_div(tp, tp + fp),
        "det_Recall": safe_div(tp, tp + fn),
        "det_meanIoU(matched)": (iou_sum / iou_n) if iou_n else 0.0,
    }


# ---------------------------- Driver -----------------------------------------

def evaluate(gt_images: Path, gt_labels: Path, pred_labels: Path,
             out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_image_det: list[dict] = []
    agg = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "rand_err_sum": 0.0, "n": 0}

    image_paths = [p for p in sorted(gt_images.iterdir())
                   if p.suffix.lower() in IMG_EXTS]
    if not image_paths:
        raise FileNotFoundError(f"No images in {gt_images}")

    for img_path in image_paths:
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_inst, gt_polys, gt_cls, _ = load_yolo_seg_label(
            gt_labels / f"{stem}.txt", w, h)
        pr_inst, pr_polys, pr_cls, pr_scores = load_yolo_seg_label(
            pred_labels / f"{stem}.txt", w, h)

        gt_bin = gt_inst > 0
        pr_bin = pr_inst > 0
        tp, fp, fn, tn = pixel_confusion(gt_bin, pr_bin)
        acc = safe_div(tp + tn, tp + fp + fn + tn)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        iou = safe_div(tp, tp + fp + fn)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        pix_err = (1.0 - acc) * 100.0
        re_err = rand_error(gt_inst, pr_inst)

        rows.append({
            "image": stem,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": prec, "Recall": rec, "Accuracy": acc,
            "IoU": iou, "F1": f1,
            "PixelError(%)": pix_err, "RandError": re_err,
            "n_gt": len(gt_polys), "n_pred": len(pr_polys),
        })

        agg["TP"] += tp; agg["FP"] += fp; agg["FN"] += fn; agg["TN"] += tn
        agg["rand_err_sum"] += re_err; agg["n"] += 1

        per_image_det.append({
            "gt_boxes": polygons_to_bboxes(gt_polys),
            "gt_classes": gt_cls,
            "pred_boxes": polygons_to_bboxes(pr_polys),
            "pred_classes": pr_cls,
            "pred_scores": pr_scores if pr_scores.size else np.ones(len(pr_polys)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_per_image.csv", index=False)

    TP, FP, FN, TN = agg["TP"], agg["FP"], agg["FN"], agg["TN"]
    pixel_summary = {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Precision": safe_div(TP, TP + FP),
        "Recall": safe_div(TP, TP + FN),
        "Accuracy": safe_div(TP + TN, TP + FP + FN + TN),
        "IoU": safe_div(TP, TP + FP + FN),
        "F1": safe_div(2 * TP, 2 * TP + FP + FN),
        "PixelError(%)": (1.0 - safe_div(TP + TN, TP + FP + FN + TN)) * 100.0,
        "RandError(mean)": agg["rand_err_sum"] / max(agg["n"], 1),
    }

    iou_thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)
    map_summary = compute_map(per_image_det, iou_thrs)
    det_pr = detection_pr_iou_summary(per_image_det, iou_thr=0.5)

    summary = {
        "pixel_level": pixel_summary,
        "detection_level": {**det_pr, **map_summary},
        "n_images": len(image_paths),
    }

    with (out_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    return summary


def print_summary(s: dict):
    p = s["pixel_level"]; d = s["detection_level"]
    print("\n========== PIXEL-LEVEL ==========")
    print(f"  TP={p['TP']}  FP={p['FP']}  FN={p['FN']}  TN={p['TN']}")
    print(f"  Precision    : {p['Precision']:.4f}")
    print(f"  Recall       : {p['Recall']:.4f}")
    print(f"  Accuracy     : {p['Accuracy']:.4f}")
    print(f"  IoU          : {p['IoU']:.4f}")
    print(f"  F1 Score     : {p['F1']:.4f}")
    print(f"  Pixel Error  : {p['PixelError(%)']:.4f} %")
    print(f"  Rand Error   : {p['RandError(mean)']:.4f}")
    print("\n========== DETECTION-LEVEL ==========")
    print(f"  TP={d['det_TP']}  FP={d['det_FP']}  FN={d['det_FN']}")
    print(f"  Precision    : {d['det_Precision']:.4f}")
    print(f"  Recall       : {d['det_Recall']:.4f}")
    print(f"  IoU(matched) : {d['det_meanIoU(matched)']:.4f}")
    print(f"  mAP@0.5      : {d['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95 : {d['mAP@0.5:0.95']:.4f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-images", type=Path, required=True,
                    help="Folder of ground-truth images")
    ap.add_argument("--gt-labels", type=Path, required=True,
                    help="Folder of YOLO-format ground-truth label .txt files")
    ap.add_argument("--pred-labels", type=Path, required=True,
                    help="Folder of predicted YOLO-format label .txt files (from predict.py)")
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics"))
    args = ap.parse_args()
    evaluate(args.gt_images, args.gt_labels, args.pred_labels, args.out)


if __name__ == "__main__":
    main()

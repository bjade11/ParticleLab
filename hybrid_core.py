"""Hybrid Fusion — U-Net 확률맵 + YOLO 인스턴스 분리선 결합.

알고리즘 (사용자 spec):
    1. M_75 = (P_unet >= 0.75)                            # 보수적 입자 마스크 (메인)
    2. 배경 픽셀 b 마다
         d_1(b) = M_75 의 가장 가까운 component 까지 거리
         d_2(b) = 두 번째 가까운 component 까지 거리
    3. τ* = Otsu({d_1+d_2})  →  M_close = (d_1+d_2 < τ*)
    4. YOLO instance 경계 B (boundary_thickness px)
    5. Fusion:
         분리선  = M_close ∩ B
         입자    = M_75 ∪ (M_close ∩ (P_unet >= 0.30) ∩ ¬B)
         semantic = 입자 - 분리선
    6. instances = connected_components(semantic)

순수 numpy/scipy/cv2. torch 비의존.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class HybridParams:
    high_thresh: float = 0.75       # U-Net "확실한 입자" 임계
    low_thresh: float = 0.40        # U-Net "가까운 픽셀에서만 살릴" 보강 임계 (운용값)
    boundary_thickness: int = 3     # YOLO instance 경계 두께 (px). 1px 는 큰 이미지에서 시각적으로 누락된 것처럼 보임
    min_anchor_area: int = 10       # M_75 의 noise component 최소 크기
    # d_1 ≈ d_2 조건: |d_1 - d_2| / (d_1+d_2) <= 1 - balance_floor 인 픽셀만 "사이"로 인정.
    # 한 입자에만 붙은 외곽 픽셀 (d_1≈0) 을 M_close 에서 제외하기 위함.
    # 0.0 이면 spec 그대로 (d_1+d_2 < τ 만 사용), 0.5 면 d_1, d_2 가 비등한 곳만.
    balance_floor: float = 0.25
    # Otsu τ* 배수. M_close = d_+ < τ* × tau_scale.
    # anchor(P≥0.75)가 입자 코어만 잡아 d_+ 가 실제 틈보다 부풀려질 때,
    # 1.0 보다 키우면 좁은 틈 검출 범위가 넓어진다.
    tau_scale: float = 1.0
    # YOLO 후처리
    yolo_round_strength: int = 3    # 0=끄기, 1~10 권장. Gaussian blur σ → 라운드/부드럽게
    yolo_remove_overlaps: bool = True  # 두 instance 가 겹친 픽셀은 배경(분리선)으로 처리
    # 포위(containment) 제거 — 작은 instance 가 더 큰 instance 의 외곽선 안에 갇히면 삭제.
    # 큰 입자 내부에 잘못 검출된 작은 detection (또는 원 안의 원) 을 제거한다.
    yolo_remove_enclosed: bool = True
    enclosed_ratio: float = 0.9     # 작은 instance 픽셀의 이 비율 이상이 큰 것 안에 들어가면 포위로 판정


@dataclass
class HybridResult:
    semantic: np.ndarray            # (H, W) bool
    instances: np.ndarray           # (H, W) int32, 0=bg
    n_instances: int
    M_high: np.ndarray              # (H, W) bool — M_75
    M_close: np.ndarray             # (H, W) bool — "가까운 픽셀"
    boundary: np.ndarray            # (H, W) bool — YOLO 분리선 (두꺼움)
    distance_sum: np.ndarray        # (H, W) float — d1+d2 (np.inf 가능)
    d1: np.ndarray                  # (H, W) float — 가장 가까운 component 거리
    d2: np.ndarray                  # (H, W) float — 두 번째 가까운 거리
    tau: float                      # Otsu 임계값
    K_unet: int                     # M_75 의 component 수
    n_yolo: int                     # 입력 YOLO instance 수
    yolo_processed: List[np.ndarray]  # 라운드/overlap 후처리 적용된 YOLO instance 들
    yolo_overlap: np.ndarray        # (H, W) bool — 두 YOLO instance 가 겹친 영역


# ───────────────────────────── 내부 helpers ─────────────────────────────


def _top2_distances(labels: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """K 개 component 에 대한 픽셀별 (d_1, d_2).

    누적 업데이트로 메모리 O(H*W), 시간 O(K * H * W).
    K = 1 이면 d_2 = inf.

    같은 component 안의 픽셀은 d_1 = 0 (distance_transform_edt(labels != i)).
    """
    H, W = labels.shape
    d1 = np.full((H, W), np.inf, dtype=np.float32)
    d2 = np.full((H, W), np.inf, dtype=np.float32)

    for i in range(1, K + 1):
        # labels != i 인 픽셀에서 → i 까지의 유클리드 거리
        # i 안의 픽셀은 0
        dist_i = ndi.distance_transform_edt(labels != i).astype(np.float32)

        closer_than_d1 = dist_i < d1
        # d1 갱신 — 기존 d1 은 d2 자리로 밀려남
        d2 = np.where(closer_than_d1, d1, d2)
        d1_new = np.where(closer_than_d1, dist_i, d1)
        # d2 갱신 (d1 갱신 안 된 픽셀 중 d2 보다 작으면)
        not_d1_update = ~closer_than_d1
        closer_than_d2 = not_d1_update & (dist_i < d2)
        d2 = np.where(closer_than_d2, dist_i, d2)

        d1 = d1_new

    return d1, d2


def _otsu_float(values: np.ndarray) -> float:
    """float 배열에서 Otsu threshold. inf/NaN 은 제거."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return float(vmin)
    # uint8 로 quantize 후 OpenCV Otsu — 256-bin
    norm = (finite - vmin) / (vmax - vmin) * 255.0
    u8 = norm.astype(np.uint8)
    thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr_u8 / 255.0 * (vmax - vmin) + vmin)


def round_smooth_mask(mask: np.ndarray, strength: int) -> np.ndarray:
    """이진 마스크를 라운드/부드럽게. Gaussian blur + threshold.

    strength=0: 끄기 (그대로 반환).
    strength=1~10: σ ~= strength, kernel ~= 6σ+1. 클수록 더 동그래짐.
    """
    if strength <= 0:
        return mask.astype(bool)
    sigma = float(strength)
    k = max(3, 2 * int(round(3 * sigma)) + 1)  # 6σ 룰
    u8 = (mask.astype(np.uint8) * 255)
    blurred = cv2.GaussianBlur(u8, (k, k), sigma)
    return blurred > 127


def remove_instance_overlaps(
    instances: List[np.ndarray],
    redundant_novelty: float = 0.5,
) -> tuple[List[np.ndarray], np.ndarray]:
    """중복 detection 제거 후, 진짜 겹친 픽셀만 배경 처리.

    YOLO 는 NMS 이후에도 한 입자를 거의 동일한 두 instance 로 중복 검출하는
    경우가 있다. 이를 단순히 "겹침" 으로 처리하면 입자 전체가 overlap 으로
    분류되어 통째로 배경이 되고 입자가 누락된다. 따라서 2 단계로 처리한다.

    1단계 — 중복 제거: 면적 큰 instance 부터 greedy 하게 채우면서, 새로
        기여하는 픽셀 비율(novelty)이 redundant_novelty 미만인 instance 는
        "이미 다른 instance 가 절반 이상 차지" = 중복 검출로 보고 버린다.
        (디폴트 0.5 — SEM 입자는 물리적으로 겹치지 않으므로, 50% 이상
         겹치는 instance 쌍은 거의 확실히 동일 입자의 중복 검출이다.)
    2단계 — 진짜 겹침: 남은 instance 들 중 2 개 이상이 겹친 픽셀만 제거.

    반환: (cleaned_list, overlap_mask)
    """
    valid = [m.astype(bool) for m in instances if m is not None]
    if len(valid) < 2:
        empty = (np.zeros_like(valid[0], dtype=bool)
                 if valid else np.zeros((1, 1), dtype=bool))
        return valid, empty

    shape = valid[0].shape
    areas = np.array([int(m.sum()) for m in valid])

    # 1단계 — 중복 detection 제거 (면적 큰 것부터 greedy)
    order = np.argsort(-areas)
    occupied = np.zeros(shape, dtype=bool)
    kept: List[np.ndarray] = []
    for idx in order:
        m = valid[idx]
        if areas[idx] == 0:
            continue
        novel = int((m & ~occupied).sum())
        if novel / areas[idx] < redundant_novelty:
            # 이미 다른 instance 가 대부분 차지 → 중복 검출, 버림
            continue
        kept.append(m)
        occupied |= m

    if len(kept) < 2:
        return kept, np.zeros(shape, dtype=bool)

    # 2단계 — 남은 instance 들 중 진짜 겹친 픽셀 제거
    count = np.zeros(shape, dtype=np.int32)
    for m in kept:
        count += m.astype(np.int32)
    overlap = count >= 2
    cleaned = [m & (~overlap) for m in kept]
    return cleaned, overlap


def _filled_outline(mask: np.ndarray) -> np.ndarray:
    """instance 의 외곽 contour 를 1px 선으로 잡고 그 안을 채운 영역.

    오목한 입자라도 외곽 폐곡선 내부 전체를 채운다 (binary_fill_holes 와 달리
    만(bay)·구멍까지 포함). 사용자 표현의 "가상 1px 선으로 둘러싼 영역".
    """
    m = mask.astype(np.uint8)
    if int(m.sum()) == 0:
        return mask.astype(bool)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def remove_enclosed_instances(
    instances: List[np.ndarray],
    enclosed_ratio: float = 0.9,
) -> tuple[List[np.ndarray], List[int]]:
    """더 큰 instance 의 외곽선 안에 포위된 작은 instance 를 제거.

    각 instance 의 외곽 contour 를 채운 영역(filled outline)을 "가상 1px 경계로
    둘러싼 영역" 으로 본다. 작은 instance 픽셀의 enclosed_ratio 이상이 더 큰
    instance 의 filled outline 안에 들어가면, 그 작은 것을 큰 입자 내부에 잘못
    검출된 detection 으로 보고 제거한다.

    overlap 제거(면적 겹침 기준)와 달리, 작은 입자가 큰 입자의 마스크와 직접
    겹치지 않아도 (예: 큰 입자 윤곽 안의 빈 공간) 윤곽 기준으로 포위를 판정한다.

    반환: (kept_list, removed_indices) — removed_indices 는 입력 valid 순서 기준.
    """
    valid = [m.astype(bool) for m in instances if m is not None and m.any()]
    n = len(valid)
    if n < 2:
        return valid, []

    areas = [int(m.sum()) for m in valid]
    filled = [_filled_outline(m) for m in valid]
    removed: set[int] = set()

    # 작은 것부터 — 더 큰 후보의 외곽선 안에 포위됐는지 검사
    order = sorted(range(n), key=lambda i: areas[i])
    for i in order:
        if i in removed:
            continue
        mi, ai = valid[i], areas[i]
        if ai == 0:
            continue
        for j in range(n):
            if j == i or j in removed or areas[j] <= ai:
                continue  # 포위하려면 j 가 더 커야 함
            inside = int((mi & filled[j]).sum())
            if inside / ai >= enclosed_ratio:
                removed.add(i)
                break

    kept = [valid[i] for i in range(n) if i not in removed]
    return kept, sorted(removed)


def _instance_boundaries(
    instances: List[np.ndarray],
    h: int,
    w: int,
    thickness: int,
) -> np.ndarray:
    """YOLO instance mask 들의 외곽선을 합쳐 dilate 한 결과."""
    B = np.zeros((h, w), dtype=np.uint8)
    if not instances:
        return B.astype(bool)
    for m in instances:
        if m.shape != (h, w):
            m_resized = cv2.resize(
                m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            m_resized = m.astype(bool)
        # 외곽: mask XOR erode(mask)
        m_u8 = m_resized.astype(np.uint8)
        mask_count = int(m_u8.sum())
        if mask_count == 0:
            continue
        eroded = cv2.erode(m_u8, np.ones((3, 3), np.uint8), iterations=1)
        edge = (m_u8 - eroded).astype(bool)
        edge_count = int(edge.sum())
        # 작은 입자는 erosion 후 내부가 거의 남지 않아 edge 가 입자의 대부분을
        # 차지한다. 그러면 그 입자가 통째로 boundary → separator → 누락된다.
        # 외곽선이 입자의 60% 이상이면 외곽선/내부를 구분할 수 없는 작은
        # 입자이므로 boundary 기여에서 제외한다 (eroded 가 비는 경우 ratio=1.0).
        if edge_count / mask_count > 0.6:
            continue
        B |= edge.astype(np.uint8)
    if thickness > 1:
        k = thickness  # 두께
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        B = cv2.dilate(B, kernel, iterations=1)
    return B.astype(bool)


def _filter_small_components(
    mask: np.ndarray, min_area: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """connected component 중 min_area 미만 제거. (filtered_mask, labels, K)."""
    labels, K = ndi.label(mask)
    if K == 0:
        return mask, labels, 0
    if min_area <= 1:
        return mask, labels, int(K)
    # 각 component 의 픽셀 수
    sizes = np.bincount(labels.ravel(), minlength=K + 1)
    keep = sizes >= min_area
    keep[0] = False  # 배경 라벨 0 은 제외
    if keep.sum() == K:
        return mask, labels, int(K)
    # 작은 component 를 0 으로 (재라벨)
    relabel = np.zeros(K + 1, dtype=labels.dtype)
    new_id = 0
    for i in range(1, K + 1):
        if keep[i]:
            new_id += 1
            relabel[i] = new_id
    new_labels = relabel[labels]
    new_mask = new_labels > 0
    return new_mask, new_labels, int(new_id)


# ───────────────────────────── public API ─────────────────────────────


def hybrid_fuse(
    unet_prob: np.ndarray,
    yolo_instances: List[np.ndarray],
    params: Optional[HybridParams] = None,
) -> HybridResult:
    """U-Net 확률맵 + YOLO 인스턴스 마스크들 → 융합 결과.

    인자:
        unet_prob: (H, W) float32, 값 범위 [0, 1]
        yolo_instances: 각 (H, W) bool, YOLO 가 내놓은 instance mask 들
        params: HybridParams

    반환:
        HybridResult (semantic + instances + 진단 데이터)
    """
    if params is None:
        params = HybridParams()

    h, w = unet_prob.shape
    P = unet_prob.astype(np.float32)
    P_low = P >= float(params.low_thresh)
    P_high = P >= float(params.high_thresh)

    # YOLO 인스턴스 후처리 — 라운드 + overlap 제거
    yolo_processed: List[np.ndarray] = []
    for m in yolo_instances:
        if m.shape != (h, w):
            m = cv2.resize(
                m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        if params.yolo_round_strength > 0:
            m = round_smooth_mask(m, int(params.yolo_round_strength))
        yolo_processed.append(m.astype(bool))
    # 포위 제거 — overlap 제거보다 먼저 (큰 입자 안에 갇힌 detection 을 먼저 솎아낸다)
    if params.yolo_remove_enclosed:
        yolo_processed, _ = remove_enclosed_instances(
            yolo_processed, float(params.enclosed_ratio)
        )
    if params.yolo_remove_overlaps:
        yolo_processed, yolo_overlap = remove_instance_overlaps(yolo_processed)
    else:
        yolo_overlap = np.zeros((h, w), dtype=bool)

    # 1) M_75 (anchor) — noise component 제거
    M_high_filt, labels, K = _filter_small_components(
        P_high, int(params.min_anchor_area)
    )

    # K < 2: d_2 정의 안 되므로 fusion 단계 스킵 (M_high 만 사용)
    if K < 2:
        # M_high(U-Net 확신 입자)는 overlap 이어도 유지 (메인 경로와 동일 정책)
        semantic = M_high_filt
        inst_lbl, n_inst = ndi.label(semantic)
        return HybridResult(
            semantic=semantic,
            instances=inst_lbl.astype(np.int32),
            n_instances=int(n_inst),
            M_high=M_high_filt,
            M_close=np.zeros((h, w), dtype=bool),
            boundary=_instance_boundaries(
                yolo_processed, h, w, params.boundary_thickness
            ) | yolo_overlap,
            distance_sum=np.full((h, w), np.inf, dtype=np.float32),
            d1=np.full((h, w), np.inf, dtype=np.float32),
            d2=np.full((h, w), np.inf, dtype=np.float32),
            tau=0.0,
            K_unet=int(K),
            n_yolo=len(yolo_instances),
            yolo_processed=yolo_processed,
            yolo_overlap=yolo_overlap,
        )

    # 2) 픽셀별 d_1 + d_2
    d1, d2 = _top2_distances(labels, K)
    d_sum = d1 + d2  # M_high 안 픽셀은 0, 배경은 양수

    # 3) Otsu — *배경 픽셀의* d_sum 분포 기준 (입자 내부 = 0 은 분포에서 제외)
    bg = ~M_high_filt
    tau = _otsu_float(d_sum[bg])
    # 좁은 틈 판정 임계 — Otsu τ* 에 tau_scale 배수 적용 (검출 범위 조절).
    tau_eff = tau * float(params.tau_scale)
    # d_1 ≈ d_2 balance 조건: 두 입자에 비등하게 가까운 픽셀만 "진짜 사이".
    # |d_1 - d_2| / (d_1+d_2 + eps) <= 1 - balance_floor.
    # 즉 min(d_1,d_2) / max(d_1,d_2) >= balance_floor 와 등가 (대략).
    eps = 1e-6
    bal = 1.0 - np.abs(d1 - d2) / (d_sum + eps)  # 0(한쪽 치우침) ~ 1(완전 중간)
    M_close = (d_sum < tau_eff) & bg & (bal >= float(params.balance_floor))

    # 4) YOLO 경계 — 후처리된 instance 의 외곽 + (옵션) 겹친 영역도 분리선에 포함
    B = _instance_boundaries(yolo_processed, h, w, params.boundary_thickness)
    B = B | yolo_overlap

    # 5) Fusion
    separator = M_close & B                                # 분리선 (좁은 틈 ∩ YOLO 경계)
    boost = M_close & P_low & (~B)                         # 분리선 아닌 가까운 픽셀, U-Net 30%+
    particle = M_high_filt | boost
    # YOLO 겹친 영역은 배경 처리하되, U-Net 이 확신(P≥high_thresh)한 곳은 예외로 살린다.
    # (겹침이 U-Net 입자 판단을 덮어쓰던 문제 수정 — 두 YOLO 박스가 겹쳐도 U-Net 이
    #  확실한 입자라고 본 자리는 입자로 유지.)
    semantic = particle & (~separator) & (~(yolo_overlap & ~M_high_filt))

    # 6) 최종 instance 라벨 — 분리선으로 쪼개진 component 재라벨.
    #    boost 가 분리선 양옆으로 살아남아 생긴 작은 조각도 같은 min_anchor_area 로 정리.
    semantic, inst_lbl, n_inst = _filter_small_components(
        semantic, int(params.min_anchor_area)
    )

    return HybridResult(
        semantic=semantic,
        instances=inst_lbl.astype(np.int32),
        n_instances=int(n_inst),
        M_high=M_high_filt,
        M_close=M_close,
        boundary=B,
        distance_sum=d_sum,
        d1=d1,
        d2=d2,
        tau=float(tau_eff),
        K_unet=int(K),
        n_yolo=len(yolo_instances),
        yolo_processed=yolo_processed,
        yolo_overlap=yolo_overlap,
    )


def colorize_instances(instances: np.ndarray, seed: int = 7) -> np.ndarray:
    """instance label (H,W) int → BGR 컬러 이미지. 입자별 다른 색."""
    h, w = instances.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    n = int(instances.max())
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    for i in range(1, n + 1):
        hue = int(rng.integers(0, 180))
        sat = int(rng.integers(150, 256))
        val = int(rng.integers(190, 256))
        hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        out[instances == i] = bgr
    return out


def overlay_on_image(
    image_gray: np.ndarray,
    instances: np.ndarray,
    alpha: float = 0.45,
    seed: int = 7,
) -> np.ndarray:
    """원본 위에 instance 컬러 overlay. image_gray (H,W) uint8."""
    if image_gray.ndim == 2:
        base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        base = image_gray
    colored = colorize_instances(instances, seed=seed)
    mask = instances > 0
    out = base.copy()
    out[mask] = (alpha * colored[mask] + (1 - alpha) * base[mask]).astype(np.uint8)
    return out

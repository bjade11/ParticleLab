"""U-Net / U-Net++ 학습 + 검증 + 추론.

논문 (Bals & Epple 2023) 의 weighted BCE 식 (1) 을 그대로 구현.
weight map 은 weight_maps.compute_weight_map 으로 생성하며 SE/STEM 모드 또는
사용자 정의를 지원한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from unet_models import build_model, pick_device
from weight_maps import WeightMapConfig, compute_weight_map


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SemMaskDataset(Dataset):
    """SEM 이미지 + binary mask + (사전계산된) weight map 페어.

    인덱스마다 패치를 무작위로 잘라서 반환 (논문 방식). 검증 시에는
    augment=False, full_image=True 로 호출하면 전체 이미지를 정해진 size 로
    리사이즈해 반환한다.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, str]],  # (stem, image_path, mask_path)
        size: int = 256,
        augment: bool = True,
        wmap_cfg: WeightMapConfig = WeightMapConfig(),
        precompute_weight: bool = True,
    ) -> None:
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.wmap_cfg = wmap_cfg

        # weight map 계산은 비싸므로 데이터셋 생성 시 한 번만 해 둔다.
        # mode='none' 이면 1.0 텐서로 채움.
        self._wmap_cache: List[np.ndarray] = []
        if precompute_weight:
            for stem, ip, mp in pairs:
                img = self._load_gray(ip)
                msk = self._load_mask(mp)
                w = compute_weight_map(msk, img, wmap_cfg) \
                    if wmap_cfg.mode != "none" else None
                self._wmap_cache.append(w)
        else:
            self._wmap_cache = [None] * len(pairs)

    @staticmethod
    def _load_gray(path: str) -> np.ndarray:
        img = np.asarray(Image.open(path).convert("L"))
        return img

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        m = np.asarray(Image.open(path).convert("L"))
        return m > 127

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        stem, ip, mp = self.pairs[idx]
        img = self._load_gray(ip).astype(np.float32) / 255.0
        msk = self._load_mask(mp).astype(np.float32)
        wmap = self._wmap_cache[idx]
        wmap = wmap.astype(np.float32) if wmap is not None else \
               np.ones_like(msk, dtype=np.float32)

        # 같은 size 로 리사이즈 (image: bilinear, mask/wmap: nearest)
        s = self.size
        img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize(
            (s, s), Image.BILINEAR)
        msk_pil = Image.fromarray((msk * 255).astype(np.uint8)).resize(
            (s, s), Image.NEAREST)
        wmap_pil = Image.fromarray(wmap).resize((s, s), Image.NEAREST)
        img = np.asarray(img_pil).astype(np.float32) / 255.0
        msk = (np.asarray(msk_pil) > 127).astype(np.float32)
        wmap = np.asarray(wmap_pil).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                msk = np.fliplr(msk).copy()
                wmap = np.fliplr(wmap).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
                msk = np.flipud(msk).copy()
                wmap = np.flipud(wmap).copy()

        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(msk).unsqueeze(0),
                torch.from_numpy(wmap).unsqueeze(0),
                stem)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted Binary Cross-Entropy (논문 식 2).
    L = - mean( w * [y*log(p) + (1-y)*log(1-p)] )
    """
    eps = 1e-7
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    bce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
    return (weight * bce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    method: str = "U-Net++"
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-3
    image_size: int = 256
    val_fraction: float = 0.2
    augment: bool = True
    seed: int = 1234
    use_dice: bool = True       # BCE + Dice 혼합
    dice_weight: float = 0.5
    wmap: WeightMapConfig = field(default_factory=WeightMapConfig)


@dataclass
class TrainResult:
    history: List[Dict[str, float]]
    best_val: float
    elapsed: float
    device: str
    state_dict: Dict[str, torch.Tensor]
    config: TrainConfig


def train_unet(
    pairs: List[Tuple[str, str, str]],
    config: TrainConfig,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
) -> TrainResult:
    """학습. progress_cb(epoch, total_epochs, train_loss, val_loss) 콜백 호출."""
    if not pairs:
        raise ValueError("No pairs provided")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    full = SemMaskDataset(pairs, size=config.image_size, augment=False,
                          wmap_cfg=config.wmap, precompute_weight=True)
    n_val = max(1, int(round(len(full) * config.val_fraction)))
    n_train = max(1, len(full) - n_val)
    if n_train + n_val > len(full):
        n_val = len(full) - n_train
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )
    # train 쪽에 augmentation 적용 (단, 같은 wmap cache 공유)
    train_set.dataset = SemMaskDataset(
        pairs, size=config.image_size, augment=config.augment,
        wmap_cfg=config.wmap, precompute_weight=False,
    )
    train_set.dataset._wmap_cache = full._wmap_cache  # 캐시 공유

    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)

    device = pick_device()
    model = build_model(config.method).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        agg = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "n": 0}
        for x, y, w, _stem in train_loader:
            x = x.to(device); y = y.to(device); w = w.to(device)
            logits = model(x)
            bce = weighted_bce_loss(logits, y, w)
            if config.use_dice:
                dl = dice_loss(logits, y)
                loss = (1.0 - config.dice_weight) * bce + config.dice_weight * dl
            else:
                dl = torch.tensor(0.0, device=device)
                loss = bce
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.size(0)
            agg["loss"] += float(loss.item()) * bs
            agg["bce"] += float(bce.item()) * bs
            agg["dice"] += float(dl.item()) * bs
            agg["n"] += bs

        train_loss = agg["loss"] / max(1, agg["n"])
        train_bce = agg["bce"] / max(1, agg["n"])
        train_dice = agg["dice"] / max(1, agg["n"])

        # validation
        model.eval()
        v = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "n": 0}
        with torch.no_grad():
            for x, y, w, _stem in val_loader:
                x = x.to(device); y = y.to(device); w = w.to(device)
                logits = model(x)
                bce = weighted_bce_loss(logits, y, w)
                dl = dice_loss(logits, y)
                if config.use_dice:
                    loss = (1.0 - config.dice_weight) * bce + config.dice_weight * dl
                else:
                    loss = bce
                bs = x.size(0)
                v["loss"] += float(loss.item()) * bs
                v["bce"] += float(bce.item()) * bs
                v["dice"] += float(dl.item()) * bs
                v["n"] += bs
        val_loss = v["loss"] / max(1, v["n"])
        val_bce = v["bce"] / max(1, v["n"])
        val_dice = v["dice"] / max(1, v["n"])

        history.append(dict(
            epoch=epoch,
            train_loss=train_loss, val_loss=val_loss,
            train_bce=train_bce, val_bce=val_bce,
            train_dice=train_dice, val_dice=val_dice,
        ))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if progress_cb:
            progress_cb(epoch, config.epochs, train_loss, val_loss)

    elapsed = time.time() - t0
    if not best_state:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
    return TrainResult(
        history=history, best_val=best_val, elapsed=elapsed,
        device=str(device), state_dict=best_state, config=config,
    )


def predict_one(
    model: nn.Module,
    image_uint8_gray: np.ndarray,
    image_size: int = 256,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """원본 이미지 1장 → (probability HxW float32 [0,1], binary HxW uint8 0/255)
    원본 해상도로 다시 업샘플."""
    if device is None:
        device = pick_device()
    h, w = image_uint8_gray.shape
    img_pil = Image.fromarray(image_uint8_gray).resize(
        (image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob_pil = Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR)
    prob_full = np.asarray(prob_pil).astype(np.float32) / 255.0
    binary = (prob_full > 0.5).astype(np.uint8) * 255
    return prob_full, binary


def predict_with_state(
    state_dict: Dict[str, torch.Tensor],
    method: str,
    image_uint8_gray: np.ndarray,
    image_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    device = pick_device()
    model = build_model(method).to(device)
    model.load_state_dict(state_dict)
    return predict_one(model, image_uint8_gray, image_size, device)


# ---------------------------------------------------------------------------
# Save / load — 학습된 모델을 한 파일(.pt)로 묶어 저장하고 다시 로드.
# state_dict 만 저장하면 method/image_size 같은 메타가 사라져서 추론 시 잘못된
# 모델로 로드될 수 있다. payload dict 로 모두 함께 저장.
# ---------------------------------------------------------------------------


def serialize_model(
    state_dict: Dict[str, torch.Tensor],
    method: str,
    image_size: int,
    settings: Optional[Dict] = None,
    history: Optional[List[Dict]] = None,
) -> bytes:
    """모델 가중치 + 메타 데이터 → bytes (다운로드 가능)."""
    import io as _io
    buf = _io.BytesIO()
    torch.save({
        "format_version": 1,
        "state_dict": state_dict,
        "method": method,
        "image_size": int(image_size),
        "settings": dict(settings or {}),
        "history": list(history or []),
    }, buf)
    return buf.getvalue()


def deserialize_model(data: bytes) -> Dict:
    """bytes → payload dict. weights_only=False (메타 dict 포함)."""
    import io as _io
    buf = _io.BytesIO(data)
    try:
        payload = torch.load(buf, map_location="cpu", weights_only=False)
    except TypeError:
        # 구버전 torch (weights_only 인자 없음)
        buf.seek(0)
        payload = torch.load(buf, map_location="cpu")
    if "state_dict" not in payload or "method" not in payload:
        raise ValueError("Invalid model file: missing state_dict / method")
    return payload

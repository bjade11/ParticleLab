"""U-Net 과 U-Net++ 아키텍처 (PyTorch).

논문 (Bals & Epple, RSC Adv. 2023, 13, 2795) 의 입력 크기 (512x512) 와 비슷한
범위에서 동작하도록 설계. 입력 H, W 는 16의 배수면 안전.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """표준 U-Net. encoder 4단, base channel 32 (가벼운 학습용)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        self.enc1 = _conv_block(in_ch, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.bottom = _conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottom(F.max_pool2d(e4, 2))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class UNetPP(nn.Module):
    """U-Net++ (Zhou et al. 2018) — nested skip pathways + deep supervision.
    논문 (Bals & Epple) 에서도 동일 architecture 사용."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.x00 = _conv_block(in_ch, chs[0])
        self.x10 = _conv_block(chs[0], chs[1])
        self.x20 = _conv_block(chs[1], chs[2])
        self.x30 = _conv_block(chs[2], chs[3])
        self.x40 = _conv_block(chs[3], chs[4])

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.x01 = _conv_block(chs[0] + chs[1], chs[0])
        self.x11 = _conv_block(chs[1] + chs[2], chs[1])
        self.x21 = _conv_block(chs[2] + chs[3], chs[2])
        self.x31 = _conv_block(chs[3] + chs[4], chs[3])

        self.x02 = _conv_block(chs[0] * 2 + chs[1], chs[0])
        self.x12 = _conv_block(chs[1] * 2 + chs[2], chs[1])
        self.x22 = _conv_block(chs[2] * 2 + chs[3], chs[2])

        self.x03 = _conv_block(chs[0] * 3 + chs[1], chs[0])
        self.x13 = _conv_block(chs[1] * 3 + chs[2], chs[1])

        self.x04 = _conv_block(chs[0] * 4 + chs[1], chs[0])

        self.heads = nn.ModuleList(
            [nn.Conv2d(chs[0], out_ch, 1) for _ in range(4)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(F.max_pool2d(x00, 2))
        x20 = self.x20(F.max_pool2d(x10, 2))
        x30 = self.x30(F.max_pool2d(x20, 2))
        x40 = self.x40(F.max_pool2d(x30, 2))

        x01 = self.x01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], 1))
        x31 = self.x31(torch.cat([x30, self.up(x40)], 1))

        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31)], 1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22)], 1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))

        # Deep supervision: 4개의 side output 평균
        outs = [self.heads[0](x01), self.heads[1](x02),
                self.heads[2](x03), self.heads[3](x04)]
        return sum(outs) / 4.0


def build_model(name: str) -> nn.Module:
    if name == "U-Net":
        return UNet()
    if name == "U-Net++":
        return UNetPP()
    raise ValueError(f"Unknown model: {name}")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

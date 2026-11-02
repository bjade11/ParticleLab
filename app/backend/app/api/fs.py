"""서버 폴더 탐색 — 배치 입력 폴더 선택용 (로컬 실행 전제)."""
from __future__ import annotations

import os
import string
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/fs", tags=["fs"])

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@router.get("/drives")
async def drives() -> list[str]:
    if sys.platform.startswith("win"):
        out = []
        for letter in string.ascii_uppercase:
            d = f"{letter}:\\"
            if os.path.exists(d):
                out.append(d)
        return out
    return ["/"]


@router.get("/list")
async def list_dir(path: str = Query("", description="폴더 절대경로. 비우면 홈")) -> dict:
    p = Path(path).expanduser() if path else Path.home()
    try:
        p = p.resolve()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"경로 해석 실패: {e}")
    if not p.exists():
        raise HTTPException(404, f"경로 없음: {p}")
    if not p.is_dir():
        p = p.parent

    dirs: list[dict] = []
    images: list[str] = []
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name.startswith((".", "$")):
                            continue
                        dirs.append({"name": e.name, "path": str(Path(p, e.name))})
                    elif e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS:
                        images.append(e.name)
                except PermissionError:
                    continue
    except PermissionError:
        raise HTTPException(403, f"접근 권한 없음: {p}")

    dirs.sort(key=lambda d: d["name"].lower())
    images.sort(key=str.lower)
    parent = str(p.parent) if p.parent != p else None
    return {
        "path": str(p),
        "parent": parent,
        "dirs": dirs,
        "n_images": len(images),
        "images": images[:50],
    }

"""ParticleLab — FastAPI 엔트리.

- 튜너 / 배치 분석 (watershed + AI Hybrid 엔진, particle_core 파이프라인)
- 기존 Particle Lab v2 AI 진단 엔드포인트 (/api/run/hybrid) 유지
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import logging
import math as _math
import sys
from pathlib import Path as _Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser
from starlette.responses import JSONResponse as _JR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# starlette multipart 한도 우회 (1000 파일 → 100,000)
MultiPartParser.max_files = 100_000
MultiPartParser.max_fields = 100_000
_orig_mpp_init = MultiPartParser.__init__
def _patched_mpp_init(self, *args, **kwargs):  # noqa: E302
    kwargs["max_files"] = 100_000
    kwargs["max_fields"] = 100_000
    kwargs.setdefault("max_part_size", 50 * 1024 * 1024 * 1024)
    try:
        return _orig_mpp_init(self, *args, **kwargs)
    except TypeError:
        kwargs.pop("max_part_size", None)
        return _orig_mpp_init(self, *args, **kwargs)
MultiPartParser.__init__ = _patched_mpp_init  # noqa: E305

# JSON NaN/Inf → null
_orig_render = _JR.render
def _sanitize(o):  # noqa: E302
    if isinstance(o, float):
        return None if (_math.isnan(o) or _math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o
def _patched_render(self, content):  # noqa: E302
    return _orig_render(self, _sanitize(content))
_JR.render = _patched_render  # noqa: E305

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .api.hybrid import router as hybrid_router  # noqa: E402
from .api.tuner import router as tuner_router  # noqa: E402
from .api.batch import router as batch_router  # noqa: E402
from .api.fs import router as fs_router  # noqa: E402

_BASIC_PT_DIR = _PROJECT_ROOT / "basic_pt"
_BASIC_PT_MAP = {
    "unet": ("U-net++", "U-net++.pt"),
    "yolo": ("yolo_11", "yolo_11.pt"),
}

app = FastAPI(
    title="ParticleLab",
    version="3.0.0",
    description="SEM 입자 분석 — watershed / AI Hybrid 엔진, 배치 분석 + 튜너",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
    except Exception:  # noqa: BLE001
        cuda, dev = False, "n/a"
    return {"ok": True, "name": "particle-lab", "version": app.version, "cuda": cuda, "device": dev}


@app.get("/api/basic_pt")
async def basic_pt_list():
    out: dict = {}
    for which, (fname, dl) in _BASIC_PT_MAP.items():
        p = _BASIC_PT_DIR / fname
        out[which] = {
            "available": p.exists(),
            "size_bytes": int(p.stat().st_size) if p.exists() else 0,
            "filename": dl,
        }
    return JSONResponse(out)


@app.get("/api/basic_pt/{which}")
async def basic_pt_get(which: str):
    info = _BASIC_PT_MAP.get(which)
    if info is None:
        raise HTTPException(404, f"unknown basic_pt key: {which}")
    fname, dl_name = info
    fp = _BASIC_PT_DIR / fname
    if not fp.exists():
        raise HTTPException(404, f"basic_pt/{fname} not found on server")
    return FileResponse(str(fp), media_type="application/octet-stream", filename=dl_name)


app.include_router(tuner_router)
app.include_router(batch_router)
app.include_router(fs_router)
app.include_router(hybrid_router)

# 데스크탑 런처용 정적 프런트 (app/frontend/out) — 있으면 루트에 마운트
_FRONT_OUT = _PROJECT_ROOT / "app" / "frontend" / "out"
if _FRONT_OUT.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONT_OUT), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {
            "name": "ParticleLab",
            "endpoints": [
                "/api/health", "GET /api/basic_pt",
                "POST /api/tuner/preprocess", "POST /api/tuner/segment",
                "POST /api/batch/jobs", "GET /api/batch/jobs/{id}",
                "GET /api/fs/list", "POST /api/run/hybrid", "/docs",
            ],
        }

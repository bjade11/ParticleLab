"""배치 분석 job API — 폴더(서버 경로 또는 업로드) → particle_core.pipeline.run_batch."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from particle_core.params import AnalysisParams  # noqa: E402
from particle_core.io_utils import list_images  # noqa: E402
from particle_core.pipeline import BatchInput, run_batch  # noqa: E402

from ..jobs import Job, registry  # noqa: E402
from .tuner import build_engine, parse_params  # noqa: E402

router = APIRouter(prefix="/api/batch", tags=["batch"])

RUNS_DIR = PROJECT_ROOT / "runs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-가-힣. ]+", "_", s).strip().strip(".")
    return s[:60] or "job"


def _make_job_fn(images: list[Path], output_dir: Path, params: AnalysisParams):
    def fn(job: Job):
        engine = build_engine(params)
        job.set_progress(1, f"엔진 준비: {params.engine}")
        engine.prepare()
        inp = BatchInput(images=images, output_dir=output_dir)
        result = run_batch(
            inp, params, engine,
            progress_cb=lambda pct, msg: job.set_progress(pct, msg),
            cancel_check=job.cancel_requested,
        )
        return result
    return fn


@router.post("/jobs")
async def create_job(
    params: str = Form("{}"),
    input_mode: str = Form("server_path"),
    input_path: Optional[str] = Form(None),
    output_dir: Optional[str] = Form(None),
    job_name: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    p = parse_params(params)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = _safe_name(job_name) if job_name else f"{stamp}_{p.engine}"

    out = Path(output_dir).expanduser() if output_dir and output_dir.strip() else RUNS_DIR / f"{stamp}_{_safe_name(name)}"
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"출력 폴더 생성 실패: {out} ({e})")

    images: list[Path]
    if input_mode == "upload":
        if not files:
            raise HTTPException(400, "업로드 모드인데 파일이 없습니다.")
        in_dir = out / "input"
        in_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for f in files:
            fname = Path(f.filename or "image.png").name
            if Path(fname).suffix.lower() not in IMAGE_EXTS:
                continue
            dst = in_dir / fname
            with open(dst, "wb") as w:
                shutil.copyfileobj(f.file, w)
            images.append(dst)
        images = list_images(in_dir)
    else:
        if not input_path:
            raise HTTPException(400, "input_path 가 필요합니다.")
        src = Path(input_path).expanduser()
        if not src.is_dir():
            raise HTTPException(400, f"입력 폴더가 없습니다: {src}")
        images = list_images(src)

    if not images:
        raise HTTPException(400, "분석할 이미지가 없습니다 (jpg/png/tif/bmp).")

    (out / "params.json").write_text(json.dumps(p.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    job = registry.submit(
        name=name, output_dir=str(out), n_images=len(images), params=p.to_dict(),
        fn=_make_job_fn(images, out, p),
    )
    return {"job_id": job.job_id, "output_dir": str(out), "n_images": len(images)}


@router.get("/jobs")
async def list_jobs():
    return [j.to_summary() for j in registry.list()]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "job 없음")
    return job.to_detail()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    ok = registry.cancel(job_id)
    return {"ok": ok}


def _resolve_in_output(job: Job, relpath: str) -> Path:
    base = Path(job.output_dir).resolve()
    target = (base / relpath).resolve()
    if base != target and base not in target.parents:
        raise HTTPException(403, "출력 폴더 밖 접근 불가")
    if not target.is_file():
        raise HTTPException(404, f"파일 없음: {relpath}")
    return target


@router.get("/jobs/{job_id}/files/{relpath:path}")
async def get_job_file(job_id: str, relpath: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "job 없음")
    target = _resolve_in_output(job, relpath)
    media = None
    ext = target.suffix.lower()
    if ext == ".xlsx":
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif ext == ".pptx":
        media = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    return FileResponse(str(target), media_type=media, filename=target.name if media else None)


@router.get("/jobs/{job_id}/zip")
async def get_job_zip(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "job 없음")
    base = Path(job.output_dir)
    if not base.is_dir():
        raise HTTPException(404, "출력 폴더 없음")

    def gen():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(base):
                for fn in files:
                    fp = Path(root) / fn
                    rel = fp.relative_to(base)
                    if rel.parts and rel.parts[0] == "input":
                        continue
                    zf.write(fp, str(rel))
                    if buf.tell() > 8 * 1024 * 1024:
                        yield buf.getvalue()
                        buf.seek(0)
                        buf.truncate()
        yield buf.getvalue()

    fname = f"{_safe_name(job.name)}.zip"
    return StreamingResponse(gen(), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})

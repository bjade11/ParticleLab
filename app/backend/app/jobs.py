"""배치 job 레지스트리 — 스레드 1개로 순차 실행, 진행률/로그/결과 보관.

서버 재시작 시 휘발. 동시 실행은 1개 (GPU 메모리 + CPU 부하 고려).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("particle_lab.jobs")

JobFn = Callable[["Job"], Any]


def _to_plain(obj: Any) -> Any:
    """dataclass / Path / numpy 스칼라 → JSON 친화 구조."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_plain(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np  # noqa
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


@dataclass
class Job:
    job_id: str
    name: str
    output_dir: str
    n_images: int
    params: dict
    status: str = "queued"          # queued | running | done | error | cancelled
    progress: int = 0
    message: str = ""
    log: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Any = None
    _cancel: bool = False
    _fn: Optional[JobFn] = None

    # ---- 콜백 (pipeline 에서 호출) ----
    def set_progress(self, pct: int, msg: str = "") -> None:
        self.progress = max(0, min(100, int(pct)))
        if msg:
            self.message = msg
            self.append_log(msg)

    def append_log(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {line}")
        if len(self.log) > 2000:
            del self.log[: len(self.log) - 2000]

    def cancel_requested(self) -> bool:
        return self._cancel

    def to_summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "output_dir": self.output_dir,
            "n_images": self.n_images,
        }

    def to_detail(self) -> dict:
        return {
            **self.to_summary(),
            "message": self.message,
            "log": self.log[-300:],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "params": self.params,
            "result": _to_plain(self.result) if self.result is not None else None,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="batch-worker", daemon=True)
        self._worker.start()

    # ---- public ----
    def submit(self, name: str, output_dir: str, n_images: int, params: dict, fn: JobFn) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            name=name,
            output_dir=output_dir,
            n_images=n_images,
            params=params,
        )
        job._fn = fn
        with self._lock:
            self._jobs[job.job_id] = job
        job.append_log(f"job 등록: {name} ({n_images}장) → {output_dir}")
        self._q.put(job.job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("done", "error", "cancelled"):
            return False
        job._cancel = True
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            job.append_log("대기 중 취소됨")
        else:
            job.append_log("취소 요청 — 현재 이미지 처리 후 중단")
        return True

    # ---- worker ----
    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            job.status = "running"
            job.started_at = time.time()
            job.append_log("실행 시작")
            try:
                result = job._fn(job) if job._fn else None
                job.result = result
                if job._cancel:
                    job.status = "cancelled"
                    job.append_log("취소로 종료 (부분 결과 저장됨)")
                else:
                    job.status = "done"
                    job.progress = 100
                    job.message = "완료"
                    job.append_log("완료")
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.append_log("오류: " + job.error)
                log.error("job %s failed\n%s", job.job_id, traceback.format_exc())
            finally:
                job.finished_at = time.time()


registry = JobRegistry()

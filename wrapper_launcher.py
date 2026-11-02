"""Particle Lab v2 — Windows desktop launcher (frozen into ParticleLab.exe).

When double-clicked, this orchestrates the existing portable bundle:
  - boots uvicorn backend (runtime/python/python.exe) on 127.0.0.1:8765
  - boots Next.js frontend (runtime/node/node.exe) on 127.0.0.1:3001
  - waits for both, then opens the default browser to localhost:3001

Both subprocesses are spawned headless (CREATE_NO_WINDOW), so the user sees
no terminal — only the browser tab.

Designed to live next to runtime/, app/, basic_pt/ etc. in the portable
bundle root. Detects whether it's running frozen (PyInstaller) or from
source via sys.frozen.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
BACKEND_PORT = 8765
FRONTEND_PORT = 3001


def _bundle_root() -> Path:
    """Where the portable bundle's runtime/ and app/ live, regardless of frozen vs source run."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _wait(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def _show_error(msg: str) -> None:
    """Pop a native MessageBox so the user actually sees failure (no console)."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "Particle Lab — 오류", 0x10)
    except Exception:
        pass


def _already_serving(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.7) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    root = _bundle_root()
    py = root / "runtime" / "python" / "python.exe"
    node = root / "runtime" / "node" / "node.exe"
    pkgs = root / "runtime" / "site-packages"
    backend = root / "app" / "backend"
    standalone = root / "app" / "frontend" / ".next" / "standalone"
    logs = root / "logs"

    for p, name in [(py, "runtime\\python\\python.exe"),
                     (node, "runtime\\node\\node.exe"),
                     (backend / "app" / "main.py", "app\\backend\\app\\main.py"),
                     (standalone / "server.js", "app\\frontend\\.next\\standalone\\server.js")]:
        if not p.exists():
            _show_error(f"번들 손상: {name} 파일이 없습니다.\n\n{p}")
            return 1

    # 이미 떠 있으면 (사용자가 .exe 더블클릭을 2번 했거나 STOP 안 누르고 다시 실행) — 그냥 브라우저만 열기
    be_up = _already_serving(f"http://127.0.0.1:{BACKEND_PORT}/api/health")
    fe_up = _already_serving(f"http://127.0.0.1:{FRONTEND_PORT}")
    if be_up and fe_up:
        try:
            webbrowser.open(f"http://localhost:{FRONTEND_PORT}", new=2)
        except Exception:
            pass
        return 0

    logs.mkdir(exist_ok=True)

    # Anaconda / system Python 차단 → 우리 번들 Python 만 사용
    env_base = os.environ.copy()
    for k in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        env_base.pop(k, None)
    env_base["PYTHONNOUSERSITE"] = "1"
    env_base["PYTHONUTF8"] = "1"

    env_be = env_base.copy()
    env_be["PYTHONPATH"] = f"{pkgs};{root};{backend}"

    if not be_up:
        be_log = open(logs / "backend.log", "wb")
        try:
            subprocess.Popen(
                [str(py), "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
                cwd=str(backend),
                env=env_be,
                stdout=be_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
                close_fds=True,
            )
        except Exception as e:
            _show_error(f"백엔드 시작 실패:\n{e}")
            return 1

    env_fe = env_base.copy()
    env_fe["PORT"] = str(FRONTEND_PORT)
    env_fe["HOSTNAME"] = "127.0.0.1"

    if not fe_up:
        fe_log = open(logs / "frontend.log", "wb")
        try:
            subprocess.Popen(
                [str(node), "server.js"],
                cwd=str(standalone),
                env=env_fe,
                stdout=fe_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
                close_fds=True,
            )
        except Exception as e:
            _show_error(f"프런트엔드 시작 실패:\n{e}")
            return 1

    if not _wait(f"http://127.0.0.1:{BACKEND_PORT}/api/health", 90):
        _show_error(
            "백엔드가 90초 안에 응답하지 않았습니다.\n"
            f"logs\\backend.log 확인 →\n{logs / 'backend.log'}"
        )
        return 1

    if not _wait(f"http://127.0.0.1:{FRONTEND_PORT}", 60):
        _show_error(
            "프런트엔드가 60초 안에 응답하지 않았습니다.\n"
            f"logs\\frontend.log 확인 →\n{logs / 'frontend.log'}"
        )
        return 1

    try:
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}", new=2)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

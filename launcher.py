"""Particle Lab v2 — 데스크탑 런처.

더블클릭 한 번으로 uvicorn(백엔드 + 정적 프런트)을 띄우고
pywebview 윈도우에 띄운다. Node.js 런타임 불필요 — 프런트는
app/frontend/out/ 의 Next.js export 산출물을 FastAPI 가 직접 서빙.

빌드:
    cd app/frontend && BUILD_TARGET=export npx next build

실행:
    python launcher.py
"""
from __future__ import annotations

import atexit
import base64
import html as _html
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller --onefile: data files are extracted to sys._MEIPASS.
    PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

BACKEND_DIR = PROJECT_ROOT / "app" / "backend"
FRONT_OUT = PROJECT_ROOT / "app" / "frontend" / "out"
DEFAULT_DOWNLOAD_DIR = str(Path.home() / "Downloads")

# 휴대용 번들 (build_desktop_app.sh 산출물) 에서는 의존성이 runtime/site-packages 에 있다.
# 개발 환경에서는 디렉토리가 없으므로 조용히 패스.
_BUNDLED_SITE_PKGS = PROJECT_ROOT / "runtime" / "site-packages"
if _BUNDLED_SITE_PKGS.is_dir() and str(_BUNDLED_SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_SITE_PKGS))

# hybrid_core / evaluate / unet_* 는 PROJECT_ROOT 에 평면 모듈로 존재. PROJECT_ROOT 를 path 에.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 임시 PDF 누적 → 런처 종료 시 일괄 cleanup
_TEMP_FILES: list[str] = []


def _cleanup_temp() -> None:
    for p in _TEMP_FILES:
        try:
            os.remove(p)
            d = os.path.dirname(p)
            if d.startswith(tempfile.gettempdir()) and not os.listdir(d):
                os.rmdir(d)
        except Exception:
            pass


atexit.register(_cleanup_temp)


# WebKit (pywebview macOS 백엔드) 은 <a download> / data: / blob: URL 을 자동 다운로드하지 않고,
# jsPDF 같은 라이브러리는 Safari 감지 시 window.open(blobURL) 로 fallback 한다.
# 두 경로를 모두 가로채서 Python 쪽 JsApi 로 넘긴다.
_DOWNLOAD_HOOK_JS = r"""
(function () {
  if (window.__plabDownloadHook) return;
  window.__plabDownloadHook = true;

  function bytesToB64(bytes) {
    let bin = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(bin);
  }

  async function hrefToB64(href) {
    if (href.startsWith('data:')) {
      const comma = href.indexOf(',');
      const meta = href.slice(5, comma);
      const payload = href.slice(comma + 1);
      if (meta.includes('base64')) return { b64: payload, mime: meta.split(';')[0] };
      const decoded = decodeURIComponent(payload);
      const bytes = new Uint8Array(decoded.length);
      for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i) & 0xff;
      return { b64: bytesToB64(bytes), mime: meta.split(';')[0] };
    }
    const r = await fetch(href, { credentials: 'same-origin' });
    if (!r.ok) throw new Error('fetch ' + r.status);
    const blob = await r.blob();
    const buf = await blob.arrayBuffer();
    return { b64: bytesToB64(new Uint8Array(buf)), mime: blob.type || '' };
  }

  function isPdf(name, mime) {
    if (mime && mime.toLowerCase().includes('pdf')) return true;
    if (name && name.toLowerCase().endsWith('.pdf')) return true;
    return false;
  }

  async function dispatch(name, b64, mime) {
    try {
      const safeName = name || (isPdf(name, mime) ? 'report.pdf' : 'download');
      const r = await window.pywebview.api.save_download(safeName, b64);
      if (r && r.ok === false && r.error !== 'cancelled') alert('저장 실패: ' + r.error);
    } catch (err) {
      console.error('[plab] dispatch:', err);
      alert('다운로드 실패: ' + (err && err.message ? err.message : String(err)));
    }
  }

  // (1) <a download> 클릭 (직접 + jsPDF 의 anchor click 경로)
  document.addEventListener('click', async function (e) {
    const a = e.target && e.target.closest ? e.target.closest('a[download]') : null;
    if (!a) return;
    const raw = a.getAttribute('href') || a.href || '';
    if (!raw || raw === '#' || raw.startsWith('javascript:')) return;
    e.preventDefault();
    e.stopPropagation();
    const name = a.getAttribute('download') || '';
    try {
      const { b64, mime } = await hrefToB64(a.href || raw);
      await dispatch(name, b64, mime);
    } catch (err) {
      console.error('[plab] click intercept:', err);
      alert('다운로드 실패: ' + (err && err.message ? err.message : String(err)));
    }
  }, true);

  // (2) window.open(blobURL / data:) — jsPDF 의 Safari fallback 경로
  const _origOpen = window.open;
  window.open = function (url, name, features) {
    if (typeof url === 'string' && (url.startsWith('blob:') || url.startsWith('data:'))) {
      (async () => {
        try {
          const { b64, mime } = await hrefToB64(url);
          // window.open 으로 띄우는 PDF/blob 은 거의 항상 사용자가 "본다 + 저장한다" 가 목적 → PDF preview 우선
          const guessName = (name && /\.\w{2,5}$/.test(name)) ? name : 'report.pdf';
          await dispatch(guessName, b64, mime || (url.startsWith('data:application/pdf') ? 'application/pdf' : ''));
        } catch (err) {
          console.error('[plab] open intercept:', err);
          alert('미리보기 실패: ' + (err && err.message ? err.message : String(err)));
        }
      })();
      return null;
    }
    return _origOpen.call(this, url, name, features);
  };

  // (helper) 페이지 코드가 직접 Blob 으로 저장하고 싶을 때:
  //   await window.__plabSaveBlob('report.pdf', blob)
  // jsPDF / pptxgenjs / xlsx 등 라이브러리의 자체 download 트리거를 건너뛰고
  // 우리 bridge 로 바로 보낸다 (macOS WebKit 의 download intercept 불안정성 회피).
  window.__plabSaveBlob = async function (name, blob) {
    const buf = await blob.arrayBuffer();
    const b64 = bytesToB64(new Uint8Array(buf));
    return await window.pywebview.api.save_download(name, b64);
  };

  // (3) <a target="_blank" href="blob:...">  (드물지만 보호)
  document.addEventListener('click', async function (e) {
    const a = e.target && e.target.closest ? e.target.closest('a[target="_blank"]') : null;
    if (!a) return;
    const href = a.href || '';
    if (!href.startsWith('blob:') && !href.startsWith('data:')) return;
    e.preventDefault();
    e.stopPropagation();
    try {
      const { b64, mime } = await hrefToB64(href);
      await dispatch(a.getAttribute('download') || 'download', b64, mime);
    } catch (err) {
      console.error('[plab] blank intercept:', err);
    }
  }, true);
})();
"""


def _preview_html(name: str, size_bytes: int, pdf_token: str) -> str:
    title = _html.escape(name)
    kb = size_bytes / 1024
    size_str = f"{kb / 1024:.2f} MB" if kb >= 1024 else f"{kb:.1f} KB"
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<style>
  *{{box-sizing:border-box}}
  html,body{{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#1a1a1a;color:#e8e8e8}}
  .wrap{{display:flex;flex-direction:column;height:100vh}}
  header{{padding:14px 18px;background:#252525;border-bottom:1px solid #3a3a3a}}
  .name{{font-size:14px;font-weight:600;color:#fff;word-break:break-all}}
  .meta{{font-size:12px;color:#999;margin-top:4px}}
  main{{flex:1;display:flex;align-items:center;justify-content:center;padding:24px}}
  .card{{text-align:center;max-width:420px}}
  .icon{{font-size:64px;margin-bottom:12px;opacity:.9}}
  .label{{font-size:13px;color:#bbb;line-height:1.55}}
  footer{{padding:14px 18px;background:#222;border-top:1px solid #3a3a3a;display:flex;gap:10px;justify-content:flex-end}}
  button{{padding:8px 18px;font-size:13px;font-weight:500;border:0;border-radius:6px;cursor:pointer;font-family:inherit}}
  button.primary{{background:#3a82f6;color:#fff}}
  button.primary:hover{{background:#2563eb}}
  button.secondary{{background:#3a3a3a;color:#e8e8e8}}
  button.secondary:hover{{background:#4a4a4a}}
  button:disabled{{opacity:.5;cursor:not-allowed}}
  .toast{{position:fixed;bottom:80px;left:50%;transform:translateX(-50%);background:#1f3a1f;border:1px solid #2d5a2d;color:#a7f3a7;padding:10px 16px;border-radius:6px;font-size:13px;opacity:0;transition:opacity .25s}}
  .toast.show{{opacity:1}}
  .toast.err{{background:#3a1f1f;border-color:#5a2d2d;color:#f7a7a7}}
</style></head>
<body>
<div class="wrap">
  <header>
    <div class="name">{title}</div>
    <div class="meta">PDF 리포트 · {size_str}</div>
  </header>
  <main>
    <div class="card">
      <div class="icon">📄</div>
      <div class="label">
        리포트가 준비되었습니다.<br/>
        저장하거나 기본 PDF 뷰어로 바로 열 수 있습니다.
      </div>
    </div>
  </main>
  <footer>
    <button class="secondary" id="btn-open">기본 뷰어로 열기</button>
    <button class="secondary" id="btn-close">닫기</button>
    <button class="primary" id="btn-save">저장하기</button>
  </footer>
</div>
<div class="toast" id="toast"></div>
<script>
  const TOKEN = {pdf_token!r};
  const NAME  = {name!r};
  function toast(msg, err) {{
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast show' + (err ? ' err' : '');
    setTimeout(() => {{ t.className = 'toast' + (err ? ' err' : ''); }}, 2200);
  }}
  function setBusy(b) {{
    for (const id of ['btn-save', 'btn-open', 'btn-close']) {{
      document.getElementById(id).disabled = b;
    }}
  }}
  document.getElementById('btn-save').addEventListener('click', async () => {{
    setBusy(true);
    try {{
      const r = await window.pywebview.api.save_temp(TOKEN, NAME);
      if (r && r.ok) toast('저장됨: ' + r.path);
      else if (r && r.error !== 'cancelled') toast('저장 실패: ' + r.error, true);
    }} catch (e) {{ toast('오류: ' + e, true); }}
    setBusy(false);
  }});
  document.getElementById('btn-open').addEventListener('click', async () => {{
    setBusy(true);
    try {{
      const r = await window.pywebview.api.open_external(TOKEN);
      if (r && !r.ok) toast('열기 실패: ' + r.error, true);
    }} catch (e) {{ toast('오류: ' + e, true); }}
    setBusy(false);
  }});
  document.getElementById('btn-close').addEventListener('click', async () => {{
    try {{ await window.pywebview.api.close_preview(TOKEN); }}
    catch (e) {{ /* ignore */ }}
  }});
</script>
</body></html>"""


class JsApi:
    """Python ↔ JS 브릿지.

    save_download:     일반 파일 → 즉시 네이티브 저장 다이얼로그.
    show_pdf_preview:  PDF → 별도 미리보기 창 (저장 / 외부 뷰어 / 닫기).
    save_temp:         미리보기 창에서 호출 — 임시파일을 사용자 지정 위치로 복사.
    open_external:     미리보기 창에서 호출 — OS 기본 뷰어로 열기.
    close_preview:     미리보기 창 닫기.
    """

    def __init__(self) -> None:
        self.window = None  # 메인 창
        self._last_dir = DEFAULT_DOWNLOAD_DIR
        # token → (temp_path, preview_window) — 미리보기 창 추적
        self._previews: dict[str, tuple[str, "object"]] = {}

    # ---------- 일반 파일 ----------
    def save_download(self, name: str, data_b64: str) -> dict:
        try:
            data = base64.b64decode(data_b64)
        except Exception as e:
            return {"ok": False, "error": f"decode: {e}"}
        path = self._ask_save_path(name)
        if path is None:
            return {"ok": False, "error": "cancelled"}
        try:
            with open(path, "wb") as f:
                f.write(data)
            self._last_dir = os.path.dirname(path) or self._last_dir
            return {"ok": True, "path": path, "bytes": len(data)}
        except Exception as e:
            return {"ok": False, "error": f"write: {e}"}

    # ---------- PDF 미리보기 ----------
    def show_pdf_preview(self, name: str, data_b64: str) -> dict:
        if not name.lower().endswith(".pdf"):
            name = name + ".pdf"
        try:
            data = base64.b64decode(data_b64)
        except Exception as e:
            return {"ok": False, "error": f"decode: {e}"}
        try:
            tmpdir = tempfile.mkdtemp(prefix="plab_pdf_")
            tmp_path = os.path.join(tmpdir, name)
            with open(tmp_path, "wb") as f:
                f.write(data)
            _TEMP_FILES.append(tmp_path)
        except Exception as e:
            return {"ok": False, "error": f"tempwrite: {e}"}

        token = base64.urlsafe_b64encode(os.urandom(9)).decode().rstrip("=")
        try:
            import webview  # type: ignore
            html_doc = _preview_html(name, len(data), token)
            preview = webview.create_window(
                title=name,
                html=html_doc,
                width=520,
                height=380,
                resizable=True,
                js_api=self,
            )
            self._previews[token] = (tmp_path, preview)

            def _on_closed(t: str = token) -> None:
                self._previews.pop(t, None)

            if hasattr(preview, "events") and hasattr(preview.events, "closed"):
                preview.events.closed += _on_closed
            return {"ok": True, "token": token}
        except Exception as e:
            return {"ok": False, "error": f"window: {e}"}

    def save_temp(self, token: str, default_name: str) -> dict:
        entry = self._previews.get(token)
        if entry is None:
            return {"ok": False, "error": "expired"}
        src_path, _ = entry
        if not os.path.exists(src_path):
            return {"ok": False, "error": "temp missing"}
        path = self._ask_save_path(default_name)
        if path is None:
            return {"ok": False, "error": "cancelled"}
        try:
            shutil.copy(src_path, path)
            self._last_dir = os.path.dirname(path) or self._last_dir
            return {"ok": True, "path": path}
        except Exception as e:
            return {"ok": False, "error": f"copy: {e}"}

    def open_external(self, token: str) -> dict:
        entry = self._previews.get(token)
        if entry is None:
            return {"ok": False, "error": "expired"}
        src_path, _ = entry
        if not os.path.exists(src_path):
            return {"ok": False, "error": "temp missing"}
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", src_path])
            elif sys.platform.startswith("win"):
                os.startfile(src_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", src_path])
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def close_preview(self, token: str) -> dict:
        entry = self._previews.pop(token, None)
        if entry is None:
            return {"ok": True}
        _, preview = entry
        try:
            preview.destroy()
        except Exception:
            pass
        return {"ok": True}

    # ---------- helpers ----------
    def _ask_save_path(self, default_name: str) -> "str | None":
        target = self.window
        if target is None and self._previews:
            target = next(iter(self._previews.values()))[1]
        if target is None:
            return None
        try:
            import webview  # type: ignore
            result = target.create_file_dialog(
                webview.SAVE_DIALOG,
                directory=self._last_dir,
                save_filename=default_name,
            )
        except Exception as e:
            print(f"[launcher] dialog error: {e}", file=sys.stderr)
            return None
        if not result:
            return None
        return result if isinstance(result, str) else result[0]


def _die(msg: str) -> "None":
    print(f"[launcher] FATAL: {msg}", file=sys.stderr)
    # frozen --noconsole 모드에서는 stderr 가 안 보이므로 네이티브 MessageBox 로도 띄움
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "Particle Lab — 오류", 0x10)
        except Exception:
            pass
    sys.exit(1)


def _pick_port(preferred: int = 8765) -> int:
    for p in (preferred,) + tuple(range(8766, 8790)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> None:
    if not FRONT_OUT.is_dir():
        _die(
            f"정적 프런트 번들이 없습니다: {FRONT_OUT}\n"
            "먼저 빌드하세요: cd app/frontend && BUILD_TARGET=export npx next build"
        )

    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        _die("uvicorn 미설치. pip install -r app/backend/requirements.txt")

    try:
        import webview  # type: ignore
    except ImportError:
        _die("pywebview 미설치. pip install pywebview")

    from app.main import app as fastapi_app  # type: ignore

    port = _pick_port(8765)
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    config = uvicorn.Config(
        fastapi_app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run_server() -> None:
        server.run()

    t = threading.Thread(target=_run_server, name="uvicorn", daemon=True)
    t.start()

    if not _wait_health(f"{base_url}/api/health"):
        _die("백엔드 시작 실패 — 로그 확인")

    print(f"[launcher] backend up → {base_url}")

    js_api = JsApi()
    window = webview.create_window(
        "Particle Lab v2",
        url=base_url,
        width=1440,
        height=900,
        min_size=(1024, 700),
        js_api=js_api,
    )
    js_api.window = window

    def _inject_hook() -> None:
        try:
            window.evaluate_js(_DOWNLOAD_HOOK_JS)
        except Exception as e:
            print(f"[launcher] inject hook failed: {e}", file=sys.stderr)

    def _shutdown() -> None:
        server.should_exit = True

    if hasattr(window, "events"):
        if hasattr(window.events, "loaded"):
            window.events.loaded += _inject_hook
        if hasattr(window.events, "closed"):
            window.events.closed += _shutdown

    webview.start()
    _shutdown()
    t.join(timeout=3.0)


if __name__ == "__main__":
    main()

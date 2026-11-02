"""API 통합 스모크 테스트.

사용법 (venv python 으로):
    python tests/smoke_api.py <images_dir> [--ai]

- uvicorn 을 서브프로세스로 띄우고 (포트 8799)
- /api/health, /api/tuner/preprocess, /api/tuner/segment (watershed, --ai 면 ai_hybrid 도)
- /api/batch/jobs (server_path) → 폴링 → 결과 파일 존재 확인
- /api/batch/jobs/{id}/zip 다운로드 크기 확인
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"


def http(method: str, path: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 600):
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            ctype = r.headers.get("content-type", "")
            return r.status, body, ctype
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"HTTP {e.code} {path}: {detail}") from None


def multipart(fields: dict[str, str], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = "----plabsmoke%d" % int(time.time() * 1000)
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8")
    for k, p in files:
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{p.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode("utf-8")
        out += p.read_bytes()
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def wait_health(timeout: float = 120) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            st, body, _ = http("GET", "/api/health", timeout=3)
            if st == 200:
                return json.loads(body)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"backend not up: {last}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    img_dir = Path(sys.argv[1]).resolve()
    use_ai = "--ai" in sys.argv
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}])
    assert images, f"no images in {img_dir}"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT};{ROOT / 'app' / 'backend'}"
    env["MPLBACKEND"] = "Agg"
    log_path = ROOT / "tests" / "smoke_api_server.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT / "app" / "backend"), env=env, stdout=log_f, stderr=subprocess.STDOUT,
    )
    ok = True
    try:
        h = wait_health()
        print("health:", h)

        default_params = {"engine": "watershed", "spatial": {"enable_boundary": True, "enable_nnd": True, "enable_dbscan": True}}

        # tuner preprocess
        body, ctype = multipart({"params": json.dumps(default_params)}, [("image", images[0])])
        st, out, _ = http("POST", "/api/tuner/preprocess", body, {"Content-Type": ctype})
        j = json.loads(out)
        assert j["preview_png"].startswith("data:image/png;base64,"), "preview missing"
        print("tuner/preprocess OK shape", j["shape"])

        # tuner segment (watershed)
        engines = ["watershed"] + (["ai_hybrid"] if use_ai else [])
        for eng in engines:
            params = dict(default_params, engine=eng)
            body, ctype = multipart({"params": json.dumps(params)}, [("image", images[0])])
            t0 = time.time()
            st, out, _ = http("POST", "/api/tuner/segment", body, {"Content-Type": ctype})
            j = json.loads(out)
            views = {k: (v is not None) for k, v in j["views"].items()}
            print(f"tuner/segment[{eng}] OK n_before={j['n_before']} n_after={j['n_after']} "
                  f"p2m={j['pixel_to_micro']:.4f} d50={j['d_values_before'].get('d50')} "
                  f"views={views} meta={j.get('engine_meta')} {time.time()-t0:.1f}s")
            assert j["n_before"] > 0, f"{eng}: no particles"
            assert j["views"]["labeled_before"], "labeled view missing"

        # batch job (server_path)
        for eng in engines:
            params = dict(default_params, engine=eng)
            body, ctype = multipart({
                "params": json.dumps(params),
                "input_mode": "server_path",
                "input_path": str(img_dir),
                "job_name": f"smoke_{eng}",
            }, [])
            st, out, _ = http("POST", "/api/batch/jobs", body, {"Content-Type": ctype})
            j = json.loads(out)
            job_id = j["job_id"]
            print(f"batch[{eng}] job {job_id} n_images={j['n_images']} out={j['output_dir']}")
            t0 = time.time()
            while True:
                st, out, _ = http("GET", f"/api/batch/jobs/{job_id}")
                d = json.loads(out)
                if d["status"] in ("done", "error", "cancelled"):
                    break
                time.sleep(1.0)
                if time.time() - t0 > 900:
                    raise TimeoutError("job too slow")
            print(f"  status={d['status']} in {time.time()-t0:.1f}s; last log: {d['log'][-3:]}")
            if d["status"] != "done":
                print("  ERROR:", d.get("error"))
                ok = False
                continue
            res = d["result"]
            out_dir = Path(res["output_dir"])
            assert Path(res["excel_path"]).exists(), "excel missing"
            if res.get("pptx_path"):
                assert Path(res["pptx_path"]).exists(), "pptx missing"
            n_files = 0
            missing = []
            for im in res["images"]:
                for cat, rel in im["files"].items():
                    n_files += 1
                    if not (out_dir / rel).exists():
                        missing.append(rel)
            for s, files in res.get("sample_files", {}).items():
                for cat, rel in files.items():
                    n_files += 1
                    if not (out_dir / rel).exists():
                        missing.append(rel)
            print(f"  images={len(res['images'])} samples={len(res['samples'])} files={n_files} missing={missing[:5]} warnings={res.get('warnings', [])[:3]}")
            assert not missing, f"missing output files: {missing[:5]}"
            # sample file serve
            first_rel = next(iter(res["images"][0]["files"].values()))
            st, out, ctype2 = http("GET", f"/api/batch/jobs/{job_id}/files/{first_rel.replace(os.sep, '/')}")
            assert st == 200 and len(out) > 100, "file serve failed"
            st, out, ctype2 = http("GET", f"/api/batch/jobs/{job_id}/zip")
            print(f"  zip {len(out)/1024:.0f} KB ctype={ctype2}")
            assert len(out) > 1000

        # fs
        st, out, _ = http("GET", f"/api/fs/list?path={urllib.request.quote(str(img_dir))}")
        j = json.loads(out)
        print("fs/list n_images", j["n_images"])
    except Exception as e:  # noqa: BLE001
        ok = False
        import traceback
        traceback.print_exc()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
        log_f.close()
    if not ok:
        print("---- server log tail ----")
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

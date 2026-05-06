#!/usr/bin/env python3
"""
vLLM Instance Manager — stdlib-only HTTP server.

Manages vLLM subprocess instances, rewrites nginx route snippets on each
change, and serves the web management UI.

Endpoints
─────────
  GET  /                     → index.html (web UI)
  GET  /api/health           → {"status": "ok"}
  GET  /api/instances        → {"instances": [...]}
  POST /api/launch           → {"cmd": "model [args]", "hf_token": "..."?}
  DELETE /api/instances/<n>  → stop instance in slot n
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

# ── Configuration ────────────────────────────────────────────────────────────
LISTEN_HOST       = "127.0.0.1"
LISTEN_PORT       = int(os.environ.get("MANAGER_PORT", 9000))
VLLM_BASE_PORT    = int(os.environ.get("VLLM_BASE_PORT", 8101))
MAX_SLOTS         = int(os.environ.get("VLLM_MAX_INSTANCES", 5))
NGINX_ROUTES_FILE = os.environ.get("NGINX_ROUTES_FILE", "/tmp/nginx/vllm_routes.conf")
NGINX_CONF_FILE   = os.environ.get("NGINX_CONF_FILE", "/tmp/nginx/nginx.conf")
NGINX_BIN_ENV     = os.environ.get("NGINX_BIN", "")
STATIC_DIR        = Path(__file__).parent / "static"

# ── State ────────────────────────────────────────────────────────────────────
# {slot: {"pid", "port", "cmd", "started_at", "proc"}}
_instances: Dict[int, dict] = {}
_lock = threading.Lock()


# ── Nginx helpers ─────────────────────────────────────────────────────────────

def _find_nginx() -> str:
    if NGINX_BIN_ENV and os.path.isfile(NGINX_BIN_ENV):
        return NGINX_BIN_ENV
    for candidate in ("/usr/sbin/nginx", "/usr/bin/nginx", "/usr/local/sbin/nginx"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    r = subprocess.run(["which", "nginx"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return "nginx"


def _write_nginx_routes() -> None:
    lines = []
    for slot, info in sorted(_instances.items()):
        port = info["port"]
        lines += [
            f"location /vllm/{slot}/ {{",
            f"    proxy_pass         http://127.0.0.1:{port}/;",
             "    proxy_set_header   Host $host;",
             "    proxy_set_header   X-Real-IP $remote_addr;",
            f"    proxy_read_timeout 600;",
            f"    proxy_send_timeout 600;",
             "    proxy_buffering    off;",   # SSE / streaming
             "    proxy_cache        off;",
            f"}}",
            "",
        ]
    path = Path(NGINX_ROUTES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _reload_nginx() -> None:
    try:
        subprocess.run([_find_nginx(), "-c", NGINX_CONF_FILE, "-s", "reload"],
                       check=True, capture_output=True, timeout=10)
    except Exception as exc:
        print(f"[manager] nginx reload failed: {exc}", flush=True)


# ── Instance lifecycle ────────────────────────────────────────────────────────

def _next_free_slot() -> Optional[int]:
    used = set(_instances)
    for s in range(1, MAX_SLOTS + 1):
        if s not in used:
            return s
    return None


def launch_instance(raw_cmd: str, hf_token: Optional[str] = None) -> dict:
    with _lock:
        slot = _next_free_slot()
        if slot is None:
            raise RuntimeError(f"All {MAX_SLOTS} slots are in use")

        port = VLLM_BASE_PORT + slot - 1

        # Normalise: strip leading "vllm serve" if user included it
        parts = raw_cmd.strip().split()
        if parts[:2] == ["vllm", "serve"]:
            parts = parts[2:]
        elif parts[:1] == ["vllm"]:
            parts = parts[1:]

        argv = ["vllm", "serve"] + parts + [
            "--port", str(port),
            "--host", "127.0.0.1",
            "--root-path", f"/vllm/{slot}",
        ]

        env = {**os.environ}
        if hf_token:
            env["HF_TOKEN"] = hf_token

        log_path = Path(f"/tmp/vllm_slot_{slot}.log")
        log_fh   = open(log_path, "w")

        proc = subprocess.Popen(argv, env=env,
                                 stdout=log_fh, stderr=subprocess.STDOUT)

        _instances[slot] = {
            "pid":        proc.pid,
            "port":       port,
            "cmd":        " ".join(parts),
            "started_at": time.time(),
            "proc":       proc,
            "log":        str(log_path),
        }

        _write_nginx_routes()
        _reload_nginx()

        print(f"[manager] launched slot={slot} port={port} "
              f"pid={proc.pid} model={parts[:1]}", flush=True)
        return {"slot": slot, "port": port, "pid": proc.pid}


def stop_instance(slot: int) -> None:
    with _lock:
        if slot not in _instances:
            raise KeyError(f"Slot {slot} not found")
        info = _instances.pop(slot)
        proc = info["proc"]
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            pass
        _write_nginx_routes()
        _reload_nginx()
        print(f"[manager] stopped slot={slot} pid={info['pid']}", flush=True)


def list_instances() -> list:
    with _lock:
        result = []
        for slot, info in sorted(_instances.items()):
            running = info["proc"].poll() is None
            result.append({
                "slot":       slot,
                "port":       info["port"],
                "cmd":        info["cmd"],
                "pid":        info["pid"],
                "started_at": info["started_at"],
                "status":     "running" if running else "exited",
                "log":        info["log"],
            })
        return result


# ── Background zombie reaper ─────────────────────────────────────────────────

def _reap_loop() -> None:
    while True:
        time.sleep(15)
        with _lock:
            dead = [s for s, i in _instances.items() if i["proc"].poll() is not None]
            for s in dead:
                code = _instances[s]["proc"].returncode
                print(f"[manager] slot {s} exited (code={code})", flush=True)
                _instances.pop(s)
            if dead:
                _write_nginx_routes()
                _reload_nginx()


# ── HTTP handler ─────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress per-request access log

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _path(self) -> str:
        return self.path.split("?")[0].rstrip("/") or "/"

    # ── GET ─────────────────────────────────────────────────────────────────
    def do_GET(self):
        p = self._path()
        if p in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif p == "/api/instances":
            self._send_json(200, {"instances": list_instances(),
                                  "max_slots": MAX_SLOTS})
        elif p == "/api/health":
            self._send_json(200, {"status": "ok", "slots_used": len(_instances),
                                  "slots_max": MAX_SLOTS})
        else:
            self._send_json(404, {"error": "not found"})

    # ── POST ────────────────────────────────────────────────────────────────
    def do_POST(self):
        if self._path() == "/api/launch":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length) or b"{}")
            cmd    = (body.get("cmd") or "").strip()
            if not cmd:
                self._send_json(400, {"error": "'cmd' is required"})
                return
            hf_token = body.get("hf_token") or None
            try:
                result = launch_instance(cmd, hf_token)
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
        else:
            self._send_json(404, {"error": "not found"})

    # ── DELETE ──────────────────────────────────────────────────────────────
    def do_DELETE(self):
        p = self._path()
        if p.startswith("/api/instances/"):
            try:
                slot = int(p.split("/")[-1])
            except ValueError:
                self._send_json(400, {"error": "invalid slot"})
                return
            try:
                stop_instance(slot)
                self._send_json(200, {"stopped": slot})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
        else:
            self._send_json(404, {"error": "not found"})

    # CORS pre-flight
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure routes file exists (empty) so nginx can include it at startup
    routes = Path(NGINX_ROUTES_FILE)
    routes.parent.mkdir(parents=True, exist_ok=True)
    if not routes.exists():
        routes.write_text("")

    threading.Thread(target=_reap_loop, daemon=True).start()

    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), _Handler)
    print(f"[manager] listening on {LISTEN_HOST}:{LISTEN_PORT} "
          f"(max_slots={MAX_SLOTS}, base_port={VLLM_BASE_PORT})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

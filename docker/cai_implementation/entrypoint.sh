#!/usr/bin/env bash
# Advanced CAI entrypoint — nginx ingress + web-based vLLM launcher.
#
# Architecture (only port 8080 is externally reachable):
#
#   External HTTPS → :8080 (nginx)
#     /vllm/<slot>/   →  127.0.0.1:(8100+slot)  — vLLM OpenAI-compat API
#     /*              →  127.0.0.1:9000          — manager.py (web UI + launch API)
#
# Environment variables:
#   MODEL               If set, auto-launch a default vLLM instance at slot 1
#   VLLM_ARGS           Extra flags for the auto-launched MODEL instance
#   HF_TOKEN            HuggingFace token (forwarded to all vLLM instances)
#   VLLM_MAX_INSTANCES  Max concurrent instances (default: 5)
#   MANAGER_PORT        Internal port for manager.py (default: 9000)
#   NGINX_RUNTIME_DIR   Directory for nginx runtime files (default: /tmp/nginx)

set -euo pipefail

NGINX_RUNTIME_DIR="${NGINX_RUNTIME_DIR:-/tmp/nginx}"
NGINX_ROUTES_FILE="${NGINX_RUNTIME_DIR}/vllm_routes.conf"
MANAGER_PORT="${MANAGER_PORT:-9000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Find nginx ──────────────────────────────────────────────────────────────
NGINX_BIN=""
for _candidate in /usr/sbin/nginx /usr/bin/nginx /usr/local/sbin/nginx; do
    if [[ -x "$_candidate" ]]; then
        NGINX_BIN="$_candidate"
        break
    fi
done

if [[ -z "$NGINX_BIN" ]]; then
    echo "[entrypoint] nginx not found — installing via apt-get..."
    apt-get update -qq && apt-get install -y -qq nginx
    NGINX_BIN="/usr/sbin/nginx"
fi
echo "[entrypoint] nginx binary: $NGINX_BIN"
export NGINX_BIN

# ── Runtime directories ─────────────────────────────────────────────────────
mkdir -p "${NGINX_RUNTIME_DIR}/logs" "${NGINX_RUNTIME_DIR}/run"
# Empty routes file must exist before nginx starts
touch "${NGINX_ROUTES_FILE}"

# ── Locate mime.types ───────────────────────────────────────────────────────
MIME_TYPES=""
for _mt in /etc/nginx/mime.types /usr/share/nginx/mime.types /usr/local/nginx/conf/mime.types; do
    if [[ -f "$_mt" ]]; then MIME_TYPES="$_mt"; break; fi
done
MIME_TYPES="${MIME_TYPES:-/etc/nginx/mime.types}"

# ── Write nginx.conf ────────────────────────────────────────────────────────
cat > "${NGINX_RUNTIME_DIR}/nginx.conf" <<NGINX_CONF
worker_processes auto;
error_log  ${NGINX_RUNTIME_DIR}/logs/error.log warn;
pid        ${NGINX_RUNTIME_DIR}/run/nginx.pid;

events { worker_connections 1024; }

http {
    include      ${MIME_TYPES};
    default_type application/octet-stream;

    sendfile           on;
    keepalive_timeout  65;
    client_max_body_size 200m;

    # Long timeouts — LLM inference can run for tens of seconds
    proxy_read_timeout    600;
    proxy_send_timeout    600;
    proxy_connect_timeout  10;

    server {
        listen      8080;
        server_name _;

        # vLLM instance routes — dynamically updated by manager.py
        # Each entry: location /vllm/<slot>/ { proxy_pass http://127.0.0.1:<port>/; }
        include ${NGINX_ROUTES_FILE};

        # Management web UI + launch API (manager.py)
        location / {
            proxy_pass         http://127.0.0.1:${MANAGER_PORT}/;
            proxy_set_header   Host \$host;
            proxy_set_header   X-Real-IP \$remote_addr;
            # SSE / streaming passthrough
            proxy_buffering    off;
            proxy_cache        off;
            proxy_read_timeout 30;
        }
    }
}
NGINX_CONF

echo "[entrypoint] nginx config: ${NGINX_RUNTIME_DIR}/nginx.conf"

# ── HuggingFace login ───────────────────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[entrypoint] logging in to Hugging Face Hub..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
fi

# ── Start manager.py ────────────────────────────────────────────────────────
NGINX_CONF_FILE="${NGINX_RUNTIME_DIR}/nginx.conf"
export NGINX_ROUTES_FILE NGINX_CONF_FILE NGINX_BIN MANAGER_PORT
export VLLM_MAX_INSTANCES="${VLLM_MAX_INSTANCES:-5}"

python3 "${SCRIPT_DIR}/manager.py" &
MANAGER_PID=$!
echo "[entrypoint] manager.py started (pid=${MANAGER_PID})"

# Wait until manager is ready (up to 10 s)
for _i in $(seq 1 20); do
    if python3 -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://127.0.0.1:${MANAGER_PORT}/api/health', timeout=1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[entrypoint] manager ready"
        break
    fi
    sleep 0.5
done

# ── Auto-launch MODEL if set ────────────────────────────────────────────────
if [[ -n "${MODEL:-}" ]]; then
    echo "[entrypoint] auto-launching MODEL=${MODEL}"
    python3 - <<'PYEOF'
import json, os, urllib.request

cmd  = os.environ["MODEL"].strip()
args = os.environ.get("VLLM_ARGS", "").strip()
if args:
    cmd = cmd + " " + args

payload = {"cmd": cmd}
hf = os.environ.get("HF_TOKEN", "")
if hf:
    payload["hf_token"] = hf

port = os.environ.get("MANAGER_PORT", "9000")
req  = urllib.request.Request(
    f"http://127.0.0.1:{port}/api/launch",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urllib.request.urlopen(req, timeout=10)
    print(f"[entrypoint] MODEL auto-launch request sent (slot 1, port 8101)")
except Exception as e:
    print(f"[entrypoint] WARNING: auto-launch request failed: {e}")
PYEOF
fi

# ── Start nginx in foreground (keeps container alive) ───────────────────────
echo ""
echo "  Web UI    : http://<host>:8080/"
echo "  Launch API: http://<host>:8080/api/launch"
echo "  vLLM API  : http://<host>:8080/vllm/<slot>/v1/"
echo ""
echo "[entrypoint] starting nginx on :8080 (foreground) ..."
exec "${NGINX_BIN}" -c "${NGINX_RUNTIME_DIR}/nginx.conf" -g "daemon off;"

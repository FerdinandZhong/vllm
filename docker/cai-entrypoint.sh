#!/usr/bin/env bash
# Entrypoint for the Cloudera AI Inference (CAI) vLLM container.
#
# CAI maps external HTTPS (443) → container port 8080.  The --port flag is
# hardcoded here so the server always binds to 8080 regardless of any extra
# arguments passed by the caller.
#
# Required environment variable:
#   MODEL   HuggingFace model ID or path to load (e.g. meta-llama/Llama-3.1-8B-Instruct)
#
# Optional environment variables:
#   VLLM_ARGS   Space-separated extra flags forwarded to `vllm serve`
#               (e.g. "--tensor-parallel-size 2 --max-model-len 8192")
#   HF_TOKEN    HuggingFace token for gated/private models

set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL environment variable is required." >&2
    echo "  Example: docker run -e MODEL=meta-llama/Llama-3.1-8B-Instruct ..." >&2
    exit 1
fi

# Pass HF_TOKEN to huggingface-cli if provided.
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Logging in to Hugging Face Hub..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
fi

echo "Starting vLLM OpenAI-compatible server"
echo "  Model : ${MODEL}"
echo "  Port  : 8080"
[[ -n "${VLLM_ARGS:-}" ]] && echo "  Extra : ${VLLM_ARGS}"

# shellcheck disable=SC2086
exec vllm serve "${MODEL}" \
    --port 8080 \
    --host 0.0.0.0 \
    ${VLLM_ARGS:-} \
    "$@"

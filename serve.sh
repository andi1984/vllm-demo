#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server on CPU.
# Usage: ./serve.sh [model] [extra vllm args...]
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
shift || true

VENV=.venv
export LD_PRELOAD="$PWD/$VENV/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:$PWD/$VENV/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-2}"   # GiB for KV cache

exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  "$@"
# --enforce-eager: skip torch.compile (took >8 min on Zen 2 CPU)
# --enable-auto-tool-choice + hermes parser: OpenAI-style tool_calls for Qwen2.5

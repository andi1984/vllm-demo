#!/usr/bin/env bash
# Create a Python 3.12 virtual environment and install vLLM's x86 CPU build.
# Requires: uv (https://docs.astral.sh/uv/), Linux x86_64 with AVX2 or AVX-512.
set -euo pipefail
cd "$(dirname "$0")"

VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_39_x86_64.whl"

uv venv --python 3.12 --seed --managed-python .venv
source .venv/bin/activate

# vLLM CPU wheel (pulls in CPU-only PyTorch, FastAPI, httpx, openai, ...)
uv pip install "$WHEEL" --torch-backend cpu

python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
echo "Done. Start the model server with ./serve.sh"

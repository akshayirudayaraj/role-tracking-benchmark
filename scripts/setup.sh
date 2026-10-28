#!/usr/bin/env bash
# One-time environment setup (local or GPU pod). Needs Python >=3.10.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv   # reuse a preinstalled CUDA torch (e.g. RunPod images)
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
# HF cache on the pod's big volume if present
[ -d /workspace ] && echo 'export HF_HOME=/workspace/hf' >> .venv/bin/activate
echo "ok: $(.venv/bin/python -c 'import torch,transformers;print(torch.__version__, transformers.__version__, torch.cuda.is_available())')"

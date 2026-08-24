#!/usr/bin/env bash
# Setup on a vast.ai RTX 5090 instance (use a recent PyTorch CUDA template).
# The 5090 (Blackwell, sm_120) needs a cu128+ torch build — recent templates
# ship one; the pip line below upgrades if not.
set -euo pipefail

pip install -U --index-url https://download.pytorch.org/whl/cu128 torch
pip install "transformers>=5.0,<6" "datasets>=3.0" pytest

# Qwen/Qwen3-0.6B-Base is ungated — no HF token needed.
python - <<'EOF'
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

# Sanity: CPU unit tests (~seconds, no model download needed)
python -m pytest tests -q

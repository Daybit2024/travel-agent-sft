#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/Agent+SFT

python3 -m pip install -q "transformers==4.57.6" "peft==0.20.0" accelerate safetensors modelscope

python3 - <<'PY'
import os
from modelscope import snapshot_download
jobs = [
    ("Qwen/Qwen3-0.6B", "/root/autodl-tmp/Agent+SFT/qwen3-0_6b"),
    ("Qwen/Qwen3-4B-Instruct-2507", "/root/autodl-tmp/Agent+SFT/qwen3-4b-instruct-2507"),
]
for mid, dst in jobs:
    if os.path.isdir(dst) and os.listdir(dst):
        print("skip (exists):", dst, flush=True)
        continue
    print("downloading", mid, "->", dst, flush=True)
    snapshot_download(mid, local_dir=dst)
    print("done", dst, flush=True)
PY

python3 - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
print("transformers", transformers.__version__, "| peft", peft.__version__)
PY

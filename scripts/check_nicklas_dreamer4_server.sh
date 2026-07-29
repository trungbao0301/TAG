#!/usr/bin/env bash
set -euo pipefail

DREAMER4_ROOT="${DREAMER4_ROOT:-$HOME/dreamer4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export CUDA_VISIBLE_DEVICES

cd "$DREAMER4_ROOT"

/home/tbt589/bin/micromamba run -n dreamer4 python - <<'PY'
import importlib
import torch

mods = [
    "torch",
    "torchvision",
    "tensordict",
    "torchrl",
    "transformers",
    "wandb",
    "lpips",
]
for name in mods:
    importlib.import_module(name)
    print(f"{name}: OK")

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
print("devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

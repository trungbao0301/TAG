#!/usr/bin/env bash
set -euo pipefail

cd "${TAG_ROOT:-$HOME/tag}"
set +u
. install/setup.bash
set -u

python3 - <<'PY'
import importlib.util

for name in ("dreamer4", "dreamerv4", "dreamerv3", "jax", "rclpy"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'OK' if spec else 'MISSING'}")
PY

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

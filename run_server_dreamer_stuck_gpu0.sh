#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/tbt589/cyberruner-main}"
LOGDIR="${CYBERRUNNER_LOGDIR:-/home/tbt589/cyberrunner_logs/maprawv2_20260728}"
PYTHON_BIN="${CYBERRUNNER_PYTHON:-/home/tbt589/micromamba/envs/cyberrunner_ros/bin/python3}"

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export AMENT_PREFIX_PATH="$PROJECT_ROOT/install/cyberrunner_dreamer_thomas:$PROJECT_ROOT/install/cyberrunner_dreamer:$PROJECT_ROOT/install/cyberrunner_interfaces"
export LD_LIBRARY_PATH="$PROJECT_ROOT/install/cyberrunner_interfaces/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT/build/cyberrunner_dreamer_thomas:$PROJECT_ROOT/install/cyberrunner_dreamer_thomas/lib/python3.11/site-packages:$PROJECT_ROOT/build/cyberrunner_dreamer:$PROJECT_ROOT/install/cyberrunner_dreamer/lib/python3.11/site-packages:$PROJECT_ROOT/install/cyberrunner_interfaces/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"

export CYBERRUNNER_BALL_LOSS_GRACE_SEC=0
export CYBERRUNNER_OCCLUSION_GRACE_SEC=0
export CYBERRUNNER_OCCLUSION_XY_ZONES=""
export CYBERRUNNER_OCCLUSION_CHECKPOINT_RANGES=""
unset CYBERRUNNER_OCCLUSION_ZONES_FILE

export CYBERRUNNER_REWARD_ON_FAIL=-0.20
export CYBERRUNNER_TIMEOUT_STEPS=3000
export CYBERRUNNER_TIMEOUT_PENALTY=-0.20
export CYBERRUNNER_ANTICHEAT_MAX_STEP_M=0.057
export CYBERRUNNER_ANTICHEAT_PENALTY=-0.50

export CYBERRUNNER_STUCK_WINDOW_SEC=5
export CYBERRUNNER_STUCK_RADIUS_M=0.003
export CYBERRUNNER_STUCK_PENALTY=0

exec "$PYTHON_BIN" -m dreamerv3.train \
  --configs cyberrunner large \
  --task gym_cyberrunner_dreamer:cyberrunner-ros-v0 \
  --logdir "$LOGDIR" \
  --replay_size 1e6 \
  --run.script train_top5 \
  --run.train_ratio 256 \
  --run.save_every 20 \
  --run.log_every 1 \
  --jax.policy_devices 0 \
  --jax.train_devices 0

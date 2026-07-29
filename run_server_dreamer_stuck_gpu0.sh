#!/usr/bin/env bash
set -eo pipefail

# Where the TAG workspace is checked out ON THE SERVER. Override as needed.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tag}"
LOGDIR="${TAG_LOGDIR:-$HOME/tag_logs/run1}"
# NOTE: this is a conda/micromamba ENVIRONMENT name on the server, not a ROS
# package -- it is deliberately not renamed to tag_*. Override with TAG_PYTHON.
PYTHON_BIN="${TAG_PYTHON:-/home/tbt589/micromamba/envs/cyberrunner_ros/bin/python3}"

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
# Only tag_dreamer is needed: the TCP gym env imports no ROS messages, just
# ament_index_python to find its share dir. So the server never has to build
# tag_interfaces (it has no ROS/colcon anyway).
export AMENT_PREFIX_PATH="$PROJECT_ROOT/install/tag_dreamer"
# $PROJECT_ROOT/dreamerv3 MUST come first: dreamerv3 is pip-installed editable
# on this server pointing at the old cyberruner-main checkout, so without this
# shadow "-m dreamerv3.train --configs tag" loads that copy and fails with an
# unknown config (it only has a "cyberrunner:" profile).
export PYTHONPATH="$PROJECT_ROOT/dreamerv3:$PROJECT_ROOT/install/tag_dreamer/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"

export TAG_BALL_LOSS_GRACE_SEC=0
export TAG_OCCLUSION_GRACE_SEC=0
export TAG_OCCLUSION_XY_ZONES=""
export TAG_OCCLUSION_CHECKPOINT_RANGES=""
unset TAG_OCCLUSION_ZONES_FILE

export TAG_REWARD_ON_FAIL=-0.20
export TAG_TIMEOUT_STEPS=3000
export TAG_TIMEOUT_PENALTY=-0.20
export TAG_ANTICHEAT_MAX_STEP_M=0.057
export TAG_ANTICHEAT_PENALTY=-0.50

export TAG_STUCK_WINDOW_SEC=5
export TAG_STUCK_RADIUS_M=0.003
export TAG_STUCK_PENALTY=0

exec "$PYTHON_BIN" -m dreamerv3.train \
  --configs tag large \
  --task gym_tag_dreamer:tag-ros-v0 \
  --logdir "$LOGDIR" \
  --replay_size 1e6 \
  --run.script train_top5 \
  --run.train_ratio 256 \
  --run.save_every 20 \
  --run.log_every 1 \
  --jax.policy_devices 0 \
  --jax.train_devices 0

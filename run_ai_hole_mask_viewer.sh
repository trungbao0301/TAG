#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PREFIX="$PROJECT_ROOT/install/cyberrunner_state_estimation"
PYTHON_PACKAGE="$PROJECT_ROOT/build/cyberrunner_state_estimation"
VIEWER="$PACKAGE_PREFIX/lib/cyberrunner_state_estimation/ai_hole_viewer"
MODEL_PATH="${1:-$PROJECT_ROOT/models/marble_detector.onnx}"

source /opt/ros/humble/setup.bash
set -u
export AMENT_PREFIX_PATH="$PACKAGE_PREFIX${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$PYTHON_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"

exec "$VIEWER" --ros-args \
  -p "model_path:=$MODEL_PATH" \
  -p confidence_threshold:=0.90 \
  -p hole_margin_m:=0.0025 \
  -p hole_rejection_delay_sec:=2.0 \
  -p process_every_n:=3 \
  -p roi_x_min:=0.25 \
  -p roi_y_min:=0.15 \
  -p roi_x_max:=0.72 \
  -p roi_y_max:=0.80

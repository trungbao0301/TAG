#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PACKAGE_PREFIX="$PROJECT_ROOT/install/cyberrunner_state_estimation"
LOCAL_PYTHON_PACKAGE="$PROJECT_ROOT/build/cyberrunner_state_estimation"
DETECTOR="$LOCAL_PACKAGE_PREFIX/lib/cyberrunner_state_estimation/ai_detector"
MODEL_PATH="${1:-$PROJECT_ROOT/models/marble_detector.onnx}"

source /opt/ros/humble/setup.bash
set -u

# Prefer this checkout over the older package in /home/trungbao/cyberrunner_ws.
export AMENT_PREFIX_PATH="$LOCAL_PACKAGE_PREFIX${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$LOCAL_PYTHON_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$DETECTOR" ]]; then
  echo "AI detector is not built. Build cyberrunner_state_estimation first." >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "AI model does not exist: $MODEL_PATH" >&2
  exit 1
fi

echo "Model: $MODEL_PATH"
echo "Diagnostic topics: /cyberrunner_ai_marble/pixel and /confidence"
echo "Press Q in the detector window to quit."
exec "$DETECTOR" --ros-args \
  -p "model_path:=$MODEL_PATH" \
  -p confidence_threshold:=0.90 \
  -p miss_grace_frames:=90 \
  -p roi_x_min:=0.25 \
  -p roi_y_min:=0.15 \
  -p roi_x_max:=0.72 \
  -p roi_y_max:=0.80 \
  -p hole_rejection_enabled:=true \
  -p hole_rejection_margin_m:=0.0025 \
  -p hole_rejection_delay_sec:=2.0 \
  -p show_image:=true \
  -p publish_diagnostics:=true

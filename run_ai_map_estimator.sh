#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-$PROJECT_ROOT/models/marble_detector_v5_moredata.onnx}"

unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH ROS_PACKAGE_PATH CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
source "$PROJECT_ROOT/install/setup.bash"

cd "$PROJECT_ROOT"
exec ros2 run cyberrunner_state_estimation estimator_ai_map --ros-args \
  -p "ai_model_path:=$MODEL_PATH" \
  -p camera_height_m:=0.29 \
  -p marker_plane_height_m:=0.0 \
  -p marble_radius_m:=0.006 \
  -p hole_rejection_delay_sec:=0.0 \
  -p publish_legacy_topics:=false \
  "${@:2}"

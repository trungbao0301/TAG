#!/usr/bin/env bash
# Run the CyberRunner state estimator.
#
# This is THE estimator: it publishes on /cyberrunner_state_estimation/* and
# replaces the retired HSV/ocam pipeline (estimator/estimator_sub, removed).
#
# Deliberately passes almost nothing: the node's own defaults are the calibrated
# values (see cyberrunner_state_estimation/AI_MAP_ESTIMATOR.md). An earlier
# version of this script hardcoded marker_plane_height_m:=0.0 and
# publish_legacy_topics:=false, which silently overrode them.
#
# Usage:
#   ./run_ai_map_estimator.sh                       # defaults
#   ./run_ai_map_estimator.sh -p show_image:=true   # extra params pass through
#   MODEL=models/other.onnx ./run_ai_map_estimator.sh
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Wipe any inherited overlay so ~/cyberrunner_ws cannot win AMENT_PREFIX_PATH.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH ROS_PACKAGE_PATH CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
source "$PROJECT_ROOT/install/setup.bash"

cd "$PROJECT_ROOT"

ARGS=()
if [ -n "${MODEL:-}" ]; then
  ARGS+=(-p "ai_model_path:=$PROJECT_ROOT/${MODEL#"$PROJECT_ROOT/"}")
fi

exec ros2 run cyberrunner_state_estimation estimator_ai_map --ros-args \
  "${ARGS[@]}" "$@"

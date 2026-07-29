#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PREFIX="$PROJECT_ROOT/install/cyberrunner_state_estimation"
PYTHON_PACKAGE="$PROJECT_ROOT/build/cyberrunner_state_estimation"
SELECTOR="$PACKAGE_PREFIX/lib/cyberrunner_state_estimation/pendulum_zone_selector"
OUTPUT_PATH="${1:-$PROJECT_ROOT/pendulum_occlusion_zones.json}"

source /opt/ros/humble/setup.bash
set -u
export AMENT_PREFIX_PATH="$PACKAGE_PREFIX${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$PYTHON_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"

exec "$SELECTOR" --ros-args \
  -p camera_topic:=/cyberrunner_camera/image \
  -p "output_path:=$OUTPUT_PATH"

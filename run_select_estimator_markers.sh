#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTOR="$PROJECT_ROOT/install/tag_state_estimation/lib/tag_state_estimation/select_markers"
OUTPUT="$PROJECT_ROOT/tag_state_estimation/markers.csv"

source /opt/ros/humble/setup.bash

if [[ ! -x "$SELECTOR" ]]; then
  echo "Marker selector is not built. Run:" >&2
  echo "  colcon build --packages-select tag_state_estimation --symlink-install" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$SELECTOR" --ros-args -p "output_path:=$OUTPUT"

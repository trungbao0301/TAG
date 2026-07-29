#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PACKAGE_PREFIX="$PROJECT_ROOT/install/cyberrunner_state_estimation"
LOCAL_PYTHON_PACKAGE="$PROJECT_ROOT/build/cyberrunner_state_estimation"
LABELER="$LOCAL_PACKAGE_PREFIX/lib/cyberrunner_state_estimation/ai_labeler"
DATASET_DIR="${1:-$PROJECT_ROOT/ai_marble_dataset}"

source /opt/ros/humble/setup.bash
set -u

# The machine also has /home/trungbao/cyberrunner_ws, which contains an older
# package with the same name. Explicitly put this checkout first.
export AMENT_PREFIX_PATH="$LOCAL_PACKAGE_PREFIX${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$LOCAL_PYTHON_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$LABELER" ]]; then
  echo "AI labeler is not built. Run:" >&2
  echo "  cd $PROJECT_ROOT" >&2
  echo "  source /opt/ros/humble/setup.bash" >&2
  echo "  colcon build --symlink-install --packages-select cyberrunner_state_estimation" >&2
  exit 1
fi

echo "Dataset: $DATASET_DIR"
echo "Controls: left-click marble, N=not visible, SPACE=freeze, Q=quit"
exec "$LABELER" --ros-args -p "output_dir:=$DATASET_DIR"

#!/usr/bin/env bash
# Launch the tag state estimator from THIS workspace
# (~/CYBER/cyberruner-main), bypassing the ~/.bashrc overlay that otherwise
# forces ~/tag_ws to take precedence on AMENT_PREFIX_PATH.
#
# Usage:  ./run_estimator_cyber.sh   (add any `ros2 run` args after)

CY=/home/trungbao/CYBER/tag

# Wipe any inherited ROS overlay so tag_ws cannot win.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH ROS_PACKAGE_PATH CMAKE_PREFIX_PATH

source /opt/ros/humble/setup.bash
source "$CY/install/setup.bash"

# Keep the same DDS domain as the camera/bridge (default 0).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

echo "estimator_sub resolves to:"
echo "  AMENT head: $(echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | grep tag | head -1)"

cd "$CY"
exec ros2 run tag_state_estimation estimator_sub "$@"

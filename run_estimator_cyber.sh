#!/usr/bin/env bash
# Launch the cyberrunner state estimator from THIS workspace
# (~/CYBER/cyberruner-main), bypassing the ~/.bashrc overlay that otherwise
# forces ~/cyberrunner_ws to take precedence on AMENT_PREFIX_PATH.
#
# Usage:  ./run_estimator_cyber.sh   (add any `ros2 run` args after)

CY=/home/trungbao/CYBER/cyberruner-main

# Wipe any inherited ROS overlay so cyberrunner_ws cannot win.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH ROS_PACKAGE_PATH CMAKE_PREFIX_PATH

source /opt/ros/humble/setup.bash
source "$CY/install/setup.bash"

# Keep the same DDS domain as the camera/bridge (default 0).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

echo "estimator_ai_map resolves to:"
echo "  AMENT head: $(echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | grep cyberrunner | head -1)"

cd "$CY"
exec ros2 run cyberrunner_state_estimation estimator_ai_map "$@"

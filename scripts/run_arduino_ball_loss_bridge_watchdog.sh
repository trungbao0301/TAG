#!/usr/bin/env bash
set -e

unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH ROS_PACKAGE_PATH CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
source /home/trungbao/CYBER/tag/install/setup.bash
cd /home/trungbao/CYBER/tag

exec python3 -u scripts/arduino_ball_loss_bridge.py \
  --port auto \
  --serial_reply_timeout 0.25 \
  --max_missing_replies 3 \
  --status_every 1.0 \
  --monitor_only

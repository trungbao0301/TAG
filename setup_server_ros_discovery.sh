#!/usr/bin/env bash
# Server-side setup for direct ROS 2 over Fast DDS Discovery Server.
# Use this for Thomas Dreamer, which reads ROS topics directly instead of TCP.

export PATH=/home/tbt589/micromamba/envs/tag_ros/bin:$PATH

if [ -f /home/tbt589/micromamba/envs/tag_ros/setup.bash ]; then
  source /home/tbt589/micromamba/envs/tag_ros/setup.bash
fi

if [ -f /home/tbt589/cyberruner-main/install/setup.bash ]; then
  source /home/tbt589/cyberruner-main/install/setup.bash
fi

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER=10.157.146.38:11811
unset ROS_SUPER_CLIENT

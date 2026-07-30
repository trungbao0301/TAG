#!/usr/bin/env bash
# NOTE: cyberrunner_ros is a micromamba ENVIRONMENT on the server, not a package
# in this repo. A cyberrunner_ -> tag_ rename broke it once; do not rename it.
# Server-side setup for direct ROS 2 over Fast DDS Discovery Server.
# Use this for Thomas Dreamer, which reads ROS topics directly instead of TCP.

export PATH=${TAG_CONDA_ENV:-/home/tbt589/micromamba/envs/cyberrunner_ros}/bin:$PATH

if [ -f ${TAG_CONDA_ENV:-/home/tbt589/micromamba/envs/cyberrunner_ros}/setup.bash ]; then
  source ${TAG_CONDA_ENV:-/home/tbt589/micromamba/envs/cyberrunner_ros}/setup.bash
fi

if [ -f /home/tbt589/cyberruner-main/install/setup.bash ]; then
  source /home/tbt589/cyberruner-main/install/setup.bash
fi

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER=10.157.146.38:11811
unset ROS_SUPER_CLIENT

#!/usr/bin/env bash
# NOTE: cyberrunner_ros is a micromamba ENVIRONMENT on the server, not a package
# in this repo. A cyberrunner_ -> tag_ rename broke it once; do not rename it.
# Source this on the server before using direct ROS 2 Dreamer training.
# It enables the micromamba ROS environment, then overlays this workspace.

export PATH=${TAG_CONDA_ENV:-/home/tbt589/micromamba/envs/cyberrunner_ros}/bin:$PATH

if [ -f ${TAG_CONDA_ENV:-/home/tbt589/micromamba/envs/cyberrunner_ros}/setup.bash ]; then
  source ${TAG_CONDA_ENV:-/home/tbt589/micromamba/envs/cyberrunner_ros}/setup.bash
fi

if [ -f /home/tbt589/cyberruner-main/install/setup.bash ]; then
  source /home/tbt589/cyberruner-main/install/setup.bash
fi

unset ROS_DISCOVERY_SERVER
unset ROS_SUPER_CLIENT
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

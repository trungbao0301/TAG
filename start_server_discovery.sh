#!/usr/bin/env bash
# Run this in a dedicated server terminal and leave it running.

source /home/tbt589/cyberruner-main/setup_server_ros_discovery.sh
exec fastdds discovery -i 0 -l 10.157.146.38 -p 11811

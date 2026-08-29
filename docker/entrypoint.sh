#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/${ROS_DISTRO}/setup.bash

# Execute the command passed to docker run
exec "$@" 
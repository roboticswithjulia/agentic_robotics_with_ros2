#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/${ROS_DISTRO}/setup.bash

# Start the Ollama server in the background (workshop section 2.2) unless
# something already serves on 11434 - with network_mode: host that could be
# an Ollama running on the host or in another container.
if ! curl -sf --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    ollama serve >/var/log/ollama.log 2>&1 &
fi

# Execute the command passed to docker run
exec "$@"

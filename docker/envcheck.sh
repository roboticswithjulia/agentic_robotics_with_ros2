#!/usr/bin/env bash
# Workshop environment check. Safe to paste, source, or run directly.
# Works on the host and inside the container.

echo "--- system ---"
. /etc/os-release; echo "$PRETTY_NAME"
uname -m
echo "cores: $(nproc)"
free -h | awk 'NR==2{print "ram: "$2}'
df -h ~ | awk 'NR==2{print "free disk: "$4}'
grep -qi microsoft /proc/version && echo "running under WSL2"

echo
echo "--- ros 2 ---"
# Source ROS if it is installed but not yet on PATH (e.g. `docker compose exec`).
if ! command -v ros2 >/dev/null 2>&1 && [ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]; then
  . "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
fi
echo "ROS_DISTRO=${ROS_DISTRO:-NOT SOURCED}"
if command -v ros2 >/dev/null 2>&1; then
  for pkg in ur_description robot_state_publisher joint_state_publisher_gui rviz2 xacro tf2_ros; do
    if ros2 pkg prefix "$pkg" >/dev/null 2>&1; then echo "  OK      $pkg"; else echo "  MISSING $pkg"; fi
  done
else
  echo "  ros2 not found - are you on the host instead of in the container?"
fi

echo
echo "--- python ---"
python3 --version
which python3
echo "virtualenv: ${VIRTUAL_ENV:-none}"
for m in rclpy numpy ikpy; do
  if python3 -c "import $m" >/dev/null 2>&1; then echo "  OK      $m"; else echo "  MISSING $m"; fi
done

# This is an auto generated Dockerfile for ros:desktop
# generated from docker_images_ros2/create_ros_image.Dockerfile.em
FROM ros:jazzy-ros-base-noble

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV LANG=en_US.UTF-8

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    && rm -rf /var/lib/apt/lists/*

# Minimal locale and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip  && rm -rf /var/lib/apt/lists/*

# Install extra ROS packages used in the workshop (base image already provides core)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ur-description \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (numpy constraint) and ikpy
# `--break-system-packages` allows pip to modify system-installed packages when necessary
RUN pip3 install --break-system-packages --no-cache-dir 'numpy<2.0.0' ikpy

# Source ROS in interactive shells
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

CMD ["bash"]

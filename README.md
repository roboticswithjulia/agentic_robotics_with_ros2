# Agentic ROS Jazzy Docker

This README explains how to build and run the minimal ROS 2 Jazzy Docker Compose setup included in this repository.

Prerequisites

- Docker (20.10+)
- Docker Compose (v2 `docker compose` or legacy `docker-compose`)

Quick start (recommended)

1. Change to the directory containing the compose file:

```bash
cd src/docker
```

2. Build the image (shows detailed output):

```bash
docker compose build --progress=plain
```

3. Bring the service up (foreground):

```bash
docker compose up
```

Or run in the background:

```bash
docker compose up -d
```

Notes on GUI (RViz) and workspace mounts

The provided `docker-compose.yml` was simplified for a minimal Jazzy image and does not mount the host workspace or enable X11 forwarding by default. If you need GUI access (for `rviz2`) or to mount your workspace into the container, re-enable the commented volume lines in `docker-compose.yml`:

- Mount X11 socket and `.Xauthority` to enable GUI:
  - `/tmp/.X11-unix:/tmp/.X11-unix`
  - `${HOME}/.Xauthority:/root/.Xauthority:rw`
- Mount your workspace for development:
  - `..:/ros2_ws/src`
- Set `DISPLAY` in the environment section (e.g. `DISPLAY=${DISPLAY}`)
- If hardware access is required, uncomment `privileged: true` and `devices:` entries.

If you re-enable GUI forwarding, allow local X11 connections with:

```bash
xhost +local:docker
```

Verifying ROS

Once the container is running you can open a shell into it and verify a basic ROS 2 setup:

```bash
# open an interactive shell (service name: unitree_ros)
docker compose exec unitree_ros bash

# inside the container
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_cpp talker &
# in another shell/container
ros2 run demo_nodes_py listener
```

Cleaning up

```bash
# stop and remove containers
docker compose down
```

Troubleshooting

- If the image build fails because a `ros-jazzy-*` package is not available, try switching the base image to `ros:jazzy` or `ros:jazzy-ros-core` and retry, or ask me to change the Dockerfile for you.

If you'd like, I can build the image here and report errors."

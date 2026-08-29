# ROS 2 Jazzy + RViz desktop + Ollama with the two workshop models,
# for the "Agentic Robotics with ROS 2" workshop.
FROM ros:jazzy-ros-base-noble

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV LANG=en_US.UTF-8

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    && rm -rf /var/lib/apt/lists/*

# Minimal tools: curl + zstd for the Ollama download, unzip for the workshop zip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    curl \
    zstd \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# --- Ollama (workshop section 2.1) ---
# Installed from the release tarball rather than install.sh: the script wants
# systemd and probes for a GPU, neither of which applies inside a container.
ARG OLLAMA_VERSION=0.33.2
RUN curl -fsSL "https://ollama.com/download/ollama-linux-amd64.tar.zst?version=${OLLAMA_VERSION}" \
    | zstd -d | tar -x -C /usr/local

# --- CPU pin (workshop section 3) ---
# No systemd in a container, so the service drop-in becomes plain env vars.
# `ollama ps` must report 100% CPU, matching the room.
ENV CUDA_VISIBLE_DEVICES=""
ENV OLLAMA_LLM_LIBRARY=cpu

# --- The two workshop models (section 2.3), baked into the image (~3.7 GB) ---
# The server must be running to pull, so start it just for this layer.
RUN bash -ec '\
    ollama serve >/tmp/ollama-build.log 2>&1 & pid=$!; \
    for i in $(seq 1 120); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 0.5; done; \
    curl -sf http://127.0.0.1:11434/api/version >/dev/null; \
    ollama pull qwen3.5:4b; \
    ollama pull nomic-embed-text; \
    ollama list; \
    kill $pid; wait $pid || true'

# Install extra ROS packages used in the workshop (base image already provides core)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ur-description \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (numpy constraint) and ikpy
# `--break-system-packages` allows pip to modify system-installed packages when necessary
RUN pip3 install --break-system-packages --no-cache-dir 'numpy<2.0.0' ikpy

# Source ROS in every interactive shell, whichever user runs it
# (/root/.bashrc is unreadable when the container runs as a non-root user)
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

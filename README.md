# Agentic Robotics with ROS 2 — Docker setup

A ROS 2 **Jazzy** container for the "Agentic Robotics with ROS 2" workshop
(natural-language robot control). The image comes with everything baked in:

- ROS 2 Jazzy desktop (RViz2, rqt) plus the workshop packages
  (`ur_description`, `joint_state_publisher_gui`, `xacro`)
- **Ollama 0.33.2** with both workshop models already pulled:
  `qwen3.5:4b` (planner) and `nomic-embed-text` (grounding) — no downloads
  needed at run time
- Inference pinned to **CPU** (workshop section 3), so `ollama ps` reads
  `100% CPU`
- `ikpy`, `numpy<2`, `colcon`, `unzip`, `zstd`

The Ollama server starts automatically with the container, and every shell
opened in the container has ROS pre-sourced. The workshop PDF's Humble paths
translate directly: wherever it says `/opt/ros/humble`, this container uses
`/opt/ros/jazzy` — and its "two lines at the top of every terminal" are
already done for you here.

## Demo

https://github.com/roboticswithjulia/agentic_robotics_with_ros2/raw/main/media/output.mp4

The robot arm executing a natural-language instruction end to end:
[media/output.mp4](media/output.mp4).

## 1. Build the image and start the container

```bash
cd docker
docker compose build          # ~17 GB image; the model layer alone is 3.7 GB
xhost +local:docker           # allow the container to open RViz on your display
docker compose up -d
```

Check it is alive:

```bash
docker compose ps                                  # service: agentic_ws_2
curl -s http://localhost:11434/api/version         # {"version":"0.33.2"}
docker compose exec agentic_ws_2 bash /agent_ws/src/docker/envcheck.sh
```

The env check should report every ROS package and Python module `OK`.

GPU (optional): the compose file ships with `gpus: all` commented out.
Uncomment it only after installing `nvidia-container-toolkit` on the host
(`sudo apt install nvidia-container-toolkit && sudo nvidia-ctk cdi generate
--output=/etc/cdi/nvidia.yaml && sudo systemctl restart docker`), otherwise
`up` fails with "no known GPU vendor found". The workshop does not need it.

## 2. Build the workshop package

Open a shell in the container — this is your **Terminal 1** for the day:

```bash
docker compose exec agentic_ws_2 bash
```

Inside (you land in `/agent_ws`, with this repository mounted at
`/agent_ws/src`):

```bash
colcon build --symlink-install
source install/setup.bash
```

Expected:

```
Starting >>> agentic_arm
Finished <<< agentic_arm
Summary: 1 package finished
```

`--symlink-install` links the Python files instead of copying them, so code
edits take effect when you restart a node — no rebuild. Launch files and the
URDF still need a rebuild.

Note: `build/` and `install/` live inside the container, not on your host.
If you recreate the container (`docker compose down && up -d`), rebuild —
it takes seconds.

## 3. Terminal 1 — launch everything

```bash
ros2 launch agentic_arm full.launch.py
```

RViz opens with a robot arm and coloured objects on a table. In the terminal
you should see, in roughly this order:

```
scene ready, 10 graspable objects in 4 colours and 3 shapes, seed 7
chain loaded, 6 DOF, tip = gripper-grasp
embedded 13 objects, 2 vectors each, with nomic-embed-text
world received: 13 objects, 10 of them graspable
planner ready, model=qwen3.5:4b
```

(13 = the 10 graspable objects plus three places: tray, shelf and the table
itself.)

**Leave this terminal running. Everything else happens elsewhere.**

## 4. Terminal 2 — talk to the robot

Open a second shell into the container; ROS is already sourced, so only the
workspace overlay is needed:

```bash
docker compose exec agentic_ws_2 bash
source install/setup.bash
```

Send your first instruction:

```bash
ros2 topic pub --once /instruction std_msgs/String \
  "{data: 'put the blue coke can on the shelf'}"
```

Watch Terminal 1. After a few seconds a plan appears, then the arm moves:

```
INSTRUCTION  "put the blue coke can on the shelf"
PLAN     4 tasks, 46 tokens, 3.2s
  1  navigate  blue cylinder      (you said "blue coke can")
  2  pick      blue cylinder      (you said "blue coke can")
  3  navigate  shelf
  4  place     blue cylinder  ->  shelf
```

Nothing is frozen while you wait: that is a four-billion-parameter model
thinking on your CPU. Ten to twenty seconds is normal; slower machines take
longer.

## 5. What is on the table

| colour | objects |
|--------|---------------------------------|
| red    | cylinder, small ball, large ball |
| green  | cuboid, small ball, large ball   |
| blue   | cylinder, cuboid                 |
| golden | cylinder, cuboid                 |

Plus **three** places to put things: the tray, the shelf, and the table
itself. Placing something "on the table" returns it to the spot where it
started the simulation, not to the table's centre.

## 6. Things worth trying

All of these work — change the text in quotes and send again:

```
'put the gold marble on the tray'
'grab the red flask and put it on the shelf'
'pick up the bigger green ball and place it on the tray'
'hey, could you move the maroon cylinder to the shelf please'
'put the sky coloured brick on the tray'
'go to the table'
'put the red cylinder on the tray, then put it back where it was'
'grab the red flask and put it on the shelf, then go back to home, and then pick the red flask and put it back on the table'
```

And these are refused, on purpose — read what it says:

```
'pick up the screwdriver'
'put the greenish thing on the shelf'
'move the blue ball to the tray'
```

Words like flask, marble and maroon appear nowhere in the code. Try your own
and see what it makes of them.

## 7. Phase 2 — the same system with a camera

Stop the launch in Terminal 1 with `Ctrl+C`, then start the other one:

```bash
ros2 launch agentic_arm perception.launch.py
```

Two new lines appear at startup:

```
synthetic camera 640x480, fx=531.4, 5 Hz
seeing 10 objects, plus 3 known fixtures
```

(3 fixtures = tray, shelf and the table.)

Now send exactly the same instruction as before. It behaves the same way.

The difference is where the world came from. In Phase 1 the robot was told
where everything is. Now a camera measured it, and every position is a few
millimetres off as a result.

### Seeing what the camera sees

In RViz, click **Add** at the bottom of the Displays panel, choose
**By topic**, then pick these two:

| topic | display | what appears |
|---|---|---|
| `/camera/color/image_raw` | Image | a panel showing what the camera renders |
| `/detected_markers` | MarkerArray | white outlines around everything perception believes it has found |

The outlines sit almost on top of the solid objects. Zoom in and you can see
the gap. That gap is what measurement costs.

### Proving which phase you are in

```bash
ros2 topic info /world_state
```

Phase 1 says the publisher is `scene_node`. Phase 2 says it is
`perception_node`. Same topic, same message, different source — and not one
line of the planner, executor or arm node changed between them.

## 8. Useful while it runs

```bash
ros2 node list
ros2 topic list
ros2 topic info /world_state
ros2 topic echo /task_plan
```

## 9. Cleaning up

```bash
docker compose down        # stop and remove the container
```

## Troubleshooting

| symptom | fix |
|---|---|
| `ros2: command not found` | you are on the host — `docker compose exec agentic_ws_2 bash` first |
| `Package 'agentic_arm' not found` | `source install/setup.bash` in that shell (and build once, section 2) |
| Cannot reach the Ollama server | `docker compose exec agentic_ws_2 bash -c 'ollama serve &'` — it normally autostarts |
| RViz window never appears | run `xhost +local:docker` on the host, then relaunch |
| edited a file and nothing changed | the running node has the old code — stop the launch and start it again |
| `no known GPU vendor found` on `up` | the `gpus:` line is uncommented but the host lacks nvidia-container-toolkit — see section 1 |

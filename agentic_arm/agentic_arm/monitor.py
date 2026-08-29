#!/usr/bin/env python3
"""
monitor.py -- a live view of what the system is doing.

Run it beside the launch terminal. It refreshes in place and answers, at a
glance:

  which phase is running, and how you can tell
  what is publishing the world, and how often
  whether a camera exists, and at what rate
  where that camera is, and what its intrinsics are
  how far perception's measurements are from the truth, in millimetres

Nothing here is required by the system. It is a window onto it, meant for
explaining what is happening rather than making it happen.

    ros2 run agentic_arm monitor
    python3 monitor.py --once      # print one snapshot and exit
"""

import json
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RED = "\033[31m"
OFF = "\033[0m"

REFRESH_S = 1.0
STALE_S = 5.0


def bar(label):
    return f"{BOLD}{label}{OFF}\n" + DIM + "-" * 66 + OFF


class Monitor(Node):
    def __init__(self, once=False):
        super().__init__("monitor")
        self.once = once

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)

        self.world = None
        self.world_at = None
        self.world_count = 0

        self.truth = None
        self.truth_at = None

        self.info = None
        self.rgb_at = None
        self.rgb_times = []
        self.depth_at = None
        self.image_shape = None

        self.create_subscription(String, "world_state", self.on_world, latched)
        self.create_subscription(String, "scene_truth", self.on_truth, latched)
        self.create_subscription(
            CameraInfo, "camera/color/camera_info", self.on_info, 1)
        self.create_subscription(
            Image, "camera/color/image_raw", self.on_rgb, 1)
        self.create_subscription(
            Image, "camera/depth/image_rect_raw", self.on_depth, 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.started = time.time()
        self.create_timer(REFRESH_S, self.draw)

    # -- subscriptions ---------------------------------------------------

    def on_world(self, msg):
        try:
            self.world = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.world_at = time.time()
        self.world_count += 1

    def on_truth(self, msg):
        try:
            self.truth = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.truth_at = time.time()

    def on_info(self, msg):
        self.info = msg

    def on_rgb(self, msg):
        now = time.time()
        self.rgb_at = now
        self.image_shape = (msg.width, msg.height, msg.encoding)
        self.rgb_times.append(now)
        self.rgb_times = [t for t in self.rgb_times if now - t < 5.0]

    def on_depth(self, msg):
        self.depth_at = time.time()

    # -- helpers ---------------------------------------------------------

    def rgb_hz(self):
        if len(self.rgb_times) < 2:
            return 0.0
        span = self.rgb_times[-1] - self.rgb_times[0]
        return (len(self.rgb_times) - 1) / span if span > 0 else 0.0

    def publishers_of(self, topic):
        return [name for name, _ in
                self.get_publishers_info_by_topic(topic)] \
            if hasattr(self, "get_publishers_info_by_topic") else []

    def world_publisher(self):
        try:
            infos = self.get_publishers_info_by_topic("/world_state")
        except Exception:
            return None
        return infos[0].node_name if infos else None

    def nodes(self):
        try:
            return sorted(n for n, _ in self.get_node_names_and_namespaces())
        except Exception:
            return []

    def camera_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                "world", "camera_optical_frame", rclpy.time.Time())
        except Exception:
            return None
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    def accuracy(self):
        """Compare what perception reports against the ground truth."""
        if not self.world or not self.truth:
            return None
        seen = {n: o for n, o in self.world.get("objects", {}).items()
                if o.get("graspable")}
        real = {n: o for n, o in self.truth.get("objects", {}).items()
                if o.get("graspable")}
        if not seen or not real:
            return None

        rows = []
        for name, obj in sorted(seen.items()):
            nearest, dist = None, 1e9
            for other, ref in real.items():
                d = math.dist(ref["xyz"], obj["xyz"])
                if d < dist:
                    nearest, dist = other, d
            rows.append((name, nearest, dist * 1000.0))
        return rows

    # -- drawing ---------------------------------------------------------

    def draw(self):
        now = time.time()
        nodes = self.nodes()
        has_camera = "camera_node" in nodes
        has_perception = "perception_node" in nodes
        phase2 = has_camera and has_perception
        pub = self.world_publisher()

        out = [] if self.once else [CLEAR]

        # -- phase --
        tag = (f"{BLUE}PHASE 2{OFF}  the world is measured by a camera"
               if phase2 else
               f"{GREEN}PHASE 1{OFF}  the world is a list of objects")
        out.append(bar("WHICH PHASE"))
        out.append(f"  {tag}")
        out.append(f"  /world_state published by   "
                   f"{BOLD}{pub or 'nobody'}{OFF}")
        out.append(f"  nodes running               {len(nodes)}")
        out.append("  " + ", ".join(nodes) if nodes else "")
        out.append("")

        # -- world --
        out.append(bar("THE WORLD"))
        if self.world is None:
            out.append(f"  {RED}nothing on /world_state{OFF}")
        else:
            objs = self.world.get("objects", {})
            grasp = sum(1 for o in objs.values() if o.get("graspable"))
            age = now - self.world_at
            stale = f"{YELLOW} (last update {age:.0f}s ago){OFF}" \
                if age > STALE_S else ""
            out.append(f"  objects                     {len(objs)}  "
                       f"({grasp} graspable){stale}")
            out.append(f"  updates received            {self.world_count}")
            out.append(f"  colours                     "
                       f"{', '.join(self.world.get('colors', []))}")
            out.append(f"  shapes                      "
                       f"{', '.join(self.world.get('shapes', []))}")
        out.append("")

        # -- camera --
        out.append(bar("THE CAMERA"))
        if not has_camera:
            out.append(f"  {DIM}no camera in this phase{OFF}")
        else:
            hz = self.rgb_hz()
            colour = GREEN if hz > 0.5 else RED
            out.append(f"  colour images               "
                       f"{colour}{hz:.1f} Hz{OFF}")
            if self.image_shape:
                w, h, enc = self.image_shape
                out.append(f"  resolution                  {w} x {h}  {enc}")
            if self.depth_at:
                out.append(f"  depth images                "
                           f"{GREEN}arriving{OFF}")
            else:
                out.append(f"  depth images                {RED}none{OFF}")
            if self.info:
                k = self.info.k
                out.append(f"  intrinsics  fx {k[0]:7.1f}   fy {k[4]:7.1f}")
                out.append(f"              cx {k[2]:7.1f}   cy {k[5]:7.1f}")
            pose = self.camera_pose()
            if pose:
                out.append(f"  position in world           "
                           f"x {pose[0]:+.3f}  y {pose[1]:+.3f}  "
                           f"z {pose[2]:+.3f}")
            else:
                out.append(f"  position in world           {RED}TF missing{OFF}")
        out.append("")

        # -- accuracy --
        rows = self.accuracy()
        out.append(bar("MEASURED AGAINST TRUTH"))
        if rows is None:
            if not phase2:
                out.append(f"  {DIM}Phase 1 is exact by construction: the "
                           f"numbers were never measured{OFF}")
            else:
                out.append(f"  {DIM}waiting for both /world_state and "
                           f"/scene_truth{OFF}")
        else:
            errs = [d for _, _, d in rows]
            out.append(f"  {len(rows)} objects   "
                       f"mean {sum(errs) / len(errs):.1f} mm   "
                       f"worst {max(errs):.1f} mm")
            out.append("")
            for name, nearest, d in rows[:12]:
                mark = " " if d < 25 else "!"
                match = "" if name == nearest else f"  = {nearest}"
                out.append(f"  {mark} {name:<20} {d:6.1f} mm{match}")
            if len(rows) > 12:
                out.append(f"    {DIM}... and {len(rows) - 12} more{OFF}")

        if not self.once:
            out.append("")
            out.append(f"{DIM}refreshing every {REFRESH_S:.0f}s, "
                       f"Ctrl-C to stop{OFF}")

        print("\n".join(out), flush=True)

        if self.once:
            raise SystemExit(0)


def main():
    once = "--once" in sys.argv
    rclpy.init()
    node = Monitor(once=once)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

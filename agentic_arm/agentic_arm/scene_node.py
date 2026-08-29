#!/usr/bin/env python3
"""
scene_node -- the source of truth about what is on the table.

Decice what is on the table, where it is, and what it looks like. Publish that
information to the executor, and update it when the executor tells us something changed.

WHY THIS IS A SEPARATE NODE

Nothing downstream should know where the world comes from. The executor asks
"what is on the table" and gets an answer; whether that answer was made up by
this node or measured by a camera is not its business.

That separation is the whole point of Phase 2. Swapping this node for a
perception node changes the source of the world and nothing else. No line of
the executor, planner or arm node changes.

    scene_node        --/world_state-->  executor_node
    perception_node   --/world_state-->  executor_node    (Phase 2)

PUBLISHES
    /world_state     std_msgs/String, latched, JSON
                     every object with its position, colour, shape, size,
                     dimensions, volume and whether it can be picked up
    /scene_markers   visualization_msgs/MarkerArray
                     the coloured blocks and their labels in RViz

SUBSCRIBES
    /scene_command   std_msgs/String, JSON
                     the executor telling us the world changed: something was
                     picked up, or something was put down.

That last topic deserves a note. A made up world has to be TOLD that an
object moved. A camera does not: it simply sees the object in its new place
on the next frame. So the perception node in Phase 2 ignores this topic
entirely, and that difference is worth pointing at.
"""

import json
import math
import random

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

CARRY_OFFSET = -0.03     # a carried object hangs just below the grasp point
GRIPPER_FRAME = "grasp_point"
WORLD_FRAME = "world"

COLORS = ["red", "green", "blue", "golden"]
SHAPES = ["cuboid", "cylinder", "sphere"]

# What is actually on the table, colour by colour.
#
# Red and green carry three objects each, including both ball sizes, so
# "the bigger red ball" has something to mean. Blue and golden carry two.
# Ten objects in total, which leaves the table readable rather than crowded.
SCENE_PLAN = {
    "red":    [("cylinder", None), ("sphere", "small"), ("sphere", "large")],
    "green":  [("cuboid", None), ("sphere", "small"), ("sphere", "large")],
    "blue":   [("cylinder", None), ("cuboid", None)],
    "golden": [("cylinder", None), ("cuboid", None)],
}

RGB = {
    "red":    (0.85, 0.15, 0.15),
    "green":  (0.15, 0.70, 0.25),
    "blue":   (0.15, 0.30, 0.85),
    "golden": (0.85, 0.68, 0.13),
}

# Physical size per shape, in metres. (x, y, z)
# Four objects per colour: one cylinder, one cuboid, and two balls.
#
# Cubes are gone. From directly above a cube and a cuboid are both filled
# rectangles, and separating them rested entirely on how elongated the blob
# looked, which perspective distorts near the edges of the image. A cube on
# the shelf read as elongated 1.71 and was renamed mid demo. Cuboids,
# cylinders and spheres separate on signals that do not move.
#
# The size pair moved from cubes to the balls, so "the bigger red ball"
# means something. Doubling the diameter also doubles the sphere's depth
# range, from about 3 cm to about 6 cm, which is why the classifier
# thresholds are re-measured below rather than carried over.
DIMS = {
    ("cuboid", None):      (0.12, 0.055, 0.055),
    ("cylinder", None):    (0.05, 0.05, 0.11),
    ("sphere", "small"):   (0.06, 0.06, 0.06),
    ("sphere", "large"):   (0.12, 0.12, 0.12),
}

MARKER_TYPE = {
    "cuboid": Marker.CUBE,
    "cylinder": Marker.CYLINDER,
    "sphere": Marker.SPHERE,
}

# ===========================================================================
# THE WORLD
# Objects are scattered at random rather than placed on a grid, so a query
# like "the red cylinder" has to be resolved by attribute rather than by
# position. Placement is seeded, so every machine in the room sees the same
# scatter and a live demo stays reproducible.
#
# The sampling box below was swept against the UR5e with the IK solver:
# every point in it is reachable at both grasp and approach height.
# ===========================================================================

PLACEMENT_SEED = 7

# Sampling area. Every point in this box was swept against the UR5e with the
# IK solver and is reachable at both grasp and approach height.
X_RANGE = (0.30, 0.74)
Y_RANGE = (-0.40, 0.40)

# More cells than objects, so the empty ones break up any visible regularity.
GRID_COLS = 3      # along x, cell 0.147 m
GRID_ROWS = 5      # along y, cell 0.160 m
JITTER = 0.9       # fraction of the free space in a cell that may be used

# Fixed, because the arm must always know where to put things down.
DESTINATIONS = {
    "tray":  {"xyz": [0.30, -0.52, 0.02], "color": "grey", "shape": "tray",
              "dims": (0.22, 0.16, 0.03)},
    "shelf": {"xyz": [0.28, 0.52, 0.28], "color": "brown", "shape": "shelf",
              "dims": (0.20, 0.16, 0.03)},
    # The tabletop itself, matching the URDF's table_link exactly: a
    # 0.90 x 1.20 slab whose top surface is the z = 0 plane. Naming it
    # makes "go to the table" a navigate target and "put it on the table"
    # a place destination, with no new machinery anywhere else.
    "table": {"xyz": [0.55, 0.0, -0.01], "color": "brown", "shape": "table",
              "dims": (0.90, 1.20, 0.02)},
}


def _scatter(specs, rng):
    """Place objects on a sparse jittered grid.

    Pure rejection sampling was tried first and is not reliable at this
    density: twenty objects in a reachable box of about 0.4 square metres
    fails to converge more often than it succeeds. A grid with more cells
    than objects, randomly chosen and then jittered inside each cell, is
    guaranteed to terminate, never overlaps, and still looks scattered
    because the empty cells break the pattern.

    Seeded, so every machine in the room sees the same arrangement.
    """
    cell_w = (X_RANGE[1] - X_RANGE[0]) / GRID_COLS
    cell_h = (Y_RANGE[1] - Y_RANGE[0]) / GRID_ROWS

    cells = [(c, r) for c in range(GRID_COLS) for r in range(GRID_ROWS)]
    rng.shuffle(cells)
    if len(specs) > len(cells):
        raise RuntimeError(
            f"{len(specs)} objects will not fit in "
            f"{GRID_COLS}x{GRID_ROWS} cells; enlarge the grid")

    for (name, shape, size, dims), (col, row) in zip(specs, cells):
        cx = X_RANGE[0] + (col + 0.5) * cell_w
        cy = Y_RANGE[0] + (row + 0.5) * cell_h
        free_x = max(0.0, (cell_w - dims[0]) / 2.0) * JITTER
        free_y = max(0.0, (cell_h - dims[1]) / 2.0) * JITTER
        x = cx + rng.uniform(-free_x, free_x)
        y = cy + rng.uniform(-free_y, free_y)
        yield name, shape, size, dims, x, y


def build_scene(seed=PLACEMENT_SEED):
    rng = random.Random(seed)

    specs = []
    for color, wanted in SCENE_PLAN.items():
        for shape, size in wanted:
            name = f"{size} {color} {shape}" if size else f"{color} {shape}"
            specs.append((name, shape, size, DIMS[(shape, size)]))

    rng.shuffle(specs)

    scene = {}
    for name, shape, size, dims, x, y in _scatter(specs, rng):
        color = name.split()[1] if size else name.split()[0]
        xyz = [round(x, 4), round(y, 4), dims[2] / 2.0]
        scene[name] = {
            "xyz": xyz,
            # Where the object STARTED. xyz changes as it is moved around;
            # origin never does, so "put it back on the table" has a
            # position to mean.
            "origin": list(xyz),
            "color": color,
            "shape": shape,
            "size": size,
            "dims": dims,
            "volume": dims[0] * dims[1] * dims[2],
            "graspable": True,
        }

    for name, d in DESTINATIONS.items():
        scene[name] = {
            "xyz": list(d["xyz"]),
            "color": d["color"],
            "shape": d["shape"],
            "size": None,
            "dims": d["dims"],
            "volume": 0.0,
            "graspable": False,
        }
    return scene

class SceneNode(Node):
    def __init__(self):
        super().__init__("scene_node")

        self.declare_parameter("scene_seed", PLACEMENT_SEED)
        seed = int(self.get_parameter("scene_seed").value)
        self.scene = build_scene(seed)

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)

        # In Phase 1 this node IS the world: it publishes /world_state and
        # the executor believes it.
        #
        # In Phase 2 the perception node publishes /world_state instead, and
        # the launch file remaps this node's output to /scene_truth, where it
        # becomes the ground truth the simulated camera renders from. Same
        # data, same code, different place in the graph.
        # ONE publisher. In Phase 1 it goes to the executor as /world_state.
        # In Phase 2 the launch file remaps it to /scene_truth so it feeds
        # the camera instead. Publishing both topics from here meant the same
        # data went out twice under two names, which is confusing to read in
        # ros2 topic list and serves no purpose.
        self.pub_world = self.create_publisher(String, "world_state", latched)
        self.pub_markers = self.create_publisher(
            MarkerArray, "scene_markers", 1)
        self.create_subscription(String, "scene_command", self.on_command, 10)

        # Listen to whatever world the executor is actually using.
        #
        # In Phase 1 that is this node's own output and the names match, so
        # this changes nothing. In Phase 2 it is the perception node's, whose
        # names are its own invention: it calls something "blue cube" where
        # this node calls it "small blue cube", because it only found one
        # blue cube and had nothing to compare sizes against.
        #
        # Keeping that world lets an unknown name be resolved by POSITION
        # instead, which is the physically meaningful test: the object that
        # was at this point is the one that moved.
        # NOT called "world_state".
        #
        # The Phase 2 launch file remaps this node's world_state to
        # scene_truth so it feeds the camera instead of the executor. A remap
        # rule matches a topic NAME, so it catches subscriptions as well as
        # publications: naming this one world_state meant it silently
        # subscribed to this node's own output, learned nothing about
        # perception's names, and every position match failed.
        #
        # A distinct name cannot be caught by that rule. The launch file
        # points it at the right place explicitly.
        self.external = {}
        self.create_subscription(
            String, "perceived_world", self.on_external, latched)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.held = None

        self.publish_world()
        self.create_timer(0.05, self.publish_markers)

        graspable = sum(1 for o in self.scene.values() if o["graspable"])
        self.get_logger().info(
            f"scene ready, {graspable} graspable objects in "
            f"{len(COLORS)} colours and {len(SHAPES)} shapes, seed {seed}")

    # ------------------------------------------------------------------

    def publish_world(self):
        """Latched, so a node that starts later still receives it."""
        msg = String()
        msg.data = json.dumps({
            "objects": self.scene,
            "colors": COLORS,
            "shapes": SHAPES,
        })
        self.pub_world.publish(msg)

    def on_external(self, msg):
        """Cache the world the executor is working from, for name matching."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objects = data.get("objects") or {}
        self.external = {n: o.get("xyz") for n, o in objects.items()
                         if o.get("xyz")}

    def resolve_name(self, name):
        """Map a name from the executor's world onto one of ours.

        Exact match first, which is every case in Phase 1. Failing that, look
        up where the executor thinks that object is and take the nearest
        object we have. Anything further than 10 cm away is a different
        object, not a naming difference, and is rejected.
        """
        if name in self.scene:
            return name

        target = self.external.get(name)
        if target is None:
            return None

        best, best_d = None, 1e9
        for other, obj in self.scene.items():
            if not obj.get("graspable", True):
                continue
            d = math.dist(obj["xyz"], target)
            if d < best_d:
                best, best_d = other, d

        if best is None or best_d > 0.10:
            self.get_logger().warn(
                f'cannot match "{name}" to anything here'
                + (f", nearest is {best_d * 100:.0f} cm away" if best else ""))
            return None

        self.get_logger().info(
            f'matched "{name}" to "{best}" by position '
            f"({best_d * 1000:.0f} mm apart)")
        return best

    def on_command(self, msg):
        """Apply a change the executor has made to the world.

        A camera would need none of this. It would simply see the object
        somewhere else on the next frame.
        """
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if "held" in cmd:
            want = cmd["held"] or None
            self.held = self.resolve_name(want) if want else None

        place = cmd.get("place")
        if place:
            obj = self.resolve_name(place.get("object"))
            dest = self.resolve_name(place.get("on"))
            if obj in self.scene and dest in self.scene:
                at = place.get("at")
                if at:
                    # The executor chose the exact spot, e.g. returning an
                    # object to its origin on the table. Trust it: the arm
                    # flew there, so the world must agree with the arm.
                    self.scene[obj]["xyz"] = [float(v) for v in at]
                else:
                    dx, dy, dz = self.scene[dest]["xyz"]
                    rest = (dz
                            + self.scene[dest]["dims"][2] / 2.0
                            + self.scene[obj]["dims"][2] / 2.0)
                    self.scene[obj]["xyz"] = [dx, dy, rest]
                self.get_logger().info(
                    f"world updated: {obj} now rests on {dest}")
                self.publish_world()

    # ------------------------------------------------------------------

    def gripper_xyz(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, GRIPPER_FRAME,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return None
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    def publish_markers(self):
        grip = self.gripper_xyz() if self.held else None
        now = self.get_clock().now().to_msg()
        array = MarkerArray()

        for i, (name, data) in enumerate(self.scene.items()):
            # The URDF already draws the table. A second coplanar slab would
            # z-fight with it, and a floating label in the middle of the
            # workspace helps nobody.
            if data["shape"] == "table":
                continue
            if name == self.held and grip is not None:
                pos = (grip[0], grip[1], grip[2] + CARRY_OFFSET)
            else:
                pos = tuple(data["xyz"])

            m = Marker()
            m.header.frame_id = WORLD_FRAME
            m.header.stamp = now
            m.ns = "scene"
            m.id = i
            m.type = MARKER_TYPE.get(data["shape"], Marker.CUBE)
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = [float(d) for d in data["dims"]]
            r, g, b = RGB.get(data["color"], (0.6, 0.6, 0.6))
            m.color.r, m.color.g, m.color.b = r, g, b
            m.color.a = 0.95
            array.markers.append(m)

            label = Marker()
            label.header = m.header
            label.ns = "scene_labels"
            label.id = 1000 + i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(pos[0])
            label.pose.position.y = float(pos[1])
            label.pose.position.z = float(pos[2]) + data["dims"][2] / 2.0 + 0.05
            label.pose.orientation.w = 1.0
            label.scale.z = 0.035
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 0.85
            label.text = f"[{name}]" if name == self.held else name
            array.markers.append(label)

        self.pub_markers.publish(array)


def main():
    rclpy.init()
    node = SceneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
perception_node -- turns pixels into a world.

THIS IS THE PHASE 2 SWAP

It publishes /world_state, exactly like scene_node does in Phase 1. The
executor cannot tell them apart, and that is the entire point: nothing
downstream of /world_state changes when the source of the world changes from
a dictionary to a camera.

    scene_node        --/world_state-->  executor    Phase 1
    perception_node   --/world_state-->  executor    Phase 2

Note what this node does NOT subscribe to: /scene_command. scene_node needs
telling when an object moves, because a made up world has no other way to
find out. A camera does not need telling. It simply sees the object somewhere
else on the next frame. That asymmetry is worth pointing at.

THE PIPELINE, WHICH IS THE REAL ONE

    1  colour segmentation      hue ranges over the RGB image
    2  connected components     which pixels form one object
    3  depth lookup             the median depth across each blob
    4  deprojection             pixel plus depth to a 3D point, via the
                                camera intrinsics
    5  transform                optical frame to world frame, via TF
    6  shape classification     from the blob outline and the depth profile

Every step is what you would run against a RealSense. Only the publisher of
the images changes.

NO NEW DEPENDENCIES

NumPy and SciPy only. No OpenCV, no cv_bridge. SciPy is already installed
because ikpy needs it. The point is to show the algorithm, not to hide it
behind a library call.
"""

import json

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from scipy import ndimage
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

WORLD_FRAME = "world"

# Hue ranges in degrees, 0 to 360. Real colour segmentation works in hue
# rather than RGB because hue is far less affected by lighting: a shadowed
# red block and a bright one have similar hue and very different RGB.
HUE_RANGES = {
    "red":    [(0, 15), (345, 360)],   # red wraps around zero
    "green":  [(90, 160)],
    "blue":   [(200, 260)],
    # Measured, not guessed. The wooden table sits at hue 29 and the brown
    # shelf at 38, both close enough to yellow to be swept up by a loose
    # range. The golden objects are at 46, so starting at 40 separates them
    # cleanly. This is the ordinary business of colour segmentation: you look
    # at the actual hues in your actual scene.
    "golden": [(40, 65)],
}

MIN_SATURATION = 0.30    # below this it is grey, not a colour
MIN_VALUE = 0.15         # below this it is shadow
MIN_BLOB_PIXELS = 60     # smaller than this is noise or a sliver

# Anything wider than this is furniture or background, not something the arm
# could pick up. Expressed in metres rather than pixels so it does not depend
# on the camera's resolution or how far away it is. Measured: the objects in
# this scene span 6 to 9 cm, the shelf spans 16.
MAX_OBJECT_SPAN_M = 0.13

# The table, the tray and the shelf are fixtures. In a real cell their positions are
# surveyed once during commissioning, not discovered by a camera every
# frame, and treating them as detections would be pretending otherwise. They
# are also the one thing perception genuinely cannot supply: a camera has no
# way to know that a flat brown surface is called "the shelf".
FIXTURES = {
    "tray":  {"xyz": [0.30, -0.52, 0.02], "color": "grey", "shape": "tray",
              "dims": [0.22, 0.16, 0.03]},
    "shelf": {"xyz": [0.28, 0.52, 0.28], "color": "brown", "shape": "shelf",
              "dims": [0.20, 0.16, 0.03]},
    # The tabletop, same numbers as scene_node and the URDF table_link.
    "table": {"xyz": [0.55, 0.0, -0.01], "color": "brown", "shape": "table",
              "dims": [0.90, 1.20, 0.02]},
}

# Shape classification thresholds. Measured across every object, in every
# place it can end up: on the table, on the shelf and on the tray.
#
#   shape      fill        depth range   height / width
#   cuboid     0.95-1.00   0.000-0.054   -
#   cylinder   0.63-0.75   0.081-0.110   1.64-2.07
#   sphere     0.74-0.81   0.029-0.081   0.64-1.07
#
# Note the depth range of a cylinder and a large sphere now overlap at
# 0.081, because doubling the ball diameter doubled its depth range. That
# discriminator worked when the balls were small and stopped working when
# they grew, which is exactly the kind of silent breakage a scene change
# causes.
#
# Height divided by width does not have that problem. A ball is as tall as
# it is wide by definition, so the ratio sits near 1. A cylinder here is
# 11 cm tall and 5 cm across, so it sits near 2. The gap from 1.07 to 1.64
# is wide, and it comes from the shape itself rather than from tuning.
FILL_IS_BOX = 0.88          # boxes fill their bounding box, round things do not
TALL_AND_NARROW = 1.35      # height / width above this is a cylinder

RATE_HZ = 2.0


def rgb_to_hsv(rgb):
    """Vectorised RGB to HSV. rgb is HxWx3 float in 0..1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    diff = mx - mn

    hue = np.zeros_like(mx)
    mask = diff > 1e-6
    idx = mask & (mx == r)
    hue[idx] = (60 * ((g[idx] - b[idx]) / diff[idx])) % 360
    idx = mask & (mx == g)
    hue[idx] = 60 * ((b[idx] - r[idx]) / diff[idx]) + 120
    idx = mask & (mx == b)
    hue[idx] = 60 * ((r[idx] - g[idx]) / diff[idx]) + 240

    sat = np.zeros_like(mx)
    sat[mx > 1e-6] = diff[mx > 1e-6] / mx[mx > 1e-6]
    return hue, sat, mx


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("min_blob_pixels", MIN_BLOB_PIXELS)
        self.declare_parameter("max_object_span_m", MAX_OBJECT_SPAN_M)
        self.min_blob = int(self.get_parameter("min_blob_pixels").value)
        self.max_span = float(self.get_parameter("max_object_span_m").value)

        self.rgb = None
        self.depth = None
        self.info = None
        self.optical_frame = None

        self.create_subscription(
            Image, "camera/color/image_raw", self.on_rgb, 1)
        self.create_subscription(
            Image, "camera/depth/image_rect_raw", self.on_depth, 1)
        self.create_subscription(
            CameraInfo, "camera/color/camera_info", self.on_info, 1)

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.pub_world = self.create_publisher(String, "world_state", latched)

        # What perception BELIEVES, drawn next to what is actually there.
        #
        # Nothing subscribes to this except RViz. It exists so the difference
        # between being told where something is and measuring where it is can
        # be seen rather than read off a table of numbers. The detections are
        # drawn as hollow wireframes so the solid ground truth shows through.
        self.pub_markers = self.create_publisher(
            MarkerArray, "detected_markers", 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.reported = 0
        self.waited = 0
        self.create_timer(1.0 / RATE_HZ, self.tick)
        self.get_logger().info("perception ready, waiting for images")

    # -- subscriptions ---------------------------------------------------

    def on_rgb(self, msg):
        if msg.encoding != "rgb8":
            self.get_logger().error(f"expected rgb8, got {msg.encoding}")
            return
        self.rgb = (np.frombuffer(msg.data, dtype=np.uint8)
                    .reshape(msg.height, msg.width, 3).astype(np.float32)
                    / 255.0)
        self.optical_frame = msg.header.frame_id

    def on_depth(self, msg):
        if msg.encoding != "32FC1":
            self.get_logger().error(f"expected 32FC1, got {msg.encoding}")
            return
        self.depth = (np.frombuffer(msg.data, dtype=np.float32)
                      .reshape(msg.height, msg.width))

    def on_info(self, msg):
        self.info = msg

    # -- the pipeline ----------------------------------------------------

    def detect(self):
        """Return a list of detections, each a dict in the camera's frame."""
        hue, sat, val = rgb_to_hsv(self.rgb)
        colourful = (sat >= MIN_SATURATION) & (val >= MIN_VALUE)
        has_depth = self.depth > 0

        out = []
        for colour, ranges in HUE_RANGES.items():
            mask = np.zeros(hue.shape, dtype=bool)
            for lo, hi in ranges:
                mask |= (hue >= lo) & (hue <= hi)
            mask &= colourful & has_depth
            if not mask.any():
                continue

            # Connected components. Pixels of the same colour that touch each
            # other are one object; two blue objects apart on the table are
            # two.
            labels, count = ndimage.label(mask)
            for i in range(1, count + 1):
                blob = labels == i
                pixels = int(blob.sum())
                if pixels < self.min_blob:
                    continue
                det = self.measure(blob, colour, pixels)
                if det is not None:
                    out.append(det)
        return out

    def measure(self, blob, colour, pixels):
        """Turn one blob of pixels into a 3D detection."""
        ys, xs = np.nonzero(blob)
        depths = self.depth[blob]
        depths = depths[depths > 0]
        if depths.size < 5:
            return None

        # The median, not the mean. A blob's edge pixels straddle the object
        # and whatever is behind it, so their depth is unreliable. The median
        # ignores those; the mean would be dragged towards the table.
        z = float(np.median(depths))

        u = float(xs.mean())
        v = float(ys.mean())

        k = self.info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        # Deprojection. The one piece of geometry in this node, and the same
        # three lines you would write for any depth camera.
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        fill = pixels / float(h * w)
        elong = max(h, w) / float(min(h, w))

        # Apparent width in metres, from the blob area and how far away it
        # is. Needed here because the shape test compares height to width.
        k = self.info.k
        span = float(np.sqrt(pixels) * z / k[0])

        # The full depth range across the blob, which turns out to be the
        # single most useful number here.
        #
        # A cylinder standing on the table shows its side wall as well as its
        # top, all the way down to the surface, so this range is essentially
        # the object's HEIGHT: measured at 0.109 to 0.110 m for the 0.11 m
        # cylinders. A sphere curves out of view long before the table, so
        # its range is only 0.03. That is a large, physically meaningful gap
        # rather than a tuned constant.
        #
        # An earlier version measured curvature near the blob's centre
        # instead, reasoning that a sphere curves and a cylinder cap is flat.
        # That is true but the signal is tiny, and every cylinder in the
        # scene was classified as a sphere. The side wall is the strong
        # signal, and it was there all along.
        depth_range = float(depths.max() - depths.min())

        return {
            "colour": colour,
            "xyz_optical": (x, y, z),
            "pixels": pixels,
            "fill": fill,
            "elongation": elong,
            "span": span,
            "depth_range": depth_range,
            "uv": (u, v),
        }

    @staticmethod
    def classify(det):
        """Name the shape from the outline and the depth profile.

        Two questions, in order.

        First, is the outline a box or is it round? A box fills its bounding
        rectangle almost completely, 0.95 and up. A circle fills only about
        three quarters of one, because of the corners it does not occupy.

        Second, for the round ones: is it tall and narrow, or is it as tall
        as it is wide? Depth range divided by width answers that. A ball
        gives about 1, a standing cylinder about 2.
        """
        if det["fill"] >= FILL_IS_BOX:
            return "cuboid"
        width = max(det["span"], 1e-6)
        return ("cylinder" if det["depth_range"] / width >= TALL_AND_NARROW
                else "sphere")

    def to_world(self, xyz_optical):
        """Optical frame to world frame, through TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, self.optical_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
        except Exception:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        p = rot @ np.array(xyz_optical) + np.array([t.x, t.y, t.z])
        return [round(float(v), 4) for v in p]

    # -- publishing ------------------------------------------------------

    def publish_markers(self, objects):
        """Draw each detection as a wireframe box around where it was found.

        Deliberately not a solid shape. A solid marker would sit on top of
        the ground truth and hide exactly the thing worth looking at, which
        is the small offset between the two. An outline lets the real object
        show through the middle.
        """
        now = self.get_clock().now().to_msg()
        array = MarkerArray()

        for i, (name, o) in enumerate(sorted(objects.items())):
            if not o.get("graspable"):
                continue
            x, y, z = o["xyz"]
            half = max(o["dims"][0], 0.02) / 2.0 + 0.008

            box = Marker()
            box.header.frame_id = WORLD_FRAME
            box.header.stamp = now
            box.ns = "detected"
            box.id = i
            box.type = Marker.LINE_LIST
            box.action = Marker.ADD
            box.pose.orientation.w = 1.0
            box.scale.x = 0.0025
            box.color.r, box.color.g, box.color.b, box.color.a = (
                1.0, 1.0, 1.0, 0.85)

            c = [(sx, sy, sz) for sx in (-half, half)
                 for sy in (-half, half) for sz in (-half, half)]
            edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                     (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            for a, b in edges:
                for idx in (a, b):
                    pt = Point()
                    pt.x = x + c[idx][0]
                    pt.y = y + c[idx][1]
                    pt.z = z + c[idx][2]
                    box.points.append(pt)
            array.markers.append(box)

            label = Marker()
            label.header = box.header
            label.ns = "detected_labels"
            label.id = 500 + i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = z + half + 0.10
            label.pose.orientation.w = 1.0
            label.scale.z = 0.028
            label.color.r, label.color.g, label.color.b, label.color.a = (
                1.0, 0.85, 0.2, 0.9)
            label.text = f"seen: {name}"
            array.markers.append(label)

        self.pub_markers.publish(array)

    def tick(self):
        # Say what is missing rather than returning silently. Three separate
        # things have to arrive before this node can do anything, and a quiet
        # return makes a broken chain look like a working one.
        missing = [n for n, v in (("rgb", self.rgb),
                                  ("depth", self.depth),
                                  ("camera_info", self.info)) if v is None]
        if missing:
            self.waited += 1
            if self.waited in (10, 50, 150):
                self.get_logger().warn(
                    f"waiting for {', '.join(missing)} after "
                    f"{self.waited / RATE_HZ:.0f}s. Is camera_node "
                    f"publishing?")
            return
        if self.optical_frame is None:
            return

        detections = self.detect()

        # Measure everything first, name it afterwards. A name like "small
        # blue cube" only means anything relative to the other blue cube, so
        # the group has to be complete before anything can be called small.
        measured = []
        for det in detections:
            world = self.to_world(det["xyz_optical"])
            if world is None:
                self.waited += 1
                if self.waited in (10, 50, 150):
                    self.get_logger().warn(
                        f"images are arriving but TF is not: cannot look up "
                        f"{WORLD_FRAME} -> {self.optical_frame}")
                return   # try again next tick

            span = det["span"]
            if span > self.max_span:
                # Too wide to be a graspable object. A workbench, a wall, or
                # in this scene the shelf, whose brown edges creep into the
                # yellow hue range. Colour alone cannot tell furniture from
                # objects; size can.
                continue

            measured.append({
                "colour": det["colour"],
                "shape": self.classify(det),
                "world": world,
                "span": span,
                "pixels": det["pixels"],
            })

        objects = {}
        groups = {}
        for m in measured:
            groups.setdefault((m["colour"], m["shape"]), []).append(m)

        for (colour, shape), members in groups.items():
            base = f"{colour} {shape}"
            members.sort(key=lambda m: m["span"])

            # Two of a kind that are clearly different sizes get called small
            # and large, which is how a person would refer to them and how
            # they are named when the world is made up rather than measured.
            # Anything else is just numbered.
            use_size = (len(members) == 2
                        and members[1]["span"] > members[0]["span"] * 1.25)

            for i, m in enumerate(members):
                if use_size:
                    name = f"{'small' if i == 0 else 'large'} {base}"
                    size = "small" if i == 0 else "large"
                elif len(members) == 1:
                    name, size = base, None
                else:
                    name, size = f"{base} {i + 1}", None

                objects[name] = {
                    "xyz": m["world"],
                    "color": colour,
                    "shape": shape,
                    "size": size,
                    "dims": [round(m["span"], 3)] * 3,
                    "volume": round(m["span"] ** 3, 6),
                    "graspable": True,
                    "pixels": m["pixels"],
                }

        if not objects:
            return

        # Fixtures are added, not detected. See the note by FIXTURES.
        for name, f in FIXTURES.items():
            objects[name] = {
                "xyz": list(f["xyz"]),
                "color": f["color"],
                "shape": f["shape"],
                "size": None,
                "dims": list(f["dims"]),
                "volume": 0.0,
                "graspable": False,
                "pixels": 0,
            }

        msg = String()
        msg.data = json.dumps({
            "objects": objects,
            "colors": sorted({o["color"] for o in objects.values()
                              if o["graspable"]}),
            "shapes": sorted({o["shape"] for o in objects.values()
                              if o["graspable"]}),
        })
        self.pub_world.publish(msg)
        self.publish_markers(objects)

        detected = sum(1 for o in objects.values() if o["graspable"])
        if detected != self.reported:
            self.reported = detected
            self.get_logger().info(
                f"seeing {detected} objects, plus {len(FIXTURES)} known "
                f"fixtures")
            for name, o in sorted(objects.items()):
                if not o["graspable"]:
                    continue
                self.get_logger().info(
                    f"  {name:<20} at {o['xyz']}  ({o['pixels']} px)")


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

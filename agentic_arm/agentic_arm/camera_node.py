#!/usr/bin/env python3
"""
camera_node -- a synthetic RGB-D camera looking down at the table.

WHY THIS EXISTS

The perception pipeline downstream is real: HSV thresholding, contour
extraction, depth lookup, deprojection through camera intrinsics, and a TF
into the world frame. All of that needs actual images.

This node produces them by raycasting the known scene through a pinhole
model, which means the geometry is exact and reproducible. Every participant
sees the same pixels, there is no hardware to fail, and the same perception
code runs unchanged against a RealSense or a rosbag: only this publisher
gets swapped out.

WHAT IT PUBLISHES

    /camera/color/image_raw          sensor_msgs/Image   rgb8
    /camera/depth/image_rect_raw     sensor_msgs/Image   32FC1, metres
    /camera/color/camera_info        sensor_msgs/CameraInfo

All in camera_optical_frame, which is REP-103: z forward, x right, y down.
That frame already exists in the URDF.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import json

WIDTH = 640
HEIGHT = 480
HFOV_DEG = 62.0          # roughly a RealSense D435 colour stream
RATE_HZ = 5.0
MAX_DEPTH = 5.0          # metres, beyond this is "no return"

WORLD_FRAME = "world"
OPTICAL_FRAME = "camera_optical_frame"

# Must match the URDF: <joint name="world-camera"> origin xyz rpy
CAM_XYZ = np.array([0.55, 0.0, 1.20])
CAM_RPY = np.array([0.0, 1.5708, 0.0])
# camera_link -> camera_optical_frame
OPT_RPY = np.array([-1.5708, 0.0, -1.5708])

TABLE_Z = 0.0
TABLE_RGB = (140, 107, 76)
BACKGROUND_RGB = (25, 25, 28)

RGB255 = {
    "red":    (217, 38, 38),
    "green":  (38, 179, 64),
    "blue":   (38, 77, 217),
    "golden": (217, 173, 33),
    "grey":   (128, 128, 140),
    "brown":  (153, 115, 51),
}


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


class Renderer:
    """Raycasts the scene. Pure numpy, no ROS, so it can be tested alone."""

    def __init__(self, width=WIDTH, height=HEIGHT, hfov_deg=HFOV_DEG):
        self.w, self.h = width, height
        self.fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Ray directions, in the optical frame, one per pixel.
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        dirs = np.stack([(u - self.cx) / self.fx,
                         (v - self.cy) / self.fy,
                         np.ones_like(u, dtype=float)], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # Optical frame -> world.
        R_world_cam = rpy_to_matrix(CAM_RPY)
        R_cam_opt = rpy_to_matrix(OPT_RPY)
        self.R = R_world_cam @ R_cam_opt
        self.origin = CAM_XYZ

        self.dirs_world = dirs @ self.R.T          # (h, w, 3)
        self.flat_dirs = self.dirs_world.reshape(-1, 3)

    # -- primitives, each returns ray distance t or inf ------------------

    def _project(self, pts):
        """World points to pixel coordinates. Used to window the raycast."""
        rel = (np.asarray(pts) - self.origin) @ self.R      # into optical
        z = np.maximum(rel[:, 2], 1e-6)
        u = self.fx * rel[:, 0] / z + self.cx
        v = self.fy * rel[:, 1] / z + self.cy
        return u, v

    def _window(self, centre, dims, margin=4):
        """Pixel indices an object could possibly occupy.

        Raycasting every pixel against every object is the obvious
        implementation and it took 950 ms a frame. Each object actually
        covers about a thousand pixels out of three hundred thousand, so
        projecting its bounding box first and testing only those is roughly
        two orders of magnitude less work.
        """
        c = np.asarray(centre, dtype=float)
        h = np.asarray(dims, dtype=float) / 2.0
        corners = c + np.array([[sx, sy, sz]
                                for sx in (-h[0], h[0])
                                for sy in (-h[1], h[1])
                                for sz in (-h[2], h[2])])
        u, v = self._project(corners)
        u0 = int(max(0, np.floor(u.min()) - margin))
        u1 = int(min(self.w, np.ceil(u.max()) + margin))
        v0 = int(max(0, np.floor(v.min()) - margin))
        v1 = int(min(self.h, np.ceil(v.max()) + margin))
        if u1 <= u0 or v1 <= v0:
            return None
        vv, uu = np.meshgrid(np.arange(v0, v1), np.arange(u0, u1),
                             indexing="ij")
        return (vv * self.w + uu).ravel()

    def _hit_plane(self, z):
        d = self.flat_dirs[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - self.origin[2]) / d
        t[(d >= -1e-9) | (t <= 0)] = np.inf
        return t

    def _hit_box(self, centre, dims, idx):
        d = self.flat_dirs[idx]
        lo = np.array(centre) - np.array(dims) / 2.0
        hi = np.array(centre) + np.array(dims) / 2.0
        inv = np.divide(1.0, d, out=np.full_like(d, np.inf),
                        where=np.abs(d) > 1e-12)
        t0 = (lo - self.origin) * inv
        t1 = (hi - self.origin) * inv
        tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)
        tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
        t = np.where((tmax >= np.maximum(tmin, 0.0)), tmin, np.inf)
        t[t <= 0] = np.inf
        return t

    def _hit_sphere(self, centre, radius, idx):
        d = self.flat_dirs[idx]
        oc = self.origin - np.array(centre)
        b = 2.0 * (d @ oc)
        c = oc @ oc - radius * radius
        disc = b * b - 4.0 * c
        t = np.full(len(d), np.inf)
        ok = disc >= 0
        root = np.sqrt(np.maximum(disc, 0.0))
        near = (-b - root) / 2.0
        far = (-b + root) / 2.0
        pick = np.where(near > 1e-6, near, far)
        t[ok & (pick > 1e-6)] = pick[ok & (pick > 1e-6)]
        return t

    def _hit_cylinder(self, centre, radius, height, idx):
        """Axis aligned with world z. Side wall plus flat caps."""
        cx, cy, cz = centre
        z_lo, z_hi = cz - height / 2.0, cz + height / 2.0
        ox, oy, oz = self.origin
        sub = self.flat_dirs[idx]
        dx = sub[:, 0]
        dy = sub[:, 1]
        dz = sub[:, 2]

        a = dx * dx + dy * dy
        b = 2.0 * (dx * (ox - cx) + dy * (oy - cy))
        c = (ox - cx) ** 2 + (oy - cy) ** 2 - radius * radius
        disc = b * b - 4.0 * a * c

        t = np.full(len(dx), np.inf)
        side_ok = (disc >= 0) & (np.abs(a) > 1e-12)
        root = np.sqrt(np.maximum(disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            near = (-b - root) / (2.0 * a)
            far = (-b + root) / (2.0 * a)
        for cand in (near, far):
            z_at = oz + cand * dz
            good = side_ok & (cand > 1e-6) & (z_at >= z_lo) & (z_at <= z_hi)
            t = np.where(good & (cand < t), cand, t)

        # Caps. The top cap is what an overhead camera actually sees, and it
        # is flat, which is how depth tells a cylinder from a sphere.
        for cap_z in (z_lo, z_hi):
            with np.errstate(divide="ignore", invalid="ignore"):
                tc = (cap_z - oz) / dz
            x_at = ox + tc * dx
            y_at = oy + tc * dy
            inside = (x_at - cx) ** 2 + (y_at - cy) ** 2 <= radius * radius
            good = inside & (tc > 1e-6) & np.isfinite(tc)
            t = np.where(good & (tc < t), tc, t)
        return t

    # -------------------------------------------------------------------

    def render(self, scene):
        """Return (rgb uint8 HxWx3, depth float32 HxW in metres)."""
        n = self.h * self.w
        best = np.full(n, np.inf)
        colour = np.zeros((n, 3), dtype=np.uint8)
        colour[:] = BACKGROUND_RGB

        # Table first, so objects overwrite it where they are nearer.
        t_table = self._hit_plane(TABLE_Z)
        hit = t_table < best
        best = np.where(hit, t_table, best)
        colour[hit] = TABLE_RGB

        for name, obj in scene.items():
            shape = obj.get("shape")
            # The tabletop is already painted by the plane above. Rendering
            # the table entity as a box would repaint the whole table region
            # and shift the hue statistics the classifier was measured on.
            if shape == "table":
                continue
            dims = obj.get("dims", (0.05, 0.05, 0.05))
            centre = obj["xyz"]

            idx = self._window(centre, dims)
            if idx is None:
                continue

            if shape == "sphere":
                t = self._hit_sphere(centre, dims[0] / 2.0, idx)
            elif shape == "cylinder":
                t = self._hit_cylinder(centre, dims[0] / 2.0, dims[2], idx)
            else:
                t = self._hit_box(centre, dims, idx)

            nearer = t < best[idx]
            if not nearer.any():
                continue
            chosen = idx[nearer]
            best[chosen] = t[nearer]
            colour[chosen] = RGB255.get(obj.get("color"), (200, 200, 200))

        # Ray distance is along the ray. Depth images store distance along
        # the optical z axis, so project it back. Getting this wrong is a
        # classic source of poses that are subtly short.
        cos = self.dirs_world.reshape(-1, 3) @ self.R[:, 2]
        depth = (best * cos).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth > MAX_DEPTH] = 0.0

        return (colour.reshape(self.h, self.w, 3),
                depth.reshape(self.h, self.w))


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.renderer = Renderer()
        self.scene = {}
        self.waited = 0

        self.pub_rgb = self.create_publisher(
            Image, "camera/color/image_raw", 1)
        self.pub_depth = self.create_publisher(
            Image, "camera/depth/image_rect_raw", 1)
        self.pub_info = self.create_publisher(
            CameraInfo, "camera/color/camera_info", 1)

        # LATCHED. scene_node publishes the world once at startup and then
        # only when something moves. A default subscription only receives
        # messages sent after it connects, so if this node is not listening
        # at that exact moment it waits forever, renders nothing, and the
        # whole perception chain sits silent with no error anywhere.
        #
        # Transient local durability asks the publisher to replay the last
        # message to any subscriber that arrives late. The executor already
        # subscribes this way for /world_state; the camera has to as well.
        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(
            String, "scene_truth", self.on_scene, latched)
        self.create_timer(1.0 / RATE_HZ, self.tick)

        self.get_logger().info(
            f"synthetic camera {WIDTH}x{HEIGHT}, "
            f"fx={self.renderer.fx:.1f}, {RATE_HZ:.0f} Hz")

    def on_scene(self, msg):
        """Ground truth from scene_node, remapped onto /scene_truth.

        The message wraps the objects alongside the colour and shape lists,
        so take the objects out of it. Rendering needs geometry, nothing
        else.
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objects = data.get("objects", data)
        if not isinstance(objects, dict) or not objects:
            return
        first = not self.scene
        self.scene = objects
        if first:
            self.get_logger().info(
                f"ground truth received, rendering {len(objects)} objects")

    def tick(self):
        if not self.scene:
            self.waited += 1
            if self.waited in (10, 50, 150):
                self.get_logger().warn(
                    f"still nothing on /scene_truth after "
                    f"{self.waited / RATE_HZ:.0f}s. Is scene_node running, "
                    f"and is its world_state remapped to scene_truth?")
            return
        rgb, depth = self.renderer.render(self.scene)
        stamp = self.get_clock().now().to_msg()

        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = OPTICAL_FRAME
        m.height, m.width = rgb.shape[0], rgb.shape[1]
        m.encoding = "rgb8"
        m.is_bigendian = 0
        m.step = m.width * 3
        m.data = rgb.tobytes()
        self.pub_rgb.publish(m)

        d = Image()
        d.header = m.header
        d.height, d.width = depth.shape
        d.encoding = "32FC1"
        d.is_bigendian = 0
        d.step = d.width * 4
        d.data = depth.tobytes()
        self.pub_depth.publish(d)

        info = CameraInfo()
        info.header = m.header
        info.height, info.width = m.height, m.width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        r = self.renderer
        info.k = [r.fx, 0.0, r.cx, 0.0, r.fy, r.cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [r.fx, 0.0, r.cx, 0.0,
                  0.0, r.fy, r.cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

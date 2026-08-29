#!/usr/bin/env python3
"""
arm_node -- the whole robot side of the workshop in one file.

Subscribes : /arm_goal   (geometry_msgs/PoseStamped)  target for the gripper tip
Publishes  : /joint_states (sensor_msgs/JointState)   at 50 Hz
             /goal_marker  (visualization_msgs/Marker) so you can see the target

There is no MoveIt, no controller, no simulator. robot_state_publisher reads
/joint_states, does forward kinematics from the URDF, and RViz draws the result.
This node's only job is to turn a Cartesian point into joint angles and then
walk the arm there smoothly.
"""

import os
import tempfile
import warnings

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from visualization_msgs.msg import Marker

warnings.filterwarnings("ignore", category=UserWarning)
from ikpy.chain import Chain  # noqa: E402

# The six revolute joints of a UR5e, in URDF order. JointState must use
# exactly these names or robot_state_publisher silently ignores them.
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ikpy walks a URDF as a tree and picks the first branch it finds. The UR5e
# description branches at wrist_3 (ft_frame, flange, tool0), so left alone it
# solves to the wrong tip. Spelling the path out forces the right chain.
CHAIN_ELEMENTS = [
    "base_link", "base_link-base_link_inertia", "base_link_inertia",
    "shoulder_pan_joint", "shoulder_link",
    "shoulder_lift_joint", "upper_arm_link",
    "elbow_joint", "forearm_link",
    "wrist_1_joint", "wrist_1_link",
    "wrist_2_joint", "wrist_2_link",
    "wrist_3_joint", "wrist_3_link",
    "wrist_3-flange", "flange",
    "flange-tool0", "tool0",
    "tool0-gripper", "gripper_link",
    "gripper-grasp", "grasp_point",
]

# Which entries in the resulting chain are real degrees of freedom.
ACTIVE_MASK = [False, False] + [True] * 6 + [False] * 4

HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

RATE_HZ = 50.0
MAX_JOINT_SPEED = 1.0  # rad/s, sets how long a move takes


class ArmNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_node")

        urdf = self.declare_parameter("robot_description", "").value
        if not urdf:
            self.get_logger().fatal(
                "robot_description parameter is empty. Launch this via "
                "display.launch.py, not with ros2 run."
            )
            raise SystemExit(1)

        # ikpy reads from a path, not a string. Write it out, load, delete.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".urdf", delete=False
        ) as handle:
            handle.write(urdf)
            path = handle.name
        try:
            self.chain = Chain.from_urdf_file(
                path, base_elements=CHAIN_ELEMENTS, active_links_mask=ACTIVE_MASK
            )
        finally:
            os.unlink(path)

        self.active_idx = [
            i for i, on in enumerate(self.chain.active_links_mask) if on
        ]
        self.get_logger().info(
            f"chain loaded, {len(self.active_idx)} DOF, "
            f"tip = {self.chain.links[-1].name}"
        )

        self.q = np.array(HOME, dtype=float)      # where we are
        self.q_target = self.q.copy()             # where we are heading
        self.step = np.zeros(6)                   # per-tick increment

        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.pub_marker = self.create_publisher(Marker, "goal_marker", 10)
        self.create_subscription(PoseStamped, "arm_goal", self.on_goal, 10)
        self.create_subscription(Empty, "arm_home", self.on_home, 10)
        self.create_timer(1.0 / RATE_HZ, self.tick)

        self.get_logger().info("ready. publish a PoseStamped to /arm_goal")

    # ------------------------------------------------------------------

    def _full(self, q6: np.ndarray) -> list:
        """Expand 6 joint values into the full ikpy chain vector."""
        vec = [0.0] * len(self.chain.links)
        for slot, value in zip(self.active_idx, q6):
            vec[slot] = value
        return vec

    def on_goal(self, msg: PoseStamped) -> None:
        target = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )

        # Seeding with the current pose matters. A 6-DOF arm has multiple valid
        # solutions for any reachable point. Without a seed the solver picks a
        # different one each call and the arm flails between elbow-up and
        # elbow-down. With a seed it stays near where it already is.
        solution = self.chain.inverse_kinematics(
            target, initial_position=self._full(self.q)
        )
        q_new = np.array([solution[i] for i in self.active_idx])

        reached = self.chain.forward_kinematics(solution)[:3, 3]
        error_mm = float(np.linalg.norm(reached - target) * 1000.0)

        if error_mm > 5.0:
            self.get_logger().warn(
                f"unreachable: asked for {np.round(target, 3)}, "
                f"best effort is off by {error_mm:.0f} mm. Ignoring."
            )
            return

        self.q_target = q_new
        travel = float(np.max(np.abs(q_new - self.q)))
        ticks = max(1, int(RATE_HZ * travel / MAX_JOINT_SPEED))
        self.step = (q_new - self.q) / ticks

        self.get_logger().info(
            f"goal {np.round(target, 3)}  err {error_mm:.2f} mm  "
            f"{ticks / RATE_HZ:.1f} s"
        )
        self.publish_marker(target, msg.header.frame_id or "world")

    def on_home(self, _msg: Empty) -> None:
        """Return to the rest pose.

        Note what is NOT happening here: no inverse kinematics. Home is
        defined as six joint angles, not as a point in space, so the target
        is already in the form the arm needs. Solving for it would be slower,
        approximate, and could land in a different elbow configuration.

        Every other motion in this system is Cartesian and needs IK. This one
        is joint space and does not. Worth knowing which is which.
        """
        target = np.array(HOME, dtype=float)
        travel = float(np.max(np.abs(target - self.q)))
        if travel < 1e-3:
            self.get_logger().info("already home")
            return

        ticks = max(1, int(RATE_HZ * travel / MAX_JOINT_SPEED))
        self.q_target = target
        self.step = (target - self.q) / ticks
        self.get_logger().info(f"homing, {ticks / RATE_HZ:.1f} s (no IK needed)")
        self.clear_marker()

    def clear_marker(self) -> None:
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "arm_goal"
        m.id = 0
        m.action = Marker.DELETE
        self.pub_marker.publish(m)

    def tick(self) -> None:
        remaining = self.q_target - self.q
        if np.any(np.abs(remaining) > 1e-4):
            move = np.where(
                np.abs(remaining) < np.abs(self.step), remaining, self.step
            )
            self.q = self.q + move

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = self.q.tolist()
        self.pub_js.publish(msg)

    def publish_marker(self, xyz: np.ndarray, frame: str) -> None:
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "arm_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, xyz)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.3, 0.9
        self.pub_marker.publish(m)


def main() -> None:
    rclpy.init()
    node = ArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

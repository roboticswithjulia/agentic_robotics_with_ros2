#!/usr/bin/env python3
"""
poke -- send the arm somewhere, from the terminal.

  ros2 run agentic_arm poke 0.5 0.15 0.10
  ros2 run agentic_arm poke --demo        # runs a loop of four poses
  ros2 run agentic_arm poke --home        # return to the rest pose

Exists so you can confirm the arm moves before any language model is involved.
Debug one layer at a time.
"""

import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from rclpy.node import Node

DEMO_POSES = [
    (0.50, 0.15, 0.10),
    (0.45, -0.20, 0.08),
    (0.60, 0.00, 0.25),
    (0.35, 0.30, 0.15),
]


class Poker(Node):
    def __init__(self) -> None:
        super().__init__("poke")
        self.pub = self.create_publisher(PoseStamped, "arm_goal", 10)
        self.pub_home = self.create_publisher(Empty, "arm_home", 10)

    def home(self) -> None:
        self.pub_home.publish(Empty())
        self.get_logger().info("sent home")

    def send(self, x: float, y: float, z: float) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.pub.publish(msg)
        self.get_logger().info(f"sent {x} {y} {z}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("__")]

    rclpy.init()
    node = Poker()
    time.sleep(0.5)  # let the publisher find its subscriber

    try:
        if args and args[0] == "--home":
            node.home()
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.1)
        elif not args or args[0] == "--demo":
            for pose in DEMO_POSES:
                node.send(*pose)
                for _ in range(30):
                    rclpy.spin_once(node, timeout_sec=0.1)
        else:
            node.send(*[float(a) for a in args[:3]])
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

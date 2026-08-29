"""
Phase 6. Robot only. No language model involved.

  ros2 launch agentic_arm display.launch.py

Starts three things:
  robot_state_publisher  reads /joint_states + URDF, publishes /tf
  arm_node               solves IK, publishes /joint_states
  rviz2                  draws /tf
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("agentic_arm")
    xacro_file = os.path.join(pkg, "urdf", "ur5e_gripper.urdf.xacro")
    rviz_file = os.path.join(pkg, "config", "arm.rviz")

    # Processed here rather than with a Command() substitution. Slower to
    # launch, but xacro errors come out readable instead of buried in a
    # substitution failure. Worth it when twenty people are debugging at once.
    robot_description = xacro.process_file(xacro_file).toxml()

    rviz_args = ["-d", rviz_file] if os.path.exists(rviz_file) else []

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="agentic_arm",
            executable="arm_node",
            name="arm_node",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=rviz_args,
        ),
    ])

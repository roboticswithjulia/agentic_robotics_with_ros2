"""
The whole pipeline, one command.

  ros2 launch agentic_arm full.launch.py

Graph:

  scene_node --/world_state--> executor_node
                                    ^
  you --/instruction--> planner_node --/task_plan--> executor_node
                        (Ollama HTTP)                     |
                                                      /arm_goal
                                                          v
                                                      arm_node  (ikpy)
                                                          |
                                                    /joint_states
                                                          v
                                         robot_state_publisher --/tf--> RViz

Options:
  ros2 launch agentic_arm full.launch.py model:=phi4-mini
  ros2 launch agentic_arm full.launch.py rviz:=false
  ros2 launch agentic_arm full.launch.py scene_seed:=42
  ros2 launch agentic_arm full.launch.py return_home:=true
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("agentic_arm")
    xacro_file = os.path.join(pkg, "urdf", "ur5e_gripper.urdf.xacro")
    rviz_file = os.path.join(pkg, "config", "arm.rviz")

    # Processed here rather than with a Command() substitution, so xacro
    # errors come out readable instead of buried in a substitution failure.
    robot_description = xacro.process_file(xacro_file).toxml()

    model = LaunchConfiguration("model")
    scene_seed = LaunchConfiguration("scene_seed")
    return_home = LaunchConfiguration("return_home")
    use_rviz = LaunchConfiguration("rviz")

    rviz_args = ["-d", rviz_file] if os.path.exists(rviz_file) else []

    return LaunchDescription([
        # Strip the timestamp and severity prefix. With five nodes logging at
        # once the default format is mostly noise, and the pipeline is much
        # easier to follow when each line is just [node] message.
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT",
                               "{message}"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),

        DeclareLaunchArgument("model", default_value="qwen3.5:4b"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("scene_seed", default_value="7"),
        DeclareLaunchArgument("return_home", default_value="false"),

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
        # Whoever publishes /world_state owns the world. In Phase 2 this
        # one node is replaced by the camera and perception nodes, and
        # nothing else in this file changes.
        Node(
            package="agentic_arm",
            executable="scene_node",
            name="scene_node",
            output="screen",
            parameters=[{"scene_seed": scene_seed}],
        ),
        Node(
            package="agentic_arm",
            executable="executor_node",
            name="executor_node",
            output="screen",
            parameters=[{"return_home": return_home}],
        ),
        Node(
            package="agentic_arm",
            executable="planner_node",
            name="planner_node",
            output="screen",
            parameters=[{"model": model}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(use_rviz),
            arguments=rviz_args,
        ),
    ])

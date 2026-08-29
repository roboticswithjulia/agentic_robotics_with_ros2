"""
Phase 2. The same system, but the world comes from a camera.

    ros2 launch agentic_arm perception.launch.py

NOTHING FROM PHASE 1 IS EDITED

This is a second launch file sitting alongside full.launch.py. Every node it
starts is the same node Phase 1 starts, with the same code. Two new ones are
added, and one existing one is rewired without being touched.

THE REWIRE

scene_node normally publishes /world_state, which the executor believes. In
Phase 2 the perception node must publish /world_state instead, so the two
would collide.

The fix is a remap, done here in the launch file:

    scene_node's "world_state"  ->  "scene_truth"

A remap renames a node's topics from the outside. scene_node still calls it
world_state in its own code and has no idea anything changed. This is one of
the genuinely useful things about ROS: you can rewire a graph without
touching the programs in it.

THE GRAPH

    scene_node --/scene_truth--> camera_node
                                      |
                                      |  /camera/color/image_raw
                                      |  /camera/depth/image_rect_raw
                                      |  /camera/color/camera_info
                                      v
                                perception_node
                                      |
                                      |  /world_state
                                      v
    you --/instruction--> planner_node --/task_plan--> executor_node
                                                            |
                                                        /arm_goal
                                                            v
                                                        arm_node

Compare that to Phase 1, where scene_node published /world_state directly.
Everything below /world_state is identical in both.

WHAT IS SIMULATED AND WHAT IS NOT

Simulated: the images. camera_node calculates what a camera would see, given
where the objects are. That is the only made up part.

Real: everything perception_node does with those images. Colour
segmentation, connected components, depth lookup, deprojection through the
camera intrinsics, and the transform into the world frame. Point a RealSense
at a table and the same code runs unchanged.

OPTIONS

    ros2 launch agentic_arm perception.launch.py scene_seed:=42
    ros2 launch agentic_arm perception.launch.py rviz:=false
    ros2 launch agentic_arm perception.launch.py model:=phi4-mini
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

    robot_description = xacro.process_file(xacro_file).toxml()

    model = LaunchConfiguration("model")
    use_rviz = LaunchConfiguration("rviz")
    scene_seed = LaunchConfiguration("scene_seed")
    return_home = LaunchConfiguration("return_home")

    rviz_args = ["-d", rviz_file] if os.path.exists(rviz_file) else []

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "{message}"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),

        DeclareLaunchArgument("model", default_value="qwen3.5:4b"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("scene_seed", default_value="7"),
        DeclareLaunchArgument("return_home", default_value="false"),

        # ---- unchanged from Phase 1 ----
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

        # ---- rewired, not edited ----
        # Same node as Phase 1, same code. Its /world_state is renamed to
        # /scene_truth so it feeds the camera instead of the executor. It
        # still draws the markers you see in RViz, which are now the ground
        # truth against which perception can be judged.
        Node(
            package="agentic_arm",
            executable="scene_node",
            name="scene_node",
            output="screen",
            parameters=[{"scene_seed": scene_seed}],
            remappings=[
                # Its output feeds the camera, not the executor.
                ("world_state", "scene_truth"),
                # And it listens to what perception decided, so it can match
                # perception's names onto its own by position. This second
                # rule is why the subscription is not itself called
                # world_state: one rule would have caught both.
                ("perceived_world", "world_state"),
            ],
        ),

        # ---- new in Phase 2 ----
        Node(
            package="agentic_arm",
            executable="camera_node",
            name="camera_node",
            output="screen",
        ),
        Node(
            package="agentic_arm",
            executable="perception_node",
            name="perception_node",
            output="screen",
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

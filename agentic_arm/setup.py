import os
from glob import glob

from setuptools import find_packages, setup

package_name = "agentic_arm"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        # Without these three lines, launch/urdf/config are NOT installed.
        # The build still succeeds and ros2 launch then cannot find the file.
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ashish",
    maintainer_email="adg@zupt.com",
    description="Agentic Robotics with ROS 2 workshop package.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "arm_node = agentic_arm.arm_node:main",
            "executor_node = agentic_arm.executor_node:main",
            "scene_node = agentic_arm.scene_node:main",            
            "planner_node = agentic_arm.planner_node:main",
            "poke = agentic_arm.poke:main",
            "camera_node = agentic_arm.camera_node:main",
            "perception_node = agentic_arm.perception_node:main",
            "monitor = agentic_arm.monitor:main",
        ],
    },
)

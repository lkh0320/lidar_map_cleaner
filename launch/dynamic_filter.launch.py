"""Real-time moving-object filter.

  ros2 launch lidar_map_cleaner dynamic_filter.launch.py cloud:=/fastlio2/body_cloud odom:=/fastlio2/lio_odom
  -> /cloud_static (feed this to octomap_server / your PGO / mapping back-end), /cloud_dynamic (removed points)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('lidar_map_cleaner'), 'config', 'dynamic_filter.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('cloud', default_value='/fastlio2/body_cloud', description='body-frame input cloud'),
        DeclareLaunchArgument('odom', default_value='/fastlio2/lio_odom', description='odometry of the same body frame'),
        DeclareLaunchArgument('static', default_value='/cloud_static'),
        DeclareLaunchArgument('dynamic', default_value='/cloud_dynamic'),
        DeclareLaunchArgument('params', default_value=cfg),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='lidar_map_cleaner', executable='dynamic_filter', name='dynamic_filter', output='screen',
             parameters=[LaunchConfiguration('params'), {'use_sim_time': LaunchConfiguration('use_sim_time')}],
             remappings=[('~/cloud_in', LaunchConfiguration('cloud')), ('~/odom_in', LaunchConfiguration('odom')),
                         ('~/static', LaunchConfiguration('static')), ('~/dynamic', LaunchConfiguration('dynamic'))]),
    ])

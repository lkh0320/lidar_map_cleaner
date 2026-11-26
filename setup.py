from glob import glob
import os

from setuptools import setup

package_name = 'lidar_map_cleaner'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lkh0320',
    maintainer_email='lkh060320@gmail.com',
    description='Moving-object removal and clean map layers for 3D LiDAR SLAM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'dynamic_filter = lidar_map_cleaner.dynamic_filter_node:main',
        'remove_dynamic = lidar_map_cleaner.remove_dynamic:main',
        'build_layers = lidar_map_cleaner.build_layers:main',
        'elevation_mapper = lidar_map_cleaner.elevation_mapper:main',
        'merge_maps = lidar_map_cleaner.merge_maps:main',
        'clean_floor = lidar_map_cleaner.clean_floor:main',
    ]},
)

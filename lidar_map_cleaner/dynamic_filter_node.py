"""ROS 2 node: removes moving objects from a body-frame point cloud in real time.

  in : ~/cloud_in  (sensor_msgs/PointCloud2, body frame, e.g. FAST-LIO /fastlio2/body_cloud or /cloud_registered_body)
       ~/odom_in   (nav_msgs/Odometry, pose of the same body frame, same stamps as the cloud)
  out: ~/static    (input cloud minus dynamic points, all fields preserved)
       ~/dynamic   (removed points, only published when subscribed)
Remap the topics in your launch file. Feed ~/static to your mapping back-end (octomap, PGO, ...).
"""
import time
from collections import deque

import message_filters
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import PointCloud2

from .dynamic_core import DEFAULTS, DynamicCore
from .sensor_model import PRESETS, SensorModel


def cloud_xyz(msg):
    """PointCloud2 (float32 x,y,z) -> ((N,3) float64, raw rows (N, point_step) uint8)"""
    off = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    out = np.empty((n, 3), np.float64)
    for k, name in enumerate('xyz'):
        out[:, k] = raw[:, off[name]:off[name] + 4].copy().view(np.float32).ravel()
    return out, raw


def cloud_rows(src, rows):
    m = PointCloud2()
    m.header = src.header
    m.height, m.width = 1, len(rows)
    m.fields, m.is_bigendian, m.point_step = src.fields, src.is_bigendian, src.point_step
    m.row_step, m.is_dense = src.point_step * len(rows), src.is_dense
    m.data = rows.tobytes()
    return m


def sensor_from_params(node):
    p = node.declare_parameter
    preset = p('sensor.preset', 'vlp16').value
    base = dict(PRESETS.get(preset, {}))
    if preset and preset not in PRESETS:
        node.get_logger().warn(f'unknown sensor.preset "{preset}", known: {sorted(PRESETS)}')
    for k in ('v_min', 'v_max', 'v_res'):
        v = p(f'sensor.{k}', float(base.get(k, getattr(SensorModel, k)))).value
        base[k] = v
    base['az_res'] = p('sensor.az_res', 1.0).value
    base['el_tol'] = p('sensor.el_tol', 0.3).value
    base['xyz'] = list(p('sensor.xyz', [0.0, 0.0, 0.0]).value)
    base['rpy'] = list(p('sensor.rpy', [0.0, 0.0, 0.0]).value)
    return SensorModel(**base)


class DynamicFilterNode(Node):
    def __init__(self):
        super().__init__('dynamic_filter')
        sensor = sensor_from_params(self)
        kw = {k: self.declare_parameter(k, v).value for k, v in DEFAULTS.items()}
        self.core = DynamicCore(sensor, **kw)
        slop = self.declare_parameter('sync_slop', 0.02).value
        self.stats = deque(maxlen=200)
        self.pub_static = self.create_publisher(PointCloud2, '~/static', 10)
        self.pub_dyn = self.create_publisher(PointCloud2, '~/dynamic', 10)
        sc = message_filters.Subscriber(self, PointCloud2, '~/cloud_in')
        so = message_filters.Subscriber(self, Odometry, '~/odom_in')
        self.sync = message_filters.ApproximateTimeSynchronizer([sc, so], 20, slop)
        self.sync.registerCallback(self.cb)
        self.create_timer(10.0, self.log)
        self.get_logger().info(f'sensor {sensor.rows}x{sensor.cols} range image, window {kw["window"]} s / {kw["n_past"]} scans')

    def cb(self, cloud, odom):
        t0 = time.perf_counter()
        st = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9
        q, p = odom.pose.pose.orientation, odom.pose.pose.position
        pb, raw = cloud_xyz(cloud)
        dyn = self.core.process(st, Rot.from_quat([q.x, q.y, q.z, q.w]), np.array([p.x, p.y, p.z]), pb)
        self.pub_static.publish(cloud_rows(cloud, raw[~dyn]))
        if self.pub_dyn.get_subscription_count() > 0:
            self.pub_dyn.publish(cloud_rows(cloud, raw[dyn]))
        self.stats.append(((time.perf_counter() - t0) * 1000, int(dyn.sum()), len(pb)))

    def log(self):
        if self.stats:
            a = np.array(self.stats)
            self.get_logger().info(f'{a[:, 0].mean():.1f} ms/scan (max {a[:, 0].max():.1f}), '
                                   f'removed {100 * a[:, 1].sum() / max(a[:, 2].sum(), 1):.2f} % of points')


def main():
    rclpy.init()
    node = DynamicFilterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

"""ROS 2 node: 2.5D elevation map -> grid_map_msgs/GridMap (layers: elevation / min / count), latched.

source parameter:
  octomap (default): floor and obstacles are handled separately
      - floor (points below robot floor + floor_split): accumulated from the input cloud
      - obstacles (above): recomputed from octomap_server occupied-voxel centers every time they arrive.
        OctoMap clears voxels that beams pass through, so traces of people who walked by disappear online.
      Using OctoMap for the floor too fails: grazing beams clear floor voxels and more than half of the floor
      cells vanish. With the default octomap_server config (occupancy band 0.15-0.8 m) obstacles taller than
      0.8 m are stored as ~0.75 m, which does not matter for traversability.
  cloud: accumulate the max height per cell from the input cloud (light, but moving objects leave traces).
Points higher than max_height_above_robot (ceilings) are dropped. Clouds are placed with the TF available at
insertion time, so cells are not re-aligned after a loop closure (same as octomap_server); rebuild offline
with build_layers for loop-corrected layers.

grid_map storage: index (i, j), i along x (largest x = 0), j along y (largest y = 0), column-major data.
"""
from array import array
from collections import deque

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener


def _transform_to_matrix(tf):
    q = tf.transform.rotation
    t = tf.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return r, np.array([t.x, t.y, t.z])


class ElevationMapper(Node):
    def __init__(self):
        super().__init__('elevation_mapper')
        p = self.declare_parameter
        self.source = p('source', 'octomap').value
        self.cloud_topic = p('cloud_topic', '/cloud_in').value
        self.octomap_topic = p('octomap_topic', '/octomap_point_cloud_centers').value
        self.floor_split = float(p('floor_split', 0.15).value)   # octomap mode: below robot floor + this = floor (accumulated)
        self.map_frame = p('map_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_footprint').value
        self.res = float(p('resolution', 0.1).value)
        self.len_x = float(p('length_x', 40.0).value)
        self.len_y = float(p('length_y', 40.0).value)
        self.max_range = float(p('max_range', 15.0).value)
        self.max_h = float(p('max_height_above_robot', 1.2).value)
        self.min_h = float(p('min_height_above_robot', -1.5).value)
        self.min_points = int(p('min_points', 2).value)
        rate = float(p('publish_rate', 2.0).value)
        self.tf_wait = float(p('tf_wait', 1.0).value)  # drop a cloud if its TF does not arrive within this time

        self.rows = int(round(self.len_x / self.res))
        self.cols = int(round(self.len_y / self.res))
        self.len_x = self.rows * self.res
        self.len_y = self.cols * self.res
        self._reset()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, qos_profile_sensor_data)
        if self.source == 'octomap':
            # octomap_server publishes transient_local when latched -> match it to also get the last map
            self.sub_oct = self.create_subscription(PointCloud2, self.octomap_topic, self._on_octomap, QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # latched so the last map is still available after the bag / SLAM stops
        self.pub = self.create_publisher(GridMap, '~/elevation_map', QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_service(Empty, '~/reset', self._on_reset)
        # wall-clock timers: keep publishing even when /clock stops at the end of a bag
        wall = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.0 / rate, self._publish, clock=wall)
        self.published_clouds = -1
        self.queue = deque(maxlen=50)
        self.create_timer(0.02, self._process_queue, clock=wall)
        self.n_dropped = 0
        self.last_stamp = None
        self.n_clouds = 0
        src = self.octomap_topic + ' (octomap mode: moving-object traces cleared online)' if self.source == 'octomap' else self.cloud_topic
        self.get_logger().info(f'{src} -> ~/elevation_map  {self.rows}x{self.cols} cells @ {self.res} m')

    def _reset(self):
        n = (self.rows, self.cols)
        # accumulated layer (cloud mode: everything, octomap mode: floor only)
        self.elev = np.full(n, -np.inf, dtype=np.float32)
        self.zmin = np.full(n, np.inf, dtype=np.float32)
        self.count = np.zeros(n, dtype=np.float32)
        # octomap obstacle layer (rebuilt on every octomap message)
        self.o_elev = np.full(n, -np.inf, dtype=np.float32)
        self.o_zmin = np.full(n, np.inf, dtype=np.float32)
        self.o_count = np.zeros(n, dtype=np.float32)

    def _on_reset(self, req, res):
        self._reset()
        self.get_logger().info('elevation map reset')
        return res

    def _on_cloud(self, msg):
        # the map <- cloud TF may arrive slightly later (e.g. from PGO), so queue instead of blocking
        self.queue.append(msg)

    def _on_octomap(self, msg):
        """recompute the obstacle layer from all occupied voxels (cleared voxels disappear with them)"""
        if msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f'octomap frame {msg.header.frame_id} != {self.map_frame}', throttle_duration_sec=10.0)
            return
        try:
            tf_robot = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, Time())
            robot = np.array([tf_robot.transform.translation.x, tf_robot.transform.translation.y,
                              tf_robot.transform.translation.z])
        except TransformException:
            robot = np.zeros(3)
        w = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True).astype(np.float64)
        n = (self.rows, self.cols)
        self.o_elev = np.full(n, -np.inf, dtype=np.float32)
        self.o_zmin = np.full(n, np.inf, dtype=np.float32)
        self.o_count = np.zeros(n, dtype=np.float32)
        if w.size:
            dz = w[:, 2] - robot[2]
            w = w[(dz <= self.max_h) & (dz >= self.floor_split)]
            self._accumulate(w, self.o_elev, self.o_zmin, self.o_count)
        self.n_clouds += 1

    def _accumulate(self, w, elev=None, zmin=None, count=None):
        elev = self.elev if elev is None else elev
        zmin = self.zmin if zmin is None else zmin
        count = self.count if count is None else count
        # grid centred on the map origin. i: largest x = 0, j: largest y = 0
        i = np.floor((self.len_x / 2 - w[:, 0]) / self.res).astype(np.int64)
        j = np.floor((self.len_y / 2 - w[:, 1]) / self.res).astype(np.int64)
        inside = (i >= 0) & (i < self.rows) & (j >= 0) & (j < self.cols)
        i, j, z = i[inside], j[inside], w[inside, 2].astype(np.float32)
        np.maximum.at(elev, (i, j), z)
        np.minimum.at(zmin, (i, j), z)
        np.add.at(count, (i, j), 1.0)

    def _process_queue(self):
        while self.queue:
            msg = self.queue[0]
            stamp = Time.from_msg(msg.header.stamp)
            if self.tf_buffer.can_transform(self.map_frame, msg.header.frame_id, stamp) and \
                    self.tf_buffer.can_transform(self.map_frame, self.robot_frame, stamp):
                self.queue.popleft()
                self._integrate(msg, stamp)
                continue
            age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            if age > self.tf_wait:
                self.queue.popleft()
                self.n_dropped += 1
                self.get_logger().warn(f'no TF for {age:.2f} s, dropping cloud (total {self.n_dropped})',
                                       throttle_duration_sec=2.0)
                continue
            break

    def _integrate(self, msg, stamp):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, stamp)
            tf_robot = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, stamp)
        except TransformException as e:
            self.get_logger().warn(f'no TF: {e}', throttle_duration_sec=2.0)
            return
        pts = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts.size == 0:
            return
        r, t = _transform_to_matrix(tf)
        w = pts.astype(np.float64) @ r.T + t
        robot = np.array([tf_robot.transform.translation.x, tf_robot.transform.translation.y,
                          tf_robot.transform.translation.z])

        dz = w[:, 2] - robot[2]
        d2 = (w[:, 0] - robot[0]) ** 2 + (w[:, 1] - robot[1]) ** 2
        top = self.floor_split if self.source == 'octomap' else self.max_h   # octomap mode: accumulate the floor only
        keep = (dz < top) & (dz >= self.min_h) & (d2 <= self.max_range ** 2)
        self._accumulate(w[keep])
        self.last_stamp = msg.header.stamp
        self.n_clouds += 1

    def _layer(self, arr):
        m = Float32MultiArray()
        m.layout.dim = [
            MultiArrayDimension(label='column_index', size=self.cols, stride=self.rows * self.cols),
            MultiArrayDimension(label='row_index', size=self.rows, stride=self.rows),
        ]
        m.data = array('f', np.asarray(arr, dtype=np.float32).ravel(order='F').tobytes())
        return m

    def _publish(self):
        if self.last_stamp is None:
            return
        floor_ok = self.count >= self.min_points
        obs = self.o_count > 0
        valid = floor_ok | obs
        elev = np.where(obs, np.maximum(self.o_elev, np.where(floor_ok, self.elev, -np.inf)), self.elev)
        zmin = np.where(floor_ok, self.zmin, self.o_zmin)
        cnt = np.where(floor_ok, self.count, 0) + self.o_count
        self.get_logger().info(f'{self.n_clouds} clouds integrated, {int(valid.sum())} valid cells, '
                               f'queued {len(self.queue)}, dropped {self.n_dropped}', throttle_duration_sec=10.0)
        if self.n_clouds == self.published_clouds:   # unchanged: do not resend (latched)
            return
        self.published_clouds = self.n_clouds
        nan = np.float32(np.nan)
        msg = GridMap()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.last_stamp
        msg.info.resolution = self.res
        msg.info.length_x = self.len_x
        msg.info.length_y = self.len_y
        msg.info.pose.orientation.w = 1.0
        msg.layers = ['elevation', 'min', 'count']
        msg.basic_layers = ['elevation']
        msg.data = [
            self._layer(np.where(valid, elev, nan)),
            self._layer(np.where(valid, zmin, nan)),
            self._layer(np.where(valid, cnt, nan)),
        ]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ElevationMapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

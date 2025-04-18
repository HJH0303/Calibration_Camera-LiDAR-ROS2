#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import struct
from message_filters import Subscriber, ApproximateTimeSynchronizer

def pointcloud2_to_xyz(pc_msg):
    """
    Convert a ROS2 PointCloud2 message to a (N, 3) numpy array.
    """
    point_count = pc_msg.width * pc_msg.height
    data = pc_msg.data
    points = []
    # Assumes fields are ordered as x, y, z
    x_offset = pc_msg.fields[0].offset
    y_offset = pc_msg.fields[1].offset
    z_offset = pc_msg.fields[2].offset
    point_step = pc_msg.point_step
    for i in range(point_count):
        base = i * point_step
        x = struct.unpack_from('f', data, base + x_offset)[0]
        y = struct.unpack_from('f', data, base + y_offset)[0]
        z = struct.unpack_from('f', data, base + z_offset)[0]
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            points.append([x, y, z])
    if not points:
        return None
    return np.array(points, dtype=np.float32)

class RealTimePointCloudProjectionNode(Node):
    def __init__(self):
        super().__init__('real_time_projection_node')
        self.bridge = CvBridge()
        self.latest_image = None
        rot_matrix = np.array([
            [9.99559090e-01, -1.02412900e-02,  2.78699500e-02],
            [1.07520400e-02,  9.99775850e-01, -1.82387500e-02],
            [-2.76769100e-02, 1.85303700e-02,  9.99445150e-01]
        ], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(rot_matrix)
        self.rvec = rvec
        self.tvec = np.array([[0.05047155], [-0.14259434], [-0.34365417]], dtype=np.float32)
        self.K = np.array([
            [2.69978350e+02, 0.00000000e+00, 3.20896825e+02],
            [0.00000000e+00, 2.70064445e+02, 1.88387816e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([[0.005504, 0.000630, -0.002514, 0.001207, 0.000000]], dtype=np.float32)
        # self.rvec = np.array([[0.05815992], [0.05923345], [-0.01574279]], dtype=np.float32)
        # self.tvec = np.array([[0.03532316], [-0.08500311], [-0.31001697]], dtype=np.float32)

        self.R_lidar2cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ], dtype=np.float32)

        self.color_mode = 0

        self.pc_sub = Subscriber(self, PointCloud2, '/velodyne_points')
        self.img_sub = Subscriber(self, Image, '/zed/zed_node/rgb/image_rect_color')
        self.ats = ApproximateTimeSynchronizer([self.pc_sub, self.img_sub], queue_size=10, slop=0.1)
        self.ats.registerCallback(self.sync_callback)

    def sync_callback(self, pc_msg, img_msg):
        points = pointcloud2_to_xyz(pc_msg)
        if points is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            self.latest_image = image.copy()

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('0'), ord('1'), ord('2')]:
                self.color_mode = int(chr(key))
                self.get_logger().info(f"Changed color mode to {['X','Y','Z'][self.color_mode]}")

            overlay_img = self.overlay_lidar_on_image(self.latest_image, points)
            cv2.imshow("Projected Point Cloud", overlay_img)
        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {e}")

    def overlay_lidar_on_image(self, img, lidar_points):
        valid_mask = lidar_points[:, 0] > 0
        pts = lidar_points[valid_mask]
        if pts.shape[0] == 0:
            return img

        pts_transformed = (self.R_lidar2cam @ pts.T).T

        image_points, _ = cv2.projectPoints(pts_transformed, self.rvec, self.tvec, self.K, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2).astype(np.int32)

        h, w = img.shape[:2]
        valid = (image_points[:, 0] >= 0) & (image_points[:, 0] < w) & (image_points[:, 1] >= 0) & (image_points[:, 1] < h)
        u_valid = image_points[valid, 0]
        v_valid = image_points[valid, 1]
        pts_valid = pts_transformed[valid]

        channel_values = pts_valid[:, self.color_mode]
        vmin, vmax = np.min(channel_values), np.max(channel_values)
        norm_vals = ((channel_values - vmin) / (vmax - vmin + 1e-6) * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(norm_vals.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = colormap.reshape(-1, 3)

        out_img = img.copy()
        for (u, v), color in zip(zip(u_valid, v_valid), colors):
            cv2.circle(out_img, (u, v), 2, (int(color[0]), int(color[1]), int(color[2])), -1)

        return out_img

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePointCloudProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

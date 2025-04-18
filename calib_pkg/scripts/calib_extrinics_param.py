#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys
import struct
import numpy as np
import open3d as o3d
import cv2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from collections import defaultdict

from message_filters import Subscriber, ApproximateTimeSynchronizer
sys.path.append("/root/calib_ws/src/calib_pkg/scripts")
from utils_3d_lidar import *
from utils_checker_img_3d import *
from utils_calib import *
class Calib_lidar2camera(Node):
    def __init__(self):
        super().__init__('calibration_node')
        self.bridge = CvBridge()
        self.frame_count = 0
        # Create message_filters subscribers for pointcloud + image
        self.pc_sub = Subscriber(self, PointCloud2, '/velodyne_points')
        self.img_sub = Subscriber(self, Image, '/zed/zed_node/rgb/image_rect_color')

        # ApproximateTimeSynchronizer -> sync with lidar and image
        self.ats = ApproximateTimeSynchronizer([self.pc_sub, self.img_sub],
                                                queue_size=10,
                                                slop=0.01)
        self.ats.registerCallback(self.sync_callback)
        ######################################
        ###### checker board parameters ######
        ######################################
        self.pattern_size = (8, 6)  # (rows, cols) internal corners
        self.square_size = 0.12     # Example: 0.12 meters per square
        # Define padding: [pad_left, pad_top, pad_right, pad_bottom] (check git usage.md)
        self.padding = [0.194, 0.155, 0.185, 0.155]

        # your camera intrinics parameters
        self.K = np.array([[269.978350, 0.0, 320.896825],
                           [0.0, 270.064445, 188.387816],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([0.005504, 0.000630, -0.002514, 0.001207, 0.000000], dtype=np.float32)


        # Lists to collect constraints data over N poses
        self.collected_edge_constraints = []  # each element: (Q_ijk_L, P_ij_C, d_ij_C) in Camera frame
        self.collected_plane_constraints = [] # each element: (P_im_L, n_i_C, d_i_C) in Camera frame
        # For rotation optimization, collect corresponding edge and plane vectors
        self.all_lidar_edge_dirs = []       # from LiDAR (after transformation)
        self.all_camera_edge_dirs = []      # from camera (img_edge_directions)
        self.all_lidar_plane_normals = []    # from LiDAR (transformed plane normal)
        self.all_camera_plane_normals = []   # from camera (img_plane_params[:3])

        self.get_logger().info("SyncLinePlotNode initialized with approximate time sync.")

    def finalize_calibration(self):
        # Aggregate all rotation constraints
        if (len(self.all_lidar_edge_dirs) == 0 or len(self.all_camera_edge_dirs) == 0 or
            len(self.all_lidar_plane_normals) == 0 or len(self.all_camera_plane_normals) == 0):
            self.get_logger().info("Insufficient rotation constraints collected!")
            return

        vectors_lidar = np.vstack(self.all_lidar_edge_dirs + self.all_lidar_plane_normals)
        vectors_camera = np.vstack(self.all_camera_edge_dirs + self.all_camera_plane_normals)
        R_opt_prior = estimate_rotation_from_edges_and_normal(vectors_lidar, vectors_camera)

        # Aggregate all translation constraints
        if len(self.collected_edge_constraints) == 0 and len(self.collected_plane_constraints) == 0:
            self.get_logger().info("No translation constraints collected!")
            return
        t_opt_prior = solve_translation(R_opt_prior, self.collected_edge_constraints, self.collected_plane_constraints)
    
        self.get_logger().info("Final Calibration Results:")
        self.get_logger().info("Rotation (R_opt_prior):\n" + str(R_opt_prior))
        self.get_logger().info("Translation (t_opt_prior):\n" + str(t_opt_prior))
        R_opt, t_opt, lm_info = run_lm_optimization(self.collected_plane_constraints, self.collected_edge_constraints, R_opt_prior, t_opt_prior)
    
        print("===== LM Joint Optimization results =====")
        print("Optimized Rotation (R_opt):\n", R_opt)
        print("Optimized Translation (t_opt):\n", t_opt)
        # print("Optimization Details:\n", lm_info)

        # Optionally, save the calibration results to file or publish them.

    def sync_callback(self, pc_msg, img_msg):
        # 1) Convert point cloud
        pts_arr, rings_arr = read_points(pc_msg)
        if pts_arr.shape[0] == 0:
            self.get_logger().info("No valid points. Skipping.")
            return

        # 2) Process image and find chessboard corners
        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        display_img = frame.copy()
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        print(f"collected frame :{self.frame_count}")
        cv2.imshow("check the frame(skip: press the key 's', add the data : press the any key ,calibration: press the key 'c' )", display_img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("check the frame")

        if key == ord('s'):
            self.get_logger().info("User skipped this frame.")
            return
        if key == ord('c'):
            self.get_logger().info("User pressed 'c'. Finalizing calibration with collected poses.")
            
            self.finalize_calibration()
            return

        # Chessboard detection and 3D corner extraction
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        checker_img = cv2.drawChessboardCorners(display_img.copy(), self.pattern_size, corners_refined, ret)
        self.get_logger().info("Checkerboard detected.")
        undistorted = cv2.undistortPoints(corners_refined, self.K, self.dist_coeffs, None, self.K)
        undistorted_points = undistorted.reshape(-1, 2)
        try:
            world_pts = generate_checkerboard_plane(self.pattern_size, self.square_size)
            extracted_3d, valid = extract_corners(undistorted_points, world_pts, self.pattern_size,
                                                    self.padding, self.K, True, display_img.copy(),self.dist_coeffs)
            # self.get_logger().info(f"Extracted 3D corners (camera frame):\n{extracted_3d}")
            img_edge_directions, img_plane_params = compute_plane_and_edge_parameters(extracted_3d)
            print(extracted_3d)
            cv2.imshow("Checkerboard Detection", checker_img)
            cv2.waitKey(0)
        except Exception as e:
            self.get_logger().error(f"Error in extract_corners: {e}")
            return

        # LiDAR plane segmentation
        plane_model, inliers, projected_arr, inlier_rings = plane_segmentation_and_projection(pts_arr, rings_arr)
        if plane_model is None:
            self.get_logger().info("Could not find plane. Skipping.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_arr)
        colors = np.tile([0.6, 0.6, 0.6], (pts_arr.shape[0], 1))
        for idx in inliers:
            colors[idx] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        proj_pcd = o3d.geometry.PointCloud()
        proj_pcd.points = o3d.utility.Vector3dVector(projected_arr)
        proj_pcd.paint_uniform_color([0, 1, 0])
        ring_dict = defaultdict(list)
        for pt, r in zip(projected_arr, inlier_rings):
            ring_dict[r].append(pt)
        boundary_points = []
        for r, lst in ring_dict.items():
            if len(lst) < 2:
                continue
            pts_sorted = sorted(lst, key=lambda p: p[0])
            boundary_points.append(pts_sorted[0])
            boundary_points.append(pts_sorted[-1])
        boundary_points = np.array(boundary_points)
        if boundary_points.shape[0] == 0:
            self.get_logger().info("No boundary points found. Skipping.")
            return
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
        boundary_pcd.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([pcd, proj_pcd, boundary_pcd])

        # 5) User selects boundary points for each edge and line fitting
        lines_info = []
        cylinders = []
        edge_boundary_points = {}
        boundary_np = np.asarray(boundary_pcd.points)
        for i in range(4):
            self.get_logger().info(f"Select boundary points for edge {i+1}. SHIFT+Click & Q to confirm.")
            picked_idx = pick_points_open3d(boundary_pcd, f"Edge {i+1}")
            if len(picked_idx) == 0:
                # self.get_logger().info(f"No points selected for edge {i+1}. Aborting.")
                return
            group_pts = boundary_np[picked_idx]
            edge_boundary_points[f"edge_{i+1}"] = group_pts
            c3d, d3d = fit_line_svd(group_pts)
            if c3d is None:
                # self.get_logger().info(f"Line fit failed for edge {i+1}. Aborting.")
                return
            lines_info.append((c3d, d3d))
            p1 = c3d + d3d * 10.0
            p2 = c3d - d3d * 10.0
            cyl = create_cylinder_line(p1, p2, radius=0.01, color=[1, 0.5, 0])
            cylinders.append(cyl)
        _, u, v = make_plane_basis(np.array(plane_model[:3]))
        plane_origin = np.mean([ln[0] for ln in lines_info], axis=0)
        lines_2d = []
        for (centroid, direction) in lines_info:
            vecC = centroid - plane_origin
            c2d = np.array([np.dot(vecC, u), np.dot(vecC, v)])
            d2d = np.array([np.dot(direction, u), np.dot(direction, v)])
            lines_2d.append((c2d, d2d))
        corner_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        corners_3d = []
        for (i, j) in corner_indices:
            inter2d = intersect_2d(lines_2d[i], lines_2d[j])
            if inter2d is None:
                self.get_logger().info(f"Lines {i} & {j} are parallel. Aborting.")
                return
            ix, iy = inter2d
            corner3d = plane_origin + ix * u + iy * v
            corners_3d.append(corner3d)
        spheres = [create_sphere(c3d, radius=0.05, color=[0.8, 0.2, 0.8]) for c3d in corners_3d]
        final_show = [pcd, proj_pcd, boundary_pcd] + cylinders + spheres
        o3d.visualization.draw_geometries(final_show)
        final_mapped_corners = np.array(corners_3d)


        # Compute LiDAR edge directions and plane parameters from mapped corners.
        lidar_edge_dirs, lidar_plane_params = compute_plane_and_edge_parameters(final_mapped_corners)
        transformed_corners, transformed_edge_dirs, transformed_plane_eq = transform_lidar_to_camera(final_mapped_corners, lidar_edge_dirs, lidar_plane_params)
        # Convert LiDAR boundary and plane points to Camera coordinates.
        rotation_matrix = get_lidar_to_camera_rotation()
        edge_boundary_points_cam = transform_edge_boundary_points(edge_boundary_points, rotation_matrix)
        projected_arr_cam = transform_projected_arr(projected_arr, rotation_matrix)
        
        # Build edge constraints from current pose (in Camera frame)
        edge_constraints_pose = []
        for i in range(4):
            key = f"edge_{i+1}"
            pts = edge_boundary_points_cam[key]
            centroid = np.mean(pts, axis=0)       # Q_ijk_L
            p_ij_C = extracted_3d[i]              # Corresponding camera edge intersection
            d_ij_C = img_edge_directions[i]       # Camera edge direction
            edge_constraints_pose.append((centroid, p_ij_C, d_ij_C))
        
        # Build plane constraints from current pose (in Camera frame)
        plane_constraints_pose = []
        n_i_C = img_plane_params[:3]
        d_i_C = img_plane_params[3]
        for P_im in projected_arr_cam:
            plane_constraints_pose.append((n_i_C, d_i_C, P_im))
        
        # Append current pose constraints to global lists
        self.collected_edge_constraints.extend(edge_constraints_pose)
        self.collected_plane_constraints.extend(plane_constraints_pose)
        
        # Collect rotation data for global optimization.
        # For edge vectors: use the fitted LiDAR edge directions (transformed_edge_dirs) and corresponding camera directions.
        for i in range(len(transformed_edge_dirs)):
            self.all_lidar_edge_dirs.append(transformed_edge_dirs[i].reshape(1, 3))
            self.all_camera_edge_dirs.append(img_edge_directions[i].reshape(1, 3))
        # For plane normal: use the transformed camera plane normal (transformed_plane_eq) and image plane normal.
        self.all_lidar_plane_normals.append(transformed_plane_eq.reshape(1, 3))
        self.all_camera_plane_normals.append(img_plane_params[:3].reshape(1, 3))
        self.frame_count+=1
def main(args=None):
    rclpy.init(args=args)
    node = Calib_lidar2camera()
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

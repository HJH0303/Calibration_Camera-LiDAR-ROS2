#!/usr/bin/env python3
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node

import struct
import numpy as np
import open3d as o3d
import cv2

from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from collections import defaultdict

###############################################################################
# 1) POINT CLOUD UTILS
###############################################################################

def read_points(pc_msg: PointCloud2):
    """
    Reads (x, y, z, ring) from a typical Velodyne-like PointCloud2.
    Modify offsets if your sensor's actual layout differs.
    """
    x_offset = 0
    y_offset = 4
    z_offset = 8
    ring_offset = 16  # <H> after x,y,z

    step = pc_msg.point_step
    data = pc_msg.data
    total_bytes = len(data)
    num_points = total_bytes // step

    points = []
    rings = []
    for i in range(num_points):
        base = i * step
        if base + ring_offset + 2 > total_bytes:
            continue
        try:
            x = struct.unpack_from('<f', data, base + x_offset)[0]
            y = struct.unpack_from('<f', data, base + y_offset)[0]
            z = struct.unpack_from('<f', data, base + z_offset)[0]
            ring = struct.unpack_from('<H', data, base + ring_offset)[0]

            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                # Example region filter
                if -1.2 < y < 1.2 and x > 0.8:
                    points.append([x, y, z])
                    rings.append(ring)
        except:
            continue

    return np.array(points, dtype=np.float32), rings


def plane_segmentation_and_projection(points_arr, rings_arr,
                                      distance_threshold=0.05,
                                      ransac_n=3,
                                      num_iterations=1000):
    """
    1) Segments a plane from the given Nx3 points using RANSAC.
    2) Returns:
       - plane_model (a,b,c,d),
       - indices of inliers,
       - projected Nx3 array of inlier points,
       - the rings of inliers (in the same order).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_arr)

    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    if len(inliers) == 0:
        return None, None, None, None

    a, b, c, d = plane_model
    plane_normal = np.array([a, b, c], dtype=float)
    normed_plane_normal = plane_normal / np.linalg.norm(plane_normal)

    inlier_points = points_arr[inliers]
    inlier_rings = [rings_arr[i] for i in inliers]

    # Project inlier points onto the plane
    projected = []
    for pt in inlier_points:
        dist = np.dot(pt, plane_normal) + d
        proj = pt - dist * normed_plane_normal
        projected.append(proj)
    projected_arr = np.array(projected)

    return plane_model, inliers, projected_arr, inlier_rings


def fit_line_svd(points_3d):
    """
    Fits a line (centroid + direction) using SVD on Nx3 data.
    Returns (centroid, direction).
    """
    if len(points_3d) == 0:
        return None, None
    arr = np.array(points_3d)
    centroid = np.mean(arr, axis=0)
    _, _, vh = np.linalg.svd(arr - centroid)
    direction = vh[0]
    direction /= np.linalg.norm(direction)  # ensure normalized
    return centroid, direction


###############################################################################
# 2) OPEN3D INTERACTION: PICKING POINT GROUPS & VISUALIZATION
###############################################################################

def pick_points_open3d(pcd, window_name="Select Points"):
    """
    SHIFT+Left Click in the VisualizerWithEditing to pick multiple points.
    Press 'Q' to confirm. Returns the indices in the order they were selected.
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()

def create_cylinder_line(p1, p2, radius=0.01, color=[1, 0.5, 0]):
    """
    Creates an open3d cylinder from p1 to p2 for visualizing a line in 3D.
    """
    vec = p2 - p1
    length = np.linalg.norm(vec)
    mesh_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    mesh_cyl.compute_vertex_normals()

    z_axis = np.array([0, 0, 1], dtype=float)
    if length < 1e-8:
        rot = np.eye(3)
    else:
        v_norm = vec / length
        axis = np.cross(z_axis, v_norm)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-12:
            dot_val = np.dot(z_axis, v_norm)
            if dot_val < 0:
                rot = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
            else:
                rot = np.eye(3)
        else:
            axis /= axis_len
            angle = np.arccos(np.dot(z_axis, v_norm))
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    mesh_cyl.rotate(rot, center=np.zeros(3))
    midpoint = 0.5 * (p1 + p2)
    mesh_cyl.translate(midpoint)
    mesh_cyl.paint_uniform_color(color)
    return mesh_cyl

def create_sphere(center, radius=0.05, color=[0.8, 0.2, 0.8]):
    """
    Create a small sphere at 'center' for corner visualization.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere


###############################################################################
# 3) IMAGE INTERACTION: PICKING 4 CORNERS WITH ZOOM/PAN
###############################################################################

class ImageCornerPicker:
    """
    Allows user to pick exactly 4 corners in an OpenCV window with zoom/pan.
    Press 'q' when done, 'r' to reset, arrow keys to pan, z/Z to zoom in/out, ESC to cancel.
    """
    def __init__(self, title="Pick 4 Corners with Zoom"):
        self.title = title
        self.points = []
        self.max_points = 4
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.original = None

    def display_to_world(self, dx, dy):
        wx = dx / self.scale + self.offset_x
        wy = dy / self.scale + self.offset_y
        return wx, wy

    def world_to_display(self, wx, wy):
        dx = (wx - self.offset_x) * self.scale
        dy = (wy - self.offset_y) * self.scale
        return dx, dy

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.max_points:
                wx, wy = self.display_to_world(x, y)
                self.points.append((wx, wy))
                print(f"Clicked corner {len(self.points)}: ({wx:.2f}, {wy:.2f})")
            else:
                print("Already have 4 corners. Press 'r' to reset if needed.")

    def draw_current_display(self):
        h, w = self.original.shape[:2]

        # Crop region
        crop_w = int(640 / self.scale)
        crop_h = int(480 / self.scale)

        sx = int(self.offset_x)
        sy = int(self.offset_y)
        sx = max(0, min(w - 1, sx))
        sy = max(0, min(h - 1, sy))

        ex = sx + crop_w
        ey = sy + crop_h
        ex = max(0, min(w, ex))
        ey = max(0, min(h, ey))

        cropped = self.original[sy:ey, sx:ex]
        disp_w = int(cropped.shape[1] * self.scale)
        disp_h = int(cropped.shape[0] * self.scale)
        disp_img = cv2.resize(cropped, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # draw points and lines
        for i, (ox, oy) in enumerate(self.points):
            dx, dy = self.world_to_display(ox, oy)
            dx, dy = int(dx), int(dy)
            if 0 <= dx < disp_w and 0 <= dy < disp_h:
                cv2.circle(disp_img, (dx, dy), 5, (0, 255, 255), -1)
                cv2.putText(disp_img, str(i + 1), (dx + 5, dy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(self.points) == self.max_points:
            for i in range(self.max_points):
                pt1 = self.points[i]
                pt2 = self.points[(i + 1) % self.max_points]
                dx1, dy1 = self.world_to_display(pt1[0], pt1[1])
                dx2, dy2 = self.world_to_display(pt2[0], pt2[1])
                cv2.line(disp_img, (int(dx1), int(dy1)), (int(dx2), int(dy2)), (0, 0, 255), 2)

        return disp_img

    def pick_corners(self, frame):
        self.original = frame.copy()
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.mouse_callback)

        while True:
            display = self.draw_current_display()
            cv2.imshow(self.title, display)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                # confirm
                break
            elif key == ord('r'):
                self.points = []
            elif key == ord('z'):
                self.scale *= 1.2
            elif key == ord('x'):
                self.scale /= 1.2
                if self.scale < 0.1:
                    self.scale = 0.1
            elif key == 81:  # left arrow
                self.offset_x -= 20 / self.scale
            elif key == 82:  # up arrow
                self.offset_y -= 20 / self.scale
            elif key == 83:  # right arrow
                self.offset_x += 20 / self.scale
            elif key == 84:  # down arrow
                self.offset_y += 20 / self.scale
            elif key == 27:  # ESC
                print("User canceled (ESC).")
                self.points = []
                break

        cv2.destroyWindow(self.title)
        if len(self.points) < self.max_points:
            return []
        return self.points


###############################################################################
# 4) LINE INTERSECTION UTILS
###############################################################################

def intersect_2d(lineA, lineB):
    cA, dA = lineA
    cB, dB = lineB
    denom = dA[0] * dB[1] - dA[1] * dB[0]
    if abs(denom) < 1e-12:
        return None
    diff = cB - cA
    cross_diff_db = diff[0] * dB[1] - diff[1] * dB[0]
    t = cross_diff_db / denom
    return cA + t * dA

def make_plane_basis(normal):
    n = normal / np.linalg.norm(normal)
    trial = np.array([0, 0, 1], dtype=float)
    if abs(np.dot(n, trial)) > 0.9:
        trial = np.array([0, 1, 0], dtype=float)
    u = np.cross(n, trial)
    u /= np.linalg.norm(u)
    v = np.cross(u, n)
    v /= np.linalg.norm(v)
    return n, u, v

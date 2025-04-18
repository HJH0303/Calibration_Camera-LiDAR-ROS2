
import cv2
import numpy as np

def generate_checkerboard_plane(board_size, square_size):
    """
    Generate 2D world coordinates (on z=0 plane) for checkerboard corners.
    (Coordinates are arranged in column-major order.)
    
    Parameters:
        board_size (tuple): (rows, cols) internal corners, e.g., (8, 6)
        square_size (float): Physical size of each square (in meters)
        
    Returns:
        np.ndarray: Array of shape (N, 2) with checkerboard corner coordinates.
    """
    rows, cols = board_size
    xs = np.linspace(0, (cols - 1) * square_size, cols)
    ys = np.linspace(0, (rows - 1) * square_size, rows)
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv.ravel(), yv.ravel()], axis=-1)
    world_reshaped = points.reshape((rows, cols, 2))
    world_col_major = np.transpose(world_reshaped, (1, 0, 2)).reshape(-1, 2)
    return world_col_major
def get_corners(world_points, padding, board_size):
    """
    Generate world coordinates for the 4 outer corners using padding.
    
    Parameters:
        world_points (np.ndarray): Full set of world points (internal corners) in column-major order,
                                   shape (N,2). For an 8x6 checkerboard internal corners, N = 8*6 = 48.
        padding (array-like): [pad_left, pad_top, pad_right, pad_bottom] (same unit as world_points).
        board_size (tuple): (rows, cols) of internal corners, e.g., (8,6).
        
    Returns:
        np.ndarray: Array of shape (4,2) with outer corner coordinates in order:
                    [top-left, top-right, bottom-right, bottom-left].
    """
    rows, cols = board_size
    # In column-major order, the points are ordered by columns:
    # Index 0: column 0, row 0 --> top-left.
    # Index rows-1: column 0, row (rows-1) --> bottom-left.
    # Index: (cols-1)*rows: last column, row 0 --> top-right.
    # Index: rows*cols - 1: last element --> bottom-right.
    top_left = world_points[0, :].copy()
    bottom_left = world_points[rows - 1, :].copy()
    top_right = world_points[(cols - 1) * rows, :].copy()
    bottom_right = world_points[rows * cols - 1, :].copy()
    
    # Apply padding adjustments:
    # For top-left: shift left by pad_left and up by pad_top.
    top_left[0] -= padding[0]
    top_left[1] -= padding[1]
    
    # For top-right: shift right by pad_right and up by pad_top.
    top_right[0] += padding[2]
    top_right[1] -= padding[1]
    
    # For bottom-left: shift left by pad_left and down by pad_bottom.
    bottom_left[0] -= padding[0]
    bottom_left[1] += padding[3]
    
    # For bottom-right: shift right by pad_right and down by pad_bottom.
    bottom_right[0] += padding[2]
    bottom_right[1] += padding[3]
    
    corners = np.vstack([top_left, top_right, bottom_right, bottom_left])
    return corners

def compute_homography(world_points, image_points):
    """
    Compute the homography matrix mapping world points (z=0) to image points.
    
    Parameters:
        world_points (np.ndarray): Array of shape (N, 2) with world coordinates.
        image_points (np.ndarray): Array of shape (N, 2) with image coordinates.
    
    Returns:
        np.ndarray: 3x3 homography matrix.
    """
    if world_points.shape[0] < 4 or image_points.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")
    H, status = cv2.findHomography(world_points, image_points, 0)
    return H

def visualize_homography(image, world_points, image_points, H, wait_ms=1):
    """
    Visualize the homography mapping by projecting world points onto the image.
    
    Parameters:
        image (np.ndarray): Original image.
        world_points (np.ndarray): (N, 2) world coordinates.
        image_points (np.ndarray): (N, 2) detected image points.
        H (np.ndarray): 3x3 homography matrix.
        wait_ms (int): cv2.waitKey delay.
    """
    num_points = world_points.shape[0]
    world_points_h = np.hstack([world_points, np.ones((num_points, 1))])
    projected_h = (H @ world_points_h.T).T
    projected_points = projected_h[:, :2] / projected_h[:, 2:3]
    
    vis_img = image.copy()
    for pt in image_points:
        cv2.circle(vis_img, tuple(np.int32(pt)), 5, (0, 0, 255), -1)
    for pt in projected_points:
        cv2.circle(vis_img, tuple(np.int32(pt)), 7, (0, 255, 0), -1)
    
    cv2.imshow("Homography Verification", vis_img)
    cv2.waitKey(wait_ms)

def compute_camera_extrinsics_from_homography(H, K):
    """
    Compute camera extrinsics (rotation matrix R and translation vector t)
    from homography H and camera intrinsic matrix K.
    
    Returns:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3-element translation vector.
    """
    H_norm = np.linalg.inv(K) @ H
    lambda_val = 1.0 / np.linalg.norm(H_norm[:, 0])
    r1 = lambda_val * H_norm[:, 0]
    r2 = lambda_val * H_norm[:, 1]
    r3 = np.cross(r1, r2)
    t = lambda_val * H_norm[:, 2]
    R = np.column_stack((r1, r2, r3))
    return R, t

def transform_world_to_camera(world_points, R, t):
    """
    Transform 2D world points (on z=0) to 3D camera coordinates.
    
    Returns:
        np.ndarray: Array of shape (N, 3) with 3D points.
    """
    N = world_points.shape[0]
    X_world = np.hstack([world_points, np.zeros((N, 1))])
    X_cam = (R @ X_world.T) + t.reshape(3, 1)
    return X_cam.T

def project_points(points3d, K,distortion_coef):
    """
    Project 3D points (in camera coordinates) onto the image plane.
    
    Returns:
        np.ndarray: Array of shape (N, 2) with projected 2D points.
    """
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    image_pts, _ = cv2.projectPoints(points3d, rvec, tvec, K, distortion_coef)
    return image_pts.reshape(-1, 2)
def arrange_image_corners_3d(image_points, projected_points_3d, points3d, board_size):
    """
    Rearranges the 3D points corresponding to the outer checkerboard corners
    using the detected image points (assumed row-major order for an 8x6 board)
    and the projected 2D points.
    
    For an 8x6 board (48 points), the outer corner indices (row-major) are:
      - top-left: index 0
      - top-right: index (cols - 1) = 5
      - bottom-left: index (rows - 1)*cols = 42
      - bottom-right: index (rows*cols - 1) = 47
      
    Then, using these detected outer points, the function selects from the provided
    3D points (points3d) the ones whose projection is closest to the corresponding
    detected 2D outer points.
    
    Parameters:
        image_points (np.ndarray): Array of shape (48, 2) with detected image corner positions.
        projected_points_3d (np.ndarray): Array of shape (4, 2) with the projection of 3D points.
        points3d (np.ndarray): Array of shape (4, 3) with 3D points in the camera coordinate system.
        board_size (tuple): (rows, cols), e.g., (8, 6).
        
    Returns:
        np.ndarray: Array of shape (4, 3) with rearranged 3D corner points in order:
                    [top-left, top-right, bottom-right, bottom-left].
                    If projected_points_3d does not have 4 points, returns an empty array.
    """
    rows, cols = board_size
    # Detected outer corners from image_points (row-major order):
    top_left_det = image_points[0, :]
    top_right_det = image_points[cols - 1, :]
    bottom_left_det = image_points[(rows - 1) * cols, :]
    bottom_right_det = image_points[rows * cols - 1, :]
    
    detected_outer = np.vstack([top_left_det, top_right_det, bottom_right_det, bottom_left_det])
    
    # Optional: check and swap if ordering seems reversed (e.g., if top_left is to the right of top_right)
    if detected_outer[0, 0] > detected_outer[1, 0]:
        detected_outer[[0, 2]] = detected_outer[[2, 0]]
        detected_outer[[1, 3]] = detected_outer[[3, 1]]
    
    # Ensure that projected_points_3d has exactly 4 points.
    if projected_points_3d.size != 0 and projected_points_3d.shape[0] == 4:
        rearranged_indices = []
        for i in range(4):
            # Compute Euclidean distance between detected outer corner and each projected point.
            distances = np.linalg.norm(projected_points_3d - detected_outer[i], axis=1)
            idx = np.argmin(distances)
            rearranged_indices.append(idx)
        rearranged = np.vstack([points3d[idx, :] for idx in rearranged_indices])
    else:
        rearranged = np.array([])
    
    return rearranged


def extract_corners(image_points, world_points, board_size, padding, camera_intrinsic, image_idx,cv_image,distortion_coef):
    """
    Extracts and rearranges 3D checkerboard corner points in the camera coordinate system.
    
    Parameters:
        image_points (np.ndarray): Detected 2D image corner points (N,2).
        world_points (np.ndarray): World coordinates from generate_checkerboard_plane.
        board_size (tuple): (rows, cols) internal corners.
        padding (array-like): [pad_left, pad_top, pad_right, pad_bottom].
        square_size (float): Physical size of each square.
        camera_intrinsic (np.ndarray): 3x3 intrinsic matrix.
        image_idx (bool): Validity flag.
        
    Returns:
        tuple: (rearranged_3d, image_idx)
            rearranged_3d: (4,3) array of arranged 3D points.
            image_idx: Updated validity flag.
    """
    H = compute_homography(world_points, image_points)
    R, t = compute_camera_extrinsics_from_homography(H, camera_intrinsic)
    visualize_homography(cv_image.copy(), world_points, image_points, H)
    # Compute padded dimensions and get outer world corners.

    a = transform_world_to_camera(world_points,R,t)
    padded_dims = np.full((4,), 0) + np.array(padding)
    corner_world_pts = get_corners(world_points, padded_dims, board_size)
    # Add z=0 for 3D world coordinates.
    corner_world_pts_3d = np.hstack([corner_world_pts, np.zeros((4, 1))])

    # Transform world corners to camera coordinate system.
    transformed_corners = (R @ corner_world_pts_3d.T + t.reshape(3, 1)).T
    tmp = transformed_corners 
    # Project the 3D points onto the image.
    projected_pts_3d = project_points(tmp, camera_intrinsic,distortion_coef)
    
    arranged = arrange_image_corners_3d(image_points, projected_pts_3d, tmp, board_size)
    if arranged.size == 0:
        image_idx = False
        return np.array([]), image_idx
    else:
        return arranged, image_idx
    


def compute_plane_and_edge_parameters(corners_3d):
    """
    Given 4 corner points (in a 4x3 array) representing the checkerboard outer corners in 3D
    (in the camera coordinate system), compute:
      - edge_directions: a (4,3) array of unit vectors representing the direction of each edge.
      - plane_params: a (4,) array where the first 3 elements are the plane normal (from the first two edges)
                      and the 4th element is d in the plane equation: ax + by + cz + d = 0.
    
    Parameters:
        corners_3d (np.ndarray): Array of shape (4, 3) with the 3D corner points in order:
                                 [top-left, top-right, bottom-right, bottom-left].
    
    Returns:
        edge_directions (np.ndarray): Array of shape (4, 3) with unit direction vectors for each edge.
        plane_params (np.ndarray): Array of shape (4,) where plane_params[:3] is the normal vector and
                                   plane_params[3] is d in the plane equation.
    """
    # Compute direction vectors for each edge.
    # Edge 1: from corner 1 (top-left) to corner 2 (top-right)
    # Edge 2: from corner 2 (top-right) to corner 3 (bottom-right)
    # Edge 3: from corner 3 (bottom-right) to corner 4 (bottom-left)
    # Edge 4: from corner 4 (bottom-left) to corner 1 (top-left)
    edge_directions = np.zeros((4, 3))
    # Ensure that the difference is not zero to avoid division by zero.
    def unit_vector(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v
    
    edge_directions[0, :] = unit_vector(corners_3d[1, :] - corners_3d[0, :])
    edge_directions[1, :] = unit_vector(corners_3d[2, :] - corners_3d[1, :])
    edge_directions[2, :] = unit_vector(corners_3d[3, :] - corners_3d[2, :])
    edge_directions[3, :] = unit_vector(corners_3d[0, :] - corners_3d[3, :])
    
    # Compute plane normal vector from the first two edge directions.
    # MATLAB: normal = cross(direction(2, :), direction(1, :));
    normal = np.cross(edge_directions[1, :], edge_directions[0, :])
    normal = unit_vector(normal)
    
    # Compute d in the plane equation: a*x + b*y + c*z + d = 0,
    # using the first corner (top-left) of the polygon.
    d = -np.dot(corners_3d[0, :], normal)
    
    plane_params = np.hstack([normal, d])
    
    return edge_directions, plane_params
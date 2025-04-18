import numpy as np
from scipy.optimize import least_squares

def get_lidar_to_camera_rotation():
    rotation_matrix = np.array([
        [0, 0, 1],    # x in LiDAR -> z in Camera
        [-1, 0, 0],   # y in LiDAR -> -x in Camera
        [0, -1, 0]    # z in LiDAR -> -y in Camera
    ])
    return rotation_matrix

def transform_points(points, rotation_matrix):
    return points.dot(rotation_matrix)

def transform_edge_boundary_points(edge_boundary_points, rotation_matrix):
    transformed = {}
    for key, pts in edge_boundary_points.items():
        transformed[key] = transform_points(pts, rotation_matrix)
    return transformed

def transform_projected_arr(projected_arr, rotation_matrix):
    return transform_points(projected_arr, rotation_matrix)

def estimate_rotation_from_edges_and_normal(vectors_lidar, vectors_camera):
    H = np.zeros((3, 3))
    for i in range(vectors_lidar.shape[0]):
        H += np.outer(vectors_camera[i], vectors_lidar[i])
    U, S, Vt = np.linalg.svd(H)
    R_tmp = U @ Vt
    D = np.diag([1, 1, np.sign(np.linalg.det(R_tmp))])
    R_opt = U @ D @ Vt
    return R_opt

def solve_translation(R_L_C, edge_constraints, plane_constraints):
    edge_rows = []
    edge_rhs = []
    for (Q_ijk_L, P_ij_C, d_ij_C) in edge_constraints:
        proj_matrix = np.eye(3) - np.outer(d_ij_C, d_ij_C)
        A_edge = proj_matrix
        b_edge = proj_matrix @ (P_ij_C - R_L_C @ Q_ijk_L)
        edge_rows.append(A_edge)
        edge_rhs.append(b_edge)
    if len(edge_rows) > 0:
        A_edge_all = np.vstack(edge_rows)
        b_edge_all = np.hstack(edge_rhs)
    else:
        A_edge_all = np.empty((0, 3))
        b_edge_all = np.empty((0,))
    plane_rows = []
    plane_rhs = []
    for (n_i_C, d_i_C, P_im_L) in plane_constraints:
        A_plane = n_i_C.reshape(1, 3)
        b_plane = - (d_i_C + np.dot(n_i_C, R_L_C @ P_im_L))
        plane_rows.append(A_plane)
        plane_rhs.append(b_plane)
    if len(plane_rows) > 0:
        A_plane_all = np.vstack(plane_rows)
        b_plane_all = np.array(plane_rhs)
    else:
        A_plane_all = np.empty((0, 3))
        b_plane_all = np.empty((0,))
    A = np.vstack([A_edge_all, A_plane_all])
    b = np.hstack([b_edge_all, b_plane_all])
    t_est, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    return t_est

def transform_lidar_to_camera(corners, edge_dirs, plane_eq):
    # Rotation matrix to convert LiDAR frame to Camera frame
    rotation_matrix = np.array([
        [0, 0, 1],    # x in LiDAR -> z in Camera
        [-1, 0, 0],   # y in LiDAR -> -x in Camera
        [0, -1, 0]    # z in LiDAR -> -y in Camera
    ])

    # Transform corners, edge directions, and normal vector using the rotation matrix
    transformed_corners = corners.dot(rotation_matrix)
    transformed_edge_dirs = edge_dirs.dot(rotation_matrix)
    transformed_plane_eq = plane_eq[:3].dot(rotation_matrix)

    return transformed_corners, transformed_edge_dirs, transformed_plane_eq

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    Assuming ZYX order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[ cz, -sz,  0 ],
                   [ sz,  cz,  0 ],
                   [  0,   0,  1 ]])
    Ry = np.array([[  cy,  0, sy ],
                   [   0,  1,  0 ],
                   [ -sy,  0, cy ]])
    Rx = np.array([[ 1,  0,   0 ],
                   [ 0, cx, -sx ],
                   [ 0, sx,  cx ]])
    R = Rz @ Ry @ Rx
    return R

def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix R (3x3) to Euler angles (roll, pitch, yaw) in ZYX order.
    This implementation follows a standard convention.
    """
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return roll, pitch, yaw

def residual_func(params, data):
    """
    params = [roll, pitch, yaw, tx, ty, tz]
    data = { 'plane_constraints': list of (n_i_C, d_i_C, P_im_L),
             'edge_constraints' : list of (Q_ijk_L, P_ij_C, d_ij_C) }
    Returns the concatenated 1-D residual vector.
    """
    roll, pitch, yaw, tx, ty, tz = params
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    t = np.array([tx, ty, tz])
    residuals = []
    
    # Plane constraints: n_i_C^T (R * P_im_L + t) + d_i_C = 0
    for (n_i_C, d_i_C, P_im_L) in data['plane_constraints']:
        # Ensure P_im_L and n_i_C are 1D arrays of length 3.
        P_im_L = np.atleast_1d(P_im_L).flatten()
        n_i_C = np.atleast_1d(n_i_C).flatten()
        if P_im_L.size != 3 or n_i_C.size != 3:
            raise ValueError("Plane constraint vectors must be of size 3. Got P_im_L: {}, n_i_C: {}"
                             .format(P_im_L.shape, n_i_C.shape))
        transformed = R @ P_im_L + t
        res_plane = np.dot(n_i_C, transformed) + d_i_C
        residuals.append(res_plane)
    
    # Edge constraints: (I - d_ij_C d_ij_C^T)(R*Q_ijk_L + t - P_ij_C) = 0
    for (Q_ijk_L, P_ij_C, d_ij_C) in data['edge_constraints']:
        Q_ijk_L = np.atleast_1d(Q_ijk_L).flatten()
        P_ij_C  = np.atleast_1d(P_ij_C).flatten()
        d_ij_C  = np.atleast_1d(d_ij_C).flatten()
        if Q_ijk_L.size != 3 or P_ij_C.size != 3 or d_ij_C.size != 3:
            raise ValueError("Edge constraint vectors must be of size 3. Got: Q_ijk_L {}, P_ij_C {}, d_ij_C {}"
                             .format(Q_ijk_L.shape, P_ij_C.shape, d_ij_C.shape))
        diff = (R @ Q_ijk_L + t) - P_ij_C
        proj_matrix = np.eye(3) - np.outer(d_ij_C, d_ij_C)
        edge_res = proj_matrix @ diff
        residuals.extend(edge_res)  # 3 residuals per constraint
    
    return np.array(residuals)

def run_lm_optimization(plane_constraints, edge_constraints, R_init, t_init):
    """
    Joint optimization using LM.
    plane_constraints: list of (n_i_C, d_i_C, P_im_L)
    edge_constraints : list of (Q_ijk_L, P_ij_C, d_ij_C)
    R_init: initial rotation matrix (3x3) from previous estimation.
    t_init: initial translation vector (3,) from previous estimation.
    
    Returns:
      R_opt: optimized rotation matrix (3x3)
      t_opt: optimized translation vector (3,)
      result: full optimization result.
    """
    roll0, pitch0, yaw0 = rotation_matrix_to_euler_angles(R_init)
    init_guess = [roll0, pitch0, yaw0, t_init[0], t_init[1], t_init[2]]
    
    data_dict = {'plane_constraints': plane_constraints,
                 'edge_constraints': edge_constraints}
    
    result = least_squares(fun=residual_func, x0=init_guess, args=(data_dict,), method='lm')
    
    opt_params = result.x
    roll, pitch, yaw, tx, ty, tz = opt_params
    R_opt = euler_to_rotation_matrix(roll, pitch, yaw)
    t_opt = np.array([tx, ty, tz])
    return R_opt, t_opt, result
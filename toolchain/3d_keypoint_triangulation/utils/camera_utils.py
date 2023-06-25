import numpy as np


def rotation_mat2quat(rot_mat):
    '''Transform the rotation matrix to quaternion(w, x, y, z).'''
    w = np.math.sqrt(np.trace(rot_mat) + 1) / 2
    x = (rot_mat[2, 1] - rot_mat[1, 2]) / (4 * w)
    y = (rot_mat[0, 2] - rot_mat[2, 0]) / (4 * w)
    z = (rot_mat[1, 0] - rot_mat[0, 1]) / (4 * w)

    return w, x, y, z


def load_camera_param_pkg(prms_path):
    # load camera parameters
    prj_mats = []
    extrinsics = []
    intrinsics = []
    cam_idx_conv = [0, 1, 3, 4]

    cam_param_pkg = np.load(prms_path, allow_pickle=True)['cam_param_pkg']
    for i in range(4):
        if i == 2:
            extrinsic = np.eye(4, dtype=float)
            extrinsics.append(extrinsic)
            intrinsic = cam_param_pkg[i]['cam2_3']['main_cam_param']
            intrinsics.append(intrinsic)
            prj_mat = np.matmul(np.hstack((intrinsic, np.array([[0], [0], [1]]))), extrinsic)
            prj_mats.append(prj_mat)
        extrinsic = np.vstack((np.hstack((cam_param_pkg[i][f'cam2_{cam_idx_conv[i]}']['vice_cam_rot'],
                                          cam_param_pkg[i][f'cam2_{cam_idx_conv[i]}']['vice_cam_trans'])),
                               np.array([[0, 0, 0, 1]])))
        extrinsics.append(extrinsic)
        intrinsic = cam_param_pkg[i][f'cam2_{cam_idx_conv[i]}']['vice_cam_param']
        intrinsics.append(intrinsic)
        prj_mat = np.matmul(np.hstack((intrinsic, np.array([[0], [0], [1]]))), extrinsic)
        prj_mats.append(prj_mat)
    prj_mats = np.asarray(prj_mats)
    extrinsics = np.asarray(extrinsics)
    intrinsics = np.asarray(intrinsics)

    return prj_mats, extrinsics, intrinsics

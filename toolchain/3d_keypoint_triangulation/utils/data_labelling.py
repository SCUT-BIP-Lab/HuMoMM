import numpy as np
import cv2
import os

from utils.camera_utils import rotation_mat2quat


def cam_reproject(input_3d, rp_cam_idx, cam_param_pkg, w_max=640, h_max=480):
    '''
    Reproject the 3d joint to the selected camera.

    :param input_3d: 3d joint coordinates.
    :param rp_cam_idx: index of the camera which need to be reprojected.
    :param cam_param_pkg: cam_param_pkg.npz['cam_param_pkg']
    :param w_max: width of image, default 640.
    :param h_max: height of image, default 480.
    :return: 2d joint of the selected camera.
    '''

    cam_idx_conv = [0, 1, None, 2, 3]
    X0Y0Z0 = np.hstack((input_3d, [[1]])).T

    if rp_cam_idx != 2:
        cam1_intri_param = cam_param_pkg[cam_idx_conv[rp_cam_idx]][f'cam2_{rp_cam_idx}']['vice_cam_param']
        cam1_rot_mat = cam_param_pkg[cam_idx_conv[rp_cam_idx]][f'cam2_{rp_cam_idx}']['vice_cam_rot']
        cam1_trans_mat = cam_param_pkg[cam_idx_conv[rp_cam_idx]][f'cam2_{rp_cam_idx}']['vice_cam_trans']

        cam1_proj = np.matmul(np.hstack((cam1_intri_param, np.array([[0], [0], [1]]))),
                              np.vstack((np.hstack((cam1_rot_mat, cam1_trans_mat)), np.array([0, 0, 0, 1]))))
        X1Y1Z1 = np.matmul(cam1_proj, X0Y0Z0)
        reproject_2d = X1Y1Z1 / X1Y1Z1[2, 0]
    else:
        X0Y0Z0 = X0Y0Z0[:3]

        cam0_intri_param = cam_param_pkg[cam_idx_conv[3]]['cam2_3']['main_cam_param']
        reproject_2d = np.matmul(cam0_intri_param, X0Y0Z0) / X0Y0Z0[2, 0]

    reproject_2d = reproject_2d.T[:, :2]

    # restrict the reprojected point in the image.
    reproject_2d[0, 0] = min(w_max-1, max(0, reproject_2d[0, 0]))
    reproject_2d[0, 1] = min(h_max - 1, max(0, reproject_2d[0, 1]))

    return reproject_2d


def joint_3d_trans_cam(input_3d, cam_view_idx, cam_param_pkg):
    '''
    Convert 3d joint from main camera NO.2 to other camera.

    :param input_3d: 3d joint under main camera NO.2.
    :param cam_view_idx: index of the camera which you need.
    :param cam_param_pkg: cam_param_pkg.npz['cam_param_pkg']
    :return: 3d joint of the expected camera
    '''

    if cam_view_idx == 2:
        return input_3d

    cam_idx_conv = [0, 1, None, 2, 3]

    cam1_rot_mat = cam_param_pkg[cam_idx_conv[cam_view_idx]][f'cam2_{cam_view_idx}']['vice_cam_rot']
    cam1_trans_mat = cam_param_pkg[cam_idx_conv[cam_view_idx]][f'cam2_{cam_view_idx}']['vice_cam_trans']

    X0Y0Z0 = np.hstack((input_3d, [[1]])).T
    trans_3d_joint = np.matmul(np.hstack((cam1_rot_mat, cam1_trans_mat)), X0Y0Z0)

    return trans_3d_joint.T


def solve_3d(uv_1, uv_2, prj_mat_1, prj_mat_2):
    '''
    Calculate 3D world coordinate using 2D pixel coordinates from two views.

    :param uv_1: array of (N, 2), 2D coordinates of one view.
    :param uv_2: array of (N, 2), corresponding 2D coordinates of another view.
    :param prj_mat_1: array of (3, 4), the projective matrix(intrinsic multiply extrinsic) of one view.
    :param prj_mat_2: array of (3, 4), the projective matrix of another view.
    :return: array of (N, 3), 3D world coordinate of the points.
    '''

    A = np.array([
        [uv_1[:, 0] * prj_mat_1[2, 0] - prj_mat_1[0, 0],
         uv_1[:, 0] * prj_mat_1[2, 1] - prj_mat_1[0, 1],
         uv_1[:, 0] * prj_mat_1[2, 2] - prj_mat_1[0, 2]],
        [uv_1[:, 1] * prj_mat_1[2, 0] - prj_mat_1[1, 0],
         uv_1[:, 1] * prj_mat_1[2, 1] - prj_mat_1[1, 1],
         uv_1[:, 1] * prj_mat_1[2, 2] - prj_mat_1[1, 2]],
        [uv_2[:, 0] * prj_mat_2[2, 0] - prj_mat_2[0, 0],
         uv_2[:, 0] * prj_mat_2[2, 1] - prj_mat_2[0, 1],
         uv_2[:, 0] * prj_mat_2[2, 2] - prj_mat_2[0, 2]],
        [uv_2[:, 1] * prj_mat_2[2, 0] - prj_mat_2[1, 0],
         uv_2[:, 1] * prj_mat_2[2, 1] - prj_mat_2[1, 1],
         uv_2[:, 1] * prj_mat_2[2, 2] - prj_mat_2[1, 2]]
    ]).transpose([2, 0, 1])
    B = np.array([
        [prj_mat_1[0, 3] - uv_1[:, 0] * prj_mat_1[2, 3]],
        [prj_mat_1[1, 3] - uv_1[:, 1] * prj_mat_1[2, 3]],
        [prj_mat_2[0, 3] - uv_2[:, 0] * prj_mat_2[2, 3]],
        [prj_mat_2[1, 3] - uv_2[:, 1] * prj_mat_2[2, 3]]
    ]).transpose([2, 0, 1])

    xyz = np.matmul(np.linalg.pinv(A), B)[:, :, 0]

    return xyz


def get_3d_joints(input_2d_0, input_2d_1, main_cam_idx, vice_cam_idx, cam_param_pkg):
    '''
    Get 3d joints coordinates from 2d input coordinates of two view points and their camera params.

    :param input_2d_0: 2d input of main camera.
    :param input_2d_1: 2d input of vice camera.
    :param main_cam_idx: main camera index.
    :param vice_cam_idx: vice camera index.
    :param cam_param: cam_param_pkg.npz['cam_param_pkg']
    :return: 3d joint coordinates.
    '''

    cam_idx_conv = [0, 1, None, 2, 3]

    if main_cam_idx == 2:
        cam0_intri_param = cam_param_pkg[cam_idx_conv[vice_cam_idx]][f'cam{main_cam_idx}_{vice_cam_idx}']['main_cam_param']
        cam1_intri_param = cam_param_pkg[cam_idx_conv[vice_cam_idx]][f'cam{main_cam_idx}_{vice_cam_idx}']['vice_cam_param']
        cam1_rot_mat = cam_param_pkg[cam_idx_conv[vice_cam_idx]][f'cam{main_cam_idx}_{vice_cam_idx}']['vice_cam_rot']
        cam1_trans_mat = cam_param_pkg[cam_idx_conv[vice_cam_idx]][f'cam{main_cam_idx}_{vice_cam_idx}']['vice_cam_trans']

        eyemat = np.eye(4, dtype=float)
        main_cam_proj = np.matmul(np.hstack((cam0_intri_param, np.array([[0], [0], [1]]))), eyemat)
        vice_cam_proj = np.matmul(np.hstack((cam1_intri_param, np.array([[0], [0], [1]]))),
                                  np.vstack((np.hstack((cam1_rot_mat, cam1_trans_mat)),
                                             np.array([[0, 0, 0, 1]]))))
        joint_idx_3d = solve_3d(input_2d_0,
                                input_2d_1,
                                main_cam_proj,
                                vice_cam_proj)
    else:
        if vice_cam_idx == 2:
            cam0_intri_param = cam_param_pkg[cam_idx_conv[main_cam_idx]][f'cam{vice_cam_idx}_{main_cam_idx}']['main_cam_param']
            cam1_intri_param = cam_param_pkg[cam_idx_conv[main_cam_idx]][f'cam{vice_cam_idx}_{main_cam_idx}']['vice_cam_param']
            cam1_rot_mat = cam_param_pkg[cam_idx_conv[main_cam_idx]][f'cam{vice_cam_idx}_{main_cam_idx}']['vice_cam_rot']
            cam1_trans_mat = cam_param_pkg[cam_idx_conv[main_cam_idx]][f'cam{vice_cam_idx}_{main_cam_idx}']['vice_cam_trans']

            eyemat = np.eye(4, dtype=float)
            main_cam_proj = np.matmul(np.hstack((cam1_intri_param, np.array([[0], [0], [1]]))), eyemat)
            vice_cam_proj = np.matmul(np.hstack((cam0_intri_param, np.array([[0], [0], [1]]))),
                                      np.linalg.inv(
                                          np.vstack((np.hstack((cam1_rot_mat, cam1_trans_mat)),
                                                     np.array([[0, 0, 0, 1]])))))

            joint_idx_3d = solve_3d(input_2d_0,
                                    input_2d_1,
                                    main_cam_proj,
                                    vice_cam_proj)

            joint_idx_3d = np.vstack((joint_idx_3d.T, np.array([[1]])))
            joint_idx_3d = np.matmul(
                np.linalg.inv(np.vstack((np.hstack((cam1_rot_mat, cam1_trans_mat)), np.array([[0, 0, 0, 1]])))),
                joint_idx_3d)
            joint_idx_3d = joint_idx_3d[:3, :].T
        else:
            cam0_intri_param = cam_param_pkg[cam_idx_conv[main_cam_idx]][f'cam2_{main_cam_idx}']['vice_cam_param']
            cam0_rot_mat = cam_param_pkg[cam_idx_conv[main_cam_idx]][f'cam2_{main_cam_idx}']['vice_cam_rot']
            cam0_trans_mat = cam_param_pkg[cam_idx_conv[main_cam_idx]][f'cam2_{main_cam_idx}']['vice_cam_trans']

            cam1_intri_param = cam_param_pkg[cam_idx_conv[vice_cam_idx]][f'cam2_{vice_cam_idx}']['vice_cam_param']
            cam1_rot_mat = cam_param_pkg[cam_idx_conv[vice_cam_idx]][f'cam2_{vice_cam_idx}']['vice_cam_rot']
            cam1_trans_mat = cam_param_pkg[cam_idx_conv[vice_cam_idx]][f'cam2_{vice_cam_idx}']['vice_cam_trans']

            eyemat = np.eye(4, dtype=float)
            main_cam_proj = np.matmul(np.hstack((cam0_intri_param, np.array([[0], [0], [1]]))), eyemat)
            vice_cam_proj = np.matmul(np.hstack((cam1_intri_param, np.array([[0], [0], [1]]))),
                                      np.linalg.inv(
                                          np.vstack((np.hstack((cam0_rot_mat, cam0_trans_mat)),
                                                     np.array([[0, 0, 0, 1]])))))
            vice_cam_proj = np.matmul(vice_cam_proj,
                                      np.vstack((np.hstack((cam1_rot_mat, cam1_trans_mat)), np.array([[0, 0, 0, 1]]))))

            joint_idx_3d = solve_3d(input_2d_0,
                                    input_2d_1,
                                    main_cam_proj,
                                    vice_cam_proj)

            joint_idx_3d = np.vstack((joint_idx_3d.T, np.array([[1]])))
            joint_idx_3d = np.matmul(
                np.linalg.inv(np.vstack((np.hstack((cam0_rot_mat, cam0_trans_mat)), np.array([[0, 0, 0, 1]])))),
                joint_idx_3d)
            joint_idx_3d = joint_idx_3d[:3, :].T

    return joint_idx_3d


def min_rep_err(joints_3d_list):
    return np.mean(joints_3d_list, axis=0)


def write_txt_colmap(keypoints_3d, coordinates_2d, intrinsics, extrinsics, write_txt_path):
    '''
    Write txt files of single frame for Bundle Adjustment calculation with colmap.

    :param keypoints_3d: np.ndarray(n_joint, 3), all initial 3D joints of single frame.
    :param coordinates_2d: np.ndarray(n_view, n_joint, 2), with elements of (u, v), original 2D joints of each view.
    :param intrinsics: np.ndarray(n_view, 3, 3), all original camera intrinsics.
    :param extrinsics: np.ndarray(n_view, 4, 4), all camera extrinsics.
    :param write_txt_path: str, path to dir saving colmap txt files.
    :param undistort: bool, whether the input 2D coordinates are undistorted.
    :return:
    '''

    # write camera.txt
    contents = []
    for i in range(5):
        fx, fy, cx, cy = intrinsics[i][(0, 1, 0, 1), (0, 1, 2, 2)]
        contents.append(f'{i+1} PINHOLE 640 480 {fx} {fy} {cx} {cy}\n')
    with open(os.path.join(write_txt_path, 'cameras.txt'), 'w') as f:
        f.writelines(contents)

    # write images.txt
    contents = []
    for n_view, joints in enumerate(coordinates_2d):
        R = extrinsics[n_view][:3, :3]
        T = extrinsics[n_view][:3, 3]
        w, x, y, z = rotation_mat2quat(R)
        contents.append(f'{n_view+1} {w} {x} {y} {z} {T[0]} {T[1]} {T[2]} {n_view+1}\n')
        points_2d = ''
        for n_joint, joint in enumerate(joints):
            u, v = joint
            if u == -1 or v == -1:
                continue
            points_2d = points_2d + f'{u} {v} {n_joint+1} '
        contents.append(points_2d + '\n')
    with open(os.path.join(write_txt_path, 'images.txt'), 'w') as f:
        f.writelines(contents)

    # write points3D.txt
    contents = []
    for n, joint_3d in enumerate(keypoints_3d):
        contents.append(f'{n+1} {joint_3d[0]} {joint_3d[1]} {joint_3d[2]}\n')
    with open(os.path.join(write_txt_path, 'points3D.txt'), 'w') as f:
        f.writelines(contents)

    return


if __name__ == '__main__':
    import json

    '''
    自定义参数段 =======================================================
    '''
    data_root = '/data/pose_datasets/scut2'
    video_name = 'P008R000A011'
    frame_idx = 30
    src_img0 = cv2.imread(f'{data_root}/RGB_frame/C000{video_name}/C000{video_name}RF{frame_idx:03d}.jpg')
    src_img1 = cv2.imread(f'{data_root}/RGB_frame/C001{video_name}/C001{video_name}RF{frame_idx:03d}.jpg')
    src_img2 = cv2.imread(f'{data_root}/RGB_frame/C002{video_name}/C002{video_name}RF{frame_idx:03d}.jpg')
    src_img3 = cv2.imread(f'{data_root}/RGB_frame/C003{video_name}/C003{video_name}RF{frame_idx:03d}.jpg')
    src_img4 = cv2.imread(f'{data_root}/RGB_frame/C004{video_name}/C004{video_name}RF{frame_idx:03d}.jpg')

    label_root = f'{data_root}/Label_3d'
    with open(f'{label_root}/C000{video_name}/C000{video_name}RF{frame_idx:03d}.json', 'r') as f:
        js_data = json.load(f)
        js_data['shapes'].sort(key=lambda x: int(x['label']))
        input_2d_0 = js_data['shapes']
    with open(f'{label_root}/C001{video_name}/C001{video_name}RF{frame_idx:03d}.json', 'r') as f:
        js_data = json.load(f)
        js_data['shapes'].sort(key=lambda x: int(x['label']))
        input_2d_1 = js_data['shapes']
    with open(f'{label_root}/C002{video_name}/C002{video_name}RF{frame_idx:03d}.json', 'r') as f:
        js_data = json.load(f)
        js_data['shapes'].sort(key=lambda x: int(x['label']))
        input_2d_2 = js_data['shapes']
    with open(f'{label_root}/C003{video_name}/C003{video_name}RF{frame_idx:03d}.json', 'r') as f:
        js_data = json.load(f)
        js_data['shapes'].sort(key=lambda x: int(x['label']))
        input_2d_3 = js_data['shapes']
    with open(f'{label_root}/C004{video_name}/C004{video_name}RF{frame_idx:03d}.json', 'r') as f:
        js_data = json.load(f)
        js_data['shapes'].sort(key=lambda x: int(x['label']))
        input_2d_4 = js_data['shapes']

    cam_param = np.load('./cam_param/cam_param_2/cam_param_pkg.npz', allow_pickle=True)['cam_param_pkg']
    '''
    =======================================================
    '''

    for joint_idx in range(21):
    #     joints_3d = get_3d_joints(np.array(input_2d_0[joint_idx]['points']), np.array(input_2d_1[joint_idx]['points']), 0, 1, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_0[joint_idx]['points']), np.array(input_2d_2[joint_idx]['points']), 0, 2, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_0[joint_idx]['points']), np.array(input_2d_3[joint_idx]['points']), 0, 3, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_0[joint_idx]['points']), np.array(input_2d_4[joint_idx]['points']), 0, 4, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_1[joint_idx]['points']), np.array(input_2d_2[joint_idx]['points']), 1, 2, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_1[joint_idx]['points']), np.array(input_2d_3[joint_idx]['points']), 1, 3, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_1[joint_idx]['points']), np.array(input_2d_4[joint_idx]['points']), 1, 4, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_2[joint_idx]['points']), np.array(input_2d_3[joint_idx]['points']), 2, 3, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_2[joint_idx]['points']), np.array(input_2d_4[joint_idx]['points']), 2, 4, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_3[joint_idx]['points']), np.array(input_2d_4[joint_idx]['points']), 3, 4, cam_param)

    #     joints_3d += get_3d_joints(np.array(input_2d_1[joint_idx]['points']), np.array(input_2d_0[joint_idx]['points']), 1, 0, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_2[joint_idx]['points']), np.array(input_2d_0[joint_idx]['points']), 2, 0, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_3[joint_idx]['points']), np.array(input_2d_0[joint_idx]['points']), 3, 0, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_4[joint_idx]['points']), np.array(input_2d_0[joint_idx]['points']), 4, 0, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_2[joint_idx]['points']), np.array(input_2d_1[joint_idx]['points']), 2, 1, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_3[joint_idx]['points']), np.array(input_2d_1[joint_idx]['points']), 3, 1, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_4[joint_idx]['points']), np.array(input_2d_1[joint_idx]['points']), 4, 1, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_3[joint_idx]['points']), np.array(input_2d_2[joint_idx]['points']), 3, 2, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_4[joint_idx]['points']), np.array(input_2d_2[joint_idx]['points']), 4, 2, cam_param)
    #     joints_3d += get_3d_joints(np.array(input_2d_4[joint_idx]['points']), np.array(input_2d_3[joint_idx]['points']), 4, 3, cam_param)
    #     joints_3d /= 20
    #     print(f'joint:{joint_idx}', joints_3d)

        # 仅用求均值优化三维坐标
        # vice_reproject0 = cam_reproject(joints_3d, 0, cam_param)
        # vice_reproject1 = cam_reproject(joints_3d, 1, cam_param)
        # vice_reproject2 = cam_reproject(joints_3d, 2, cam_param)
        # vice_reproject3 = cam_reproject(joints_3d, 3, cam_param)
        # vice_reproject4 = cam_reproject(joints_3d, 4, cam_param)

        # Label_3d_ba中用光束平差优化有的三维坐标
        vice_reproject0 = cam_reproject(input_2d_2[joint_idx]['points_3d'], 0, cam_param)
        vice_reproject1 = cam_reproject(input_2d_2[joint_idx]['points_3d'], 1, cam_param)
        vice_reproject2 = cam_reproject(input_2d_2[joint_idx]['points_3d'], 2, cam_param)
        vice_reproject3 = cam_reproject(input_2d_2[joint_idx]['points_3d'], 3, cam_param)
        vice_reproject4 = cam_reproject(input_2d_2[joint_idx]['points_3d'], 4, cam_param)

        print('0 view error: ', vice_reproject0 - np.array(input_2d_0[joint_idx]['points']))
        print('1 view error: ', vice_reproject1 - np.array(input_2d_1[joint_idx]['points']))
        print('2 view error: ', vice_reproject2 - np.array(input_2d_2[joint_idx]['points']))
        print('3 view error: ', vice_reproject3 - np.array(input_2d_3[joint_idx]['points']))
        print('4 view error: ', vice_reproject4 - np.array(input_2d_4[joint_idx]['points']))

        cv2.circle(src_img0, (int(input_2d_0[joint_idx]['points'][0][0]), int(input_2d_0[joint_idx]['points'][0][1])), 5, (0, 0, 255), 2)
        cv2.circle(src_img1, (int(input_2d_1[joint_idx]['points'][0][0]), int(input_2d_1[joint_idx]['points'][0][1])), 5, (0, 0, 255), 2)
        cv2.circle(src_img2, (int(input_2d_2[joint_idx]['points'][0][0]), int(input_2d_2[joint_idx]['points'][0][1])), 5, (0, 0, 255), 2)
        cv2.circle(src_img3, (int(input_2d_3[joint_idx]['points'][0][0]), int(input_2d_3[joint_idx]['points'][0][1])), 5, (0, 0, 255), 2)
        cv2.circle(src_img4, (int(input_2d_4[joint_idx]['points'][0][0]), int(input_2d_4[joint_idx]['points'][0][1])), 5, (0, 0, 255), 2)

        cv2.circle(src_img0, (int(vice_reproject0[0, 0]), int(vice_reproject0[0, 1])), 8, (0, 255, 0), 2)
        cv2.circle(src_img1, (int(vice_reproject1[0, 0]), int(vice_reproject1[0, 1])), 8, (0, 255, 0), 2)
        cv2.circle(src_img2, (int(vice_reproject2[0, 0]), int(vice_reproject2[0, 1])), 8, (0, 255, 0), 2)
        cv2.circle(src_img3, (int(vice_reproject3[0, 0]), int(vice_reproject3[0, 1])), 8, (0, 255, 0), 2)
        cv2.circle(src_img4, (int(vice_reproject4[0, 0]), int(vice_reproject4[0, 1])), 8, (0, 255, 0), 2)

    h, w, c = src_img0.shape

    src_img0 = cv2.resize(src_img0, (int(w / 1.5), int(h / 1.5)))
    src_img1 = cv2.resize(src_img1, (int(w / 1.5), int(h / 1.5)))
    src_img2 = cv2.resize(src_img2, (int(w / 1.5), int(h / 1.5)))
    src_img3 = cv2.resize(src_img3, (int(w / 1.5), int(h / 1.5)))
    src_img4 = cv2.resize(src_img4, (int(w / 1.5), int(h / 1.5)))

    # cv2.imshow('a', np.vstack((np.hstack((src_img0, src_img1)),
    #                            np.hstack((src_img2, src_img3)),
    #                            np.hstack((src_img4, np.zeros_like(src_img4))))))

    # cv2.waitKey(0)

    cv2.imwrite('reproject.png', np.vstack((np.hstack((src_img0, src_img1)),
                               np.hstack((src_img2, src_img3)),
                               np.hstack((src_img4, np.zeros_like(src_img4))))))
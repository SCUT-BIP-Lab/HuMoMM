import os
import sys
import cv2
import json
import argparse
import numpy as np
from termcolor import cprint

from utils.general import check_dir


def win_avg(depmap, row, col):
    h, w = depmap.shape

    lr = max(0, row - 2)
    rr = min(h, row + 3)
    lc = max(0, col - 2)
    rc = min(w, col + 3)

    depmap_plot = depmap[min(lr, rr):max(lr, rr), min(lc, rc):max(lc, rc)]
    avg_dep = depmap_plot.flatten()[np.nonzero(depmap_plot.flatten())].mean()

    return avg_dep


def split_depmap(depmap, joints_2d):
    h, w = depmap.shape
    padding = 20

    left = max(0, int(np.min(joints_2d[:, 0]) - padding))
    top = max(0, int(np.min(joints_2d[:, 1]) - padding))
    right = min(w, int(np.max(joints_2d[:, 0]) + padding))
    bottom = min(h, int(np.max(joints_2d[:, 1]) + padding))

    dep_roi = depmap[top:bottom, left:right]
    dep_roi = cv2.resize(dep_roi, (128, 256), interpolation=cv2.INTER_AREA)

    return dep_roi


def main(args):
    check_dir(args.output_root)
    if args.depmap:
        depmap_root = os.path.join(args.output_root, 'Depmap')
        check_dir(depmap_root)

    for cam_idx in range(5):
        cam_data_2d = []
        cam_data_2d_dep = []
        cam_data_3d = []
        root_file_list = os.listdir(args.data_root)
        root_file_list.sort()
        for sample_idx, sample_file in enumerate(root_file_list):
            if 'C000' not in sample_file:
                continue

            sample_data_2d = []
            sample_data_2d_dep = []
            sample_data_3d = []
            sample_data_depmap = []
            sample_file_cam = sample_file.replace('C000', f'C{cam_idx:03d}')

            frame_list = os.listdir(os.path.join(args.data_root, sample_file_cam))
            frame_list.sort()
            for frame in frame_list:
                frame_data_2d = []
                frame_data_2d_dep = []
                frame_data_3d = []
                js_path = os.path.join(args.data_root.replace('RGB_frame', args.label_file), sample_file_cam, frame.replace('jpg', 'json'))
                with open(js_path, 'r') as f:
                    js_data = json.load(f)
                    js_data['shapes'].sort(key=lambda x: int(x['label']))
                    for joint_data in js_data['shapes']:
                        frame_data_2d.append(joint_data['points'][0])
                        frame_data_3d.append(joint_data['points_3d'][0])

                depmap_path = os.path.join(args.dep_data_root, sample_file_cam, frame.replace('RF', 'DF').replace('jpg', 'png'))
                depmap = cv2.imread(depmap_path, cv2.IMREAD_ANYDEPTH)
                depmap = cv2.split(depmap)[0]

                for joint in frame_data_2d:
                    row = int(joint[1])
                    col = int(joint[0])
                    dep = depmap[row, col]

                    if dep == 0:
                        dep = win_avg(depmap, row, col)
                    frame_data_2d_dep.append([dep])

                if args.depmap:
                    depmap_roi = split_depmap(depmap, np.array(frame_data_2d))
                    sample_data_depmap.append(depmap_roi)

                sample_data_2d.append(frame_data_2d)
                sample_data_2d_dep.append(frame_data_2d_dep)
                sample_data_3d.append(frame_data_3d)
            cam_data_2d.append(sample_data_2d)
            cam_data_2d_dep.append(sample_data_2d_dep)
            cam_data_3d.append(sample_data_3d)

            if args.depmap:
                np.savez_compressed(os.path.join(depmap_root, f'data_depmap_cusdb_cam{cam_idx}_{sample_idx}'),
                                    positions_depmap=np.array(sample_data_depmap))  # 深度图

        np.savez_compressed(os.path.join(args.output_root, f'data_2d_cusdb_gt_cam{cam_idx}'),
                            positions_2d_rgb=np.array(cam_data_2d),  # rgb图片的二维坐标
                            positions_2d_dep=np.array(cam_data_2d_dep))  # 对应坐标深度
        np.savez_compressed(os.path.join(args.output_root, f'data_3d_cusdb_cam{cam_idx}'),
                            positions_3d=np.array(cam_data_3d))  # 三维坐标

    cprint('Finish generating training data package.', color='cyan', attrs=['bold'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../dataset/scut_key_frame/RGB_frame', help='path to the root of RGB frame.')
    parser.add_argument('--dep_data_root', type=str, default='../../dataset/scut/Depth_frame', help='path to the root of depth frame.')
    parser.add_argument('--label_file', type=str, default='Label_3d_ba', help='file of the label which you want, e.g. Label_3d or Label_3d_ba.')
    parser.add_argument('--output_root', type=str, default='../../dataset/scut_key_frame/Label_npz', help='path to the npz file.')
    parser.add_argument('--depmap', action='store_true', help='enable --depmap to generate depmap ROI package.')
    args = parser.parse_args()
    main(args)

import argparse
from glob import glob
import os
import sys
import json
from termcolor import cprint
from tqdm import tqdm
import numpy as np

from utils.data_labelling import write_txt_colmap
from utils.camera_utils import load_camera_param_pkg
from utils.general import check_dir


def main(args):
    prj_mats, extrinsics, intrinsics = load_camera_param_pkg(args.cam_param_path)

    colmap_txt_root = args.data_root.replace('RGB_frame', 'Colmap_txt')
    check_dir(colmap_txt_root)

    if args.sample_idx:
        sample_idx_list = []
        for item in args.sample_idx:
            item = item.split(',')
            sample_idx_list.append(f'C000P{int(item[0]):03d}R{int(item[1]):03d}A{int(item[2]):03d}')
    else:
        sample_idx_list = [item for item in os.listdir(args.data_root) if 'C000' in item]
    sample_idx_list.sort()

    for sample_idx, sample in enumerate(tqdm(sample_idx_list)):
        frame_list = os.listdir(os.path.join(args.data_root, sample))
        frame_list.sort()
        for frame in frame_list:
            frame_joints_3d = []
            frame_joints_2d = []

            # 3d joint of the main camera
            frame_js_path = os.path.join(args.data_root.replace('RGB_frame', 'Label_3d'), sample,
                                         frame.replace('.jpg', '.json')).replace('C000', 'C002')
            with open(frame_js_path, 'r') as f_3d:
                js_data_3d = json.load(f_3d)['shapes']
                js_data_3d.sort(key=lambda x: int(x['label']))
                for joint_data in js_data_3d:
                    frame_joints_3d.append(joint_data['points_3d'][0])

            for cam_num in range(5):
                frame_js_path = os.path.join(args.data_root.replace('RGB_frame', 'Label'), sample,
                                             frame.replace('.jpg', '.json')).replace('C000', f'C00{cam_num}')
                view_joints_2d = []
                with open(frame_js_path, 'r') as f:
                    js_data = json.load(f)
                    js_data['shapes'].sort(key=lambda x: int(x['label']))
                    for joint_data in js_data['shapes']:
                        if int(joint_data['label']) > len(view_joints_2d):
                            for push_joint in range(len(view_joints_2d), int(joint_data['label'])):
                                view_joints_2d.append([-1, -1])
                        view_joints_2d.append(joint_data['points'][0])
                    while len(view_joints_2d) < 21:
                        view_joints_2d.append([-1, -1])

                frame_joints_2d.append(view_joints_2d)
            frame_joints_3d = np.array(frame_joints_3d)
            frame_joints_2d = np.array(frame_joints_2d)
            write_txt_path = os.path.join(colmap_txt_root, frame[4:-4])
            if not os.path.exists(write_txt_path):
                os.mkdir(write_txt_path)
            write_txt_colmap(frame_joints_3d, frame_joints_2d, intrinsics, extrinsics, write_txt_path)

    cprint('Finish generating information txt for colmap.', color='cyan', attrs=['bold'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_param_path', type=str, default='./cam_param/cam_param_pkg.npz', help='path to the cam_param_pkg.npz.')
    parser.add_argument('--data_root', type=str, default='../../dataset/scut_key_frame/RGB_frame', help='path to the root of dataset.')
    parser.add_argument('--sample_idx', type=str, nargs='*', help='specified samples of if provided, corresponding to P, R and A, e.g. --sample_idx 1,1,1.')
    args = parser.parse_args()
    main(args)

import os
import sys
import json
import argparse
import numpy as np
import shutil
from tqdm import tqdm
import cv2
from termcolor import cprint

from utils.data_labelling import joint_3d_trans_cam, cam_reproject
from utils.general import check_dir, ske_vis


def l2_dis(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum(np.square(x-y)))


def main(args):
    check_dir(args.output_root)
    cam_param_pkg = np.load(args.cam_param_path, allow_pickle=True)['cam_param_pkg']

    vis_file = []
    if args.vis:
        vis_path = args.colmap_txt_root.replace('Colmap_txt', 'vis_reprojected_lost_joints_ba')
        check_dir(vis_path)

    frame_file_list = os.listdir(args.colmap_txt_root)
    frame_file_list.sort()
    ba_failed_frame = []
    for frame_file in tqdm(frame_file_list):
        txt_3d_path = os.path.join(args.colmap_txt_root, frame_file, 'colmap_out', 'points3D.txt')

        if not os.path.exists(txt_3d_path):
            ba_failed_frame.append(frame_file)
            for cam_idx in range(5):
                if not os.path.exists(os.path.join(args.output_root, f'C{cam_idx:03d}{frame_file[:12]}')):
                    os.mkdir(os.path.join(args.output_root, f'C{cam_idx:03d}{frame_file[:12]}'))

                js_path = os.path.join(args.update_root, f'C{cam_idx:03d}{frame_file[:12]}',
                                       f'C{cam_idx:03d}{frame_file}.json')
                shutil.copy(js_path, js_path.replace(args.update_root, args.output_root))
            continue

        joints_3d_lists = ['']*21
        with open(txt_3d_path, "r") as f:
            data = f.readlines()[3:]
            for line in data:
                joint_idx = int(line.split(' ')[0]) - 1
                joint_3d = list(map(float, line.split(' ')[1:4]))
                joints_3d_lists[joint_idx] = [joint_3d]
        joints_3d_lists = np.array(joints_3d_lists)

        for cam_idx in range(5):
            if not os.path.exists(os.path.join(args.output_root, f'C{cam_idx:03d}{frame_file[:12]}')):
                os.mkdir(os.path.join(args.output_root, f'C{cam_idx:03d}{frame_file[:12]}'))

            js_path = os.path.join(args.update_root, f'C{cam_idx:03d}{frame_file[:12]}', f'C{cam_idx:03d}{frame_file}.json')
            shutil.copy(js_path, js_path.replace(args.update_root, args.output_root))
            with open(js_path, 'r') as f:
                js_data = json.load(f)
                js_data['shapes'].sort(key=lambda x: int(x['label']))
            for joint_idx, joint_data in enumerate(js_data['shapes']):
                joint_3d_opt = joint_3d_trans_cam(joints_3d_lists[joint_idx], cam_idx, cam_param_pkg)
                # l2_distance = l2_dis(joint_data['points_3d'][0], joint_3d_opt)
                # if l2_distance > 300:
                #     cprint('Optimized joint is unreliable and has been discarded.', color='yellow', attrs=['bold'])
                #     cprint(f'file: C{cam_idx:03d}{frame_file}.json\njoint: {joint_idx}\nL2 distance: {l2_distance}\n', color='yellow')
                #     continue

                joint_data['points_3d'] = joint_3d_opt.tolist()
                if joint_data['vis'] == '0':
                    rep_joint_2d = cam_reproject(joints_3d_lists[joint_idx], cam_idx, cam_param_pkg)
                    joint_data['points'] = rep_joint_2d.tolist()

                    if args.vis and js_path not in vis_file:
                        vis_file.append(f'C{cam_idx:03d}{frame_file}.json')
            with open(js_path.replace(args.update_root, args.output_root), 'w') as dump_f:
                json.dump(js_data, dump_f)

    if len(ba_failed_frame) > 0:
        cprint(f'The bundle adjustment of following frames failed, please check their labels.', color='yellow')
        for item in ba_failed_frame:
            print(item)
        # 'P000R000A007RF038', 'P000R001A009RF064'

    if args.vis:
        cprint('Skeleton visualization...', color='blue')
        for file in tqdm(vis_file):
            json_path_2d = os.path.join(args.output_root, file[:16], file)
            joint_list = []
            with open(json_path_2d, 'r') as f:
                js_data = json.load(f)
                js_data['shapes'].sort(key=lambda x: int(x['label']))
                for joint_data in js_data['shapes']:
                    joint_list.append(joint_data['points'][0])

            src_img_path = os.path.join(args.colmap_txt_root.replace('Colmap_txt', 'RGB_frame'),
                                        file[:16],
                                        file.replace('json', 'jpg'))
            src_img = cv2.imread(src_img_path)
            tar_img = ske_vis(joint_list, src_img)

            if not os.path.exists(os.path.join(vis_path, file[:16])):
                os.mkdir(os.path.join(vis_path, file[:16]))

            tar_img_path = os.path.join(vis_path, file[:16], file.replace('.json', '.jpg'))
            cv2.imwrite(tar_img_path, tar_img)

    cprint('Finish updating labels.', color='cyan', attrs=['bold'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_txt_root', type=str, default='../../dataset/scut_key_frame/Colmap_txt', help='path to Colmap_txt.')
    parser.add_argument('--update_root', type=str, default='../../dataset/scut_key_frame/Label_3d', help='path to the 3d label file which need to be updated.')
    parser.add_argument('--output_root', type=str, default='../../dataset/scut_key_frame/Label_3d_ba', help='path to the label file after BA.')
    parser.add_argument('--cam_param_path', type=str, default='./cam_param/cam_param_pkg.npz', help='path to the cam_param_pkg.npz.')
    parser.add_argument('--vis', action='store_true', help='enable --vis to visualize the reprojected 2d label.')
    args = parser.parse_args()
    main(args)

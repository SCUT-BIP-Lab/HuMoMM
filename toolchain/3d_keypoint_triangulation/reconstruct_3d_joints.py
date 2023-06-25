import os
import sys
import shutil
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm
from termcolor import cprint

from utils.general import check_dir, ske_vis
from utils.data_labelling import get_3d_joints, cam_reproject, min_rep_err, joint_3d_trans_cam


def main(args):
    check_dir(args.output_root)

    vis_file = []
    if args.vis:
        vis_path = args.data_root.replace('Label', 'vis_reprojected_lost_joints')
        check_dir(vis_path)

    cam_param_pkg = np.load(args.cam_param_path, allow_pickle=True)['cam_param_pkg']

    root_file_list = os.listdir(args.data_root)
    root_file_list.sort()
    for video_file in tqdm(root_file_list):
        if 'C000' not in video_file:
            continue
        file_list = os.listdir(os.path.join(args.data_root, video_file))
        file_list.sort()
        for file in file_list:
            frame_list = []
            for cam_idx in range(5):
                if not os.path.exists(os.path.join(args.output_root, video_file).replace('C000', f'C00{cam_idx}')):
                    os.mkdir(os.path.join(args.output_root, video_file).replace('C000', f'C00{cam_idx}'))

                json_path = os.path.join(args.data_root, video_file, file).replace('C000', f'C00{cam_idx}')
                shutil.copy(json_path, json_path.replace(args.data_root, args.output_root))
                with open(json_path, 'r') as f:
                    js_data = json.load(f)
                    js_data['shapes'].sort(key=lambda x: int(x['label']))
                    joint_list = []

                    for joint_data in js_data['shapes']:
                        if int(joint_data['label']) > len(joint_list):
                            for push_joint in range(len(joint_list), int(joint_data['label'])):
                                joint_list.append([None, None])
                        joint_list.append(joint_data['points'][0])
                    while len(joint_list) < 21:
                        joint_list.append([None, None])

                frame_list.append(joint_list)

            frame_list = np.array(frame_list)
            for joint_idx in range(frame_list.shape[1]):
                joint_idx_3d_all = []
                joint_lost_cam_idx = []

                for main_cam_idx in range(frame_list.shape[0]):
                    if frame_list[main_cam_idx, joint_idx, 0] is None:
                        if file.replace('C000', f'C00{main_cam_idx}') not in vis_file:
                            vis_file.append(file.replace('C000', f'C00{main_cam_idx}'))
                        joint_lost_cam_idx.append(main_cam_idx)
                        continue

                    for vice_cam_idx in range(main_cam_idx+1, frame_list.shape[0]):
                        if frame_list[vice_cam_idx, joint_idx, 0] is None:
                            continue

                        joint_idx_3d = get_3d_joints(np.array([frame_list[main_cam_idx, joint_idx]], dtype=float),
                                                     np.array([frame_list[vice_cam_idx, joint_idx]], dtype=float),
                                                     main_cam_idx,
                                                     vice_cam_idx,
                                                     cam_param_pkg)
                        joint_idx_3d_all.append(joint_idx_3d)

                        joint_idx_3d = get_3d_joints(np.array([frame_list[vice_cam_idx, joint_idx]], dtype=float),
                                                     np.array([frame_list[main_cam_idx, joint_idx]], dtype=float),
                                                     vice_cam_idx,
                                                     main_cam_idx,
                                                     cam_param_pkg)
                        joint_idx_3d_all.append(joint_idx_3d)

                joint_idx_3d_opt = min_rep_err(np.array(joint_idx_3d_all))

                for cam_idx in range(5):
                    json_path_2d = os.path.join(args.output_root, video_file, file).replace('C000', f'C00{cam_idx}')
                    with open(json_path_2d, 'r') as f:
                        js_data = json.load(f)
                        js_data['shapes'].sort(key=lambda x: int(x['label']))
                        if cam_idx in joint_lost_cam_idx:
                            # print(file.replace('C000', f'C00{cam_idx}'), joint_idx_3d_opt.shape, joint_idx)
                            rep_2d_joint = cam_reproject(joint_idx_3d_opt, cam_idx, cam_param_pkg)
                            js_data['shapes'].insert(joint_idx,
                                {
                                    'label': '%02d' % joint_idx,
                                    'points': rep_2d_joint.tolist(),
                                    'vis': '0'
                                }
                            )
                        else:
                            js_data['shapes'][joint_idx]['vis'] = '1'
                        joint_idx_3d_opt_cam = joint_3d_trans_cam(joint_idx_3d_opt, cam_idx, cam_param_pkg)
                        js_data['shapes'][joint_idx]['points_3d'] = joint_idx_3d_opt_cam.tolist()

                    with open(json_path_2d, 'w') as dump_f:
                        json.dump(js_data, dump_f)

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

            src_img_path = os.path.join(args.data_root.replace('Label', 'RGB_frame'),
                                        file[:16],
                                        file.replace('json', 'jpg'))
            src_img = cv2.imread(src_img_path)
            tar_img = ske_vis(joint_list, src_img)

            if not os.path.exists(os.path.join(vis_path, file[:16])):
                os.mkdir(os.path.join(vis_path, file[:16]))

            tar_img_path = os.path.join(vis_path, file[:16], file.replace('.json', '.jpg'))
            cv2.imwrite(tar_img_path, tar_img)

    cprint('Finish generating 3D ground truth and reprojecting lost 2d joints.', color='cyan', attrs=['bold'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../dataset/scut_key_frame/Label', help='path to the root of the reconstructed dataset label.')
    parser.add_argument('--output_root', type=str, default='../../dataset/scut_key_frame/Label_3d', help='path to the root of reprojected 2d label and 3d label.')
    parser.add_argument('--cam_param_path', type=str, default='./cam_param/cam_param_pkg.npz', help='path to the cam_param_pkg.npz.')
    parser.add_argument('--vis', action='store_true', help='enable --vis to visualize the reprojected lost 2d label.')
    args = parser.parse_args()
    main(args)


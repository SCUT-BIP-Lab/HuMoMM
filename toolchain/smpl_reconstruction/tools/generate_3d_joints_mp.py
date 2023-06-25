import os
import argparse
import glob
import shutil
import copy
import os.path as osp
import cv2
import torch
import numpy as np
from subprocess import PIPE, STDOUT, Popen
from easymocap.mytools.camera_utils import read_camera
from easymocap.mytools.file_utils import read_json, save_json
from tools.konia import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from tools.process_2d_label import process_2d_labels

from tools.process_cam_param import process_cam_param


def exe_command(command):
    """
    执行 shell 命令并实时打印输出
    :param command: shell 命令
    :return: process, exitcode
    """
    print(command)
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip())
    exitcode = process.wait()
    return process, exitcode


def trans_3d_joints_with_cam(joints_annots, cam_param):
    R = cam_param['R']
    RT = cam_param['RT']
    results={}
    results['shapes']=[]
    for i,  annots in enumerate(joints_annots):
        person_id=annots['id']
        joints_3d=annots['keypoints3d']
        if len(joints_3d)!=21:
            print('num joints less than 21, exit!')
            exit()
        for joint_idx, joint in enumerate(joints_3d):
            if joint[-1]==0.: # unvisble
                trans_joints=[[0., 0., 0.],]
                results['shapes'].append({
                    'label': f'{joint_idx:02d}',
                    'vis': '0', 
                    "group_id": f'{person_id}',
                    'points_3d': trans_joints
                })
            else:
                homo_joint = np.array(joint[0:3] + [1.0, ])[:, None]
                trans_joints = RT @ homo_joint
                results['shapes'].append({
                    'label': f'{joint_idx:02d}',
                    'vis': '1', 
                    "group_id": f'{person_id}',
                    'points_3d': trans_joints.T.tolist()
                })
    return results


def main(args):
    video_pathes = sorted(glob.glob(osp.join(args.data_root, 'RGB_frame', 'C000P*R*A*')))

    for video_path in video_pathes:
        mv_video_pathes = sorted(glob.glob(video_path.replace('C000', 'C*')))
        if osp.isdir(args.tmp_dir):
            shutil.rmtree(args.tmp_dir)
        os.makedirs(args.tmp_dir)
        # copy video and 2d label to tmp dir
        for video_dir in mv_video_pathes:
            video_name = video_dir.split('/')[-1]
            shutil.copytree(video_dir, os.path.join(args.tmp_dir, 'images', video_name), copy_function=shutil.copy2)
            shutil.copytree(video_dir.replace('RGB_frame', 'Label'), os.path.join(
                args.tmp_dir, 'annots_ori', video_name), copy_function=shutil.copy2)

        # process 2d label
        process_2d_labels(args.tmp_dir, 'body21')

        # copy and process cam param
        cam_param_path = osp.join(args.tmp_dir, 'cam_param_pkg.npz')
        shutil.copy2(args.cam_param_path, cam_param_path)
        process_cam_param(args.tmp_dir, video_name[4:])

        # generate multi person smpl label
        exe_command(
            f"python apps/demo/mvmp.py {args.tmp_dir} --out {args.tmp_dir}/output --cfg config/exp/my_mvmp1f_body21.yml --annot annots --body body21 --undis ") #--vis_repro --vis_det
        exe_command(
            f"python apps/demo/auto_track.py {args.tmp_dir}/output {args.tmp_dir}/output-track --track3d --body body21")

        # trans joints annot with cam and save to output dir
        src_joints_files = sorted(glob.glob(osp.join(args.tmp_dir, 'output-track', 'keypoints3d', '*.json')))

        cam_params = read_camera(os.path.join(args.tmp_dir, 'intri.yml'),
                                    os.path.join(args.tmp_dir, 'extri.yml'))

        for frame_idx, joints_file in enumerate(src_joints_files):
            joints_annot = read_json(joints_file)
            for cam_name in cam_params['basenames']:
                trans_joints_annot =trans_3d_joints_with_cam(joints_annot, cam_params[cam_name])
                joints_path = osp.join(args.output_joint_root, cam_name, f'{cam_name}RF{frame_idx:03d}.json')
                save_json(joints_path, trans_joints_annot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../dataset/scut_key_frame',
                        help='path to the root of the dataset')
    parser.add_argument('--output_joint_root', type=str, default='../../dataset/scut_key_frame/Label_3d',
                        help='path to the generated 3d joints labels.')
    parser.add_argument('--tmp_dir', type=str, default='tmp/sequence', help='path to restore the intermediate data.')
    parser.add_argument('--cam_param_path', type=str, default='./cam_param/cam_param_pkg.npz',
                        help='path to the cam_param_pkg.npz.')
    parser.add_argument('--vis', action='store_true', help='enable --vis to visualize the reprojected lost 2d label.')
    args = parser.parse_args()
    main(args)

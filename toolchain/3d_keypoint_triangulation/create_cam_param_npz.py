import os
import sys
import argparse
import scipy.io as sio
import numpy as np
import yaml
from termcolor import cprint


def print_npz(npz_path):
    # param vis format exp:
    # cam2_0_vice_cam_rot = np.load('cam_param_pkg.npz')['cam_param_pkg'][0]['cam2_0']['vice_cam_rot']

    cam_param_pkg = np.load(npz_path, allow_pickle=True)['cam_param_pkg']
    for i in range(len(cam_param_pkg)):

        for pair_idx in cam_param_pkg[i]:
            cprint('camera pair: %s' % pair_idx, color='blue', attrs=['bold'])

            for item_mat in cam_param_pkg[i][pair_idx]:
                cprint('\n'+item_mat, color='green', attrs=['bold'])
                cprint(cam_param_pkg[i][pair_idx][item_mat], color='green')
        print('\n')


def main(args):
    cam_param_pkg = []
    root_file_list = os.listdir(args.data_root)
    root_file_list.sort()
    for file in root_file_list:
        if 'param.mat' not in file:
            continue
        mat_file = sio.loadmat(os.path.join(args.data_root, file))

        cam_param_item = {
            'main_cam_param': np.array(mat_file['main_cam_param']).T,
            'vice_cam_param': np.array(mat_file['vice_cam_param']).T,
            'vice_cam_rot': np.array(mat_file['vice_cam_rot']).T,
            'vice_cam_trans': np.array(mat_file['vice_cam_trans']).T
        }

        cam_param_pkg.append({
            file[:6]: cam_param_item
        })

    yaml_path = os.path.join(args.data_root, 'cam2_4_param.yaml')
    with open(yaml_path, 'r') as f:
        cam2_4_param = yaml.load(f, Loader=yaml.FullLoader)

        main_intr_param = np.array([[cam2_4_param['cam0']['intrinsics'][0], 0, cam2_4_param['cam0']['intrinsics'][2]],
                                    [0, cam2_4_param['cam0']['intrinsics'][1], cam2_4_param['cam0']['intrinsics'][3]],
                                    [0, 0, 1]])
        vice_intr_param = np.array([[cam2_4_param['cam1']['intrinsics'][0], 0, cam2_4_param['cam1']['intrinsics'][2]],
                                    [0, cam2_4_param['cam1']['intrinsics'][1], cam2_4_param['cam1']['intrinsics'][3]],
                                    [0, 0, 1]])
        cam_param_item = {
            'main_cam_param': main_intr_param,
            'vice_cam_param': vice_intr_param,
            'vice_cam_rot': np.array(cam2_4_param['cam1']['T_cn_cnm1'])[0:3, 0:3],
            'vice_cam_trans': np.array(cam2_4_param['cam1']['T_cn_cnm1'])[0:3, 3:4]*1000
        }

        cam_param_pkg.append({
            'cam2_4': cam_param_item
        })

    # 相机参数微调，按需添加，不需要则删除
    cam_param_pkg[0]['cam2_0']['vice_cam_trans'][0, 0] += 50
    cam_param_pkg[1]['cam2_1']['vice_cam_trans'][0, 0] -= 150
    cam_param_pkg[0]['cam2_0']['vice_cam_rot'][0, 0] -= 0.1
    cam_param_pkg[1]['cam2_1']['vice_cam_rot'][1, 1] += 0.1

    npz_path = os.path.join(args.data_root, 'cam_param_pkg.npz')
    np.savez(npz_path, cam_param_pkg=cam_param_pkg)

    print_npz(npz_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./cam_param', help='path to the root of input mat file.')
    parser.add_argument('--output_root', type=str, default='./cam_param', help='path to the root of output npz file.')
    args = parser.parse_args()
    main(args)

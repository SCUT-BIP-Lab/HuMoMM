import argparse
import os
import sys
import shutil
import cv2
from tqdm import tqdm
import numpy as np

from utils.frame_diff import thres_diff
from utils.general import check_dir


def main(args):
    """
    Extract key frames according to the designated camera and other cameras align to it.
    :param args:
    :return: Key frame sub-dataset.
    """

    check_dir(args.output_root)
    src_path = os.path.join(args.data_root, 'RGB_frame')
    align_cams = [f'C00{i}' for i in range(5) if i != args.key_frame_cam]
    if args.ig_p:
        ignore_P = [int(idx) for idx in args.ig_p.split(',')]
    else:
        ignore_P = []

    file_list = os.listdir(src_path)
    file_list.sort()
    for file in file_list:
        if int(file[5:8]) in ignore_P:
            # 跳过不需要处理的人的数据
            continue

        tar_path = os.path.join(args.output_root, 'RGB_frame', file)
        check_dir(tar_path)
        if file[:4] in align_cams:
            continue
        names = os.listdir(os.path.join(src_path, file))
        names.sort()
        frames = np.asarray([cv2.imread(os.path.join(src_path, file, name)) for name in sorted(names)])
        _, key_idx = thres_diff(frames, key_frame_number=int(len(names) * float(args.ex_rate)))
        print(f'{file} key frame indices: {key_idx}')

        for idx in key_idx:
            cv2.imwrite(os.path.join(tar_path, names[idx]), frames[idx])

            auto_cam = f'C00{args.key_frame_cam}'
            for align_cam in align_cams:
                shutil.copy(os.path.join(src_path, file, names[idx]).replace(auto_cam, align_cam),
                            os.path.join(tar_path, names[idx]).replace(auto_cam, align_cam))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../dataset/scut', help='path to the root of raw dataset.')
    parser.add_argument('--output_root', type=str, default='../../dataset/scut_key_frame', help='path to the root of output dataset.')
    parser.add_argument('--key_frame_cam', type=int, default=4, help='camera used to extract key frames, from 0 to 4.')
    parser.add_argument('--ex_rate', type=str, default='0.2', help='ratio of key frame extraction.')
    parser.add_argument('--ig_p', type=str, help='index of ignored persons, e.g. 0,1,2,3.')
    args = parser.parse_args()
    main(args)

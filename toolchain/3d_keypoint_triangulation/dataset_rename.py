import os
import shutil
import argparse
from tqdm import tqdm

from utils.general import check_dir


def main(args):
    """
    Rename the original dataset.
    :param args:
    :return: File name format:
                C: camera num
                P: person num
                R: repeat num
                A: action num
                R/D: RGB or depth image
                F: frame index
    """

    src_path = args.data_root
    tar_path = args.output_root

    check_dir(tar_path)
    tar_path_rgb = os.path.join(tar_path, 'RGB_frame')
    check_dir(tar_path_rgb)
    tar_path_d = os.path.join(tar_path, 'Depth_frame')
    check_dir(tar_path_d)

    P_len = len(os.listdir(src_path))
    R = 2
    start_P = 15
    for p_item in tqdm(range(start_P, start_P+P_len)):
        A = len(os.listdir(os.path.join(src_path, f'sample_{p_item}')))//R
        for a_item in range(A):
            for r_item in range(R):
                C = len(os.listdir(os.path.join(src_path,
                                                f'sample_{p_item}',
                                                f'{a_item}_{r_item}')))//2
                for c_item in range(C):
                    sub_tar_path_d = os.path.join(tar_path_d,
                                                  'C%03dP%03dR%03dA%03d' % (c_item, p_item, r_item, a_item))
                    sub_tar_path_rgb = os.path.join(tar_path_rgb,
                                                    'C%03dP%03dR%03dA%03d' % (c_item, p_item, r_item, a_item))
                    check_dir(sub_tar_path_d)
                    check_dir(sub_tar_path_rgb)

                    sub_path = os.path.join(src_path,
                                            f'sample_{p_item}',
                                            f'{a_item}_{r_item}',
                                            f'realsense{c_item}_depth')
                    F = len(os.listdir(sub_path))
                    if F > 75:
                        print(f'Discard overflow frames {F} in: ', sub_path)
                        F = 75
                    for f_item in range(F):
                        img_src_path = os.path.join(sub_path,
                                                    '%03d.png' % f_item)
                        img_tar_path = os.path.join(sub_tar_path_d,
                                                    'C%03dP%03dR%03dA%03dDF%03d.png' %
                                                    (c_item, p_item, r_item, a_item, f_item))
                        shutil.copy(img_src_path, img_tar_path)

                    sub_path = os.path.join(src_path,
                                            f'sample_{p_item}',
                                            f'{a_item}_{r_item}',
                                            f'realsense{c_item}_rgb')
                    F = len(os.listdir(sub_path))
                    if F > 75:
                        print(f'Discard overflow frames {F} in: ', sub_path)
                        F = 75
                    for f_item in range(F):
                        img_src_path = os.path.join(sub_path,
                                                    '%03d.jpg' % f_item)
                        img_tar_path = os.path.join(sub_tar_path_rgb,
                                                    'C%03dP%03dR%03dA%03dRF%03d.jpg' %
                                                    (c_item, p_item, r_item, a_item, f_item))
                        shutil.copy(img_src_path, img_tar_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../dataset/scut_ori', help='path to the root of raw dataset.')
    parser.add_argument('--output_root', type=str, default='../../dataset/scut', help='path to the root of output dataset.')
    args = parser.parse_args()
    main(args)

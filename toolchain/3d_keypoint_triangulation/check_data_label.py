import os, sys
import shutil
import argparse
import json
from termcolor import cprint
from tqdm import tqdm
import numpy as np


def main(args):
    label_path = os.path.join(args.data_root, 'Label')
    rgb_path = os.path.join(args.data_root, 'RGB_frame')
    lost_jsons = []

    if not os.path.exists(label_path):
        os.mkdir(label_path)

    if os.path.exists(os.path.join(args.data_root, 'lost_jsons.txt')):
        os.remove(os.path.join(args.data_root, 'lost_jsons.txt'))
    if os.path.exists(os.path.join(args.data_root, 'refine_joint.txt')):
        os.remove(os.path.join(args.data_root, 'refine_joint.txt'))

    video_file_lists = os.listdir(rgb_path)
    video_file_lists.sort()
    for video_file in video_file_lists:
        if not os.path.exists(os.path.join(label_path, video_file)):
            os.mkdir(os.path.join(label_path, video_file))
        file_list = os.listdir(os.path.join(rgb_path, video_file))
        file_list.sort()
        for file in file_list:
            if '.jpg' in file:
                if not os.path.exists(os.path.join(rgb_path, video_file, file).replace('jpg', 'json')):
                    lost_jsons.append(file)
                else:
                    src_path = os.path.join(rgb_path, video_file, file.replace('jpg', 'json'))
                    tar_path = os.path.join(label_path, video_file, file.replace('jpg', 'json'))
                    shutil.move(src_path, tar_path)

    label_num = np.sum(np.array([len(os.listdir(os.path.join(label_path, item))) for item in os.listdir(label_path)]))
    rgb_num = np.sum(np.array([len(os.listdir(os.path.join(rgb_path, item))) for item in os.listdir(rgb_path)]))

    if label_num == rgb_num:
        lost_jsons = []

    if len(lost_jsons) != 0:
        with open(os.path.join(args.data_root, 'lost_jsons.txt'), "w") as f:
            for lost_idx in lost_jsons:
                f.write(lost_idx+'\n')
        cprint(f'Find lost labels of {len(lost_jsons)} frames, '
               f'frame idx have been saved in {args.data_root}/lost_jsons.txt',
               color='yellow', attrs=['bold'])
        sys.exit()

    cam0_2d_label = []
    cam1_2d_label = []
    cam2_2d_label = []
    cam3_2d_label = []
    cam4_2d_label = []

    wrong_label = []

    video_file_list = os.listdir(label_path)
    video_file_list.sort()
    cprint('Packing...', color='blue')
    for video_file in tqdm(video_file_list):
        js_file_list = os.listdir(os.path.join(label_path, video_file))
        js_file_list.sort()
        for js_file in js_file_list:
            single_frame_points = []
            with open(os.path.join(label_path, video_file, js_file),'r',encoding='utf8') as f:
                js_data = json.load(f)
                js_data['shapes'].sort(key=lambda x: int(x['label']))
                for joint_data in js_data['shapes']:
                    if int(joint_data['label']) > len(single_frame_points):
                        for push_joint in range(len(single_frame_points), int(joint_data['label'])):
                            single_frame_points.append({
                                'label': '%02d' % push_joint,
                                'points': None
                            })
                    single_frame_points.append({
                        'label': joint_data['label'],
                        'points': joint_data['points']
                    })
            while len(single_frame_points) < 21:
                single_frame_points.append({
                    'label': '%02d' % len(single_frame_points),
                    'points': None
                })

            if len(single_frame_points) > 21:
                wrong_label.append({
                    'js_file': js_file,
                    'joint_inf': single_frame_points
                })

            if 'C000' in js_file:
                cam0_2d_label.append({
                    'frame_idx': js_file,
                    'label': single_frame_points
                })
            elif 'C001' in js_file:
                cam1_2d_label.append({
                    'frame_idx': js_file,
                    'label': single_frame_points
                })
            elif 'C002' in js_file:
                cam2_2d_label.append({
                    'frame_idx': js_file,
                    'label': single_frame_points
                })
            elif 'C003' in js_file:
                cam3_2d_label.append({
                    'frame_idx': js_file,
                    'label': single_frame_points
                })
            elif 'C004' in js_file:
                cam4_2d_label.append({
                    'frame_idx': js_file,
                    'label': single_frame_points
                })

    if len(wrong_label) > 0:
        cprint(f'number of joints > 21, please check this label.', color='yellow')
        for item in wrong_label:
            cprint(item['js_file'], color='yellow')
            print(item['joint_inf'])
            print('\n')
        sys.exit()

    refine_joint = []
    max_unlabeled_view_point = 3
    cprint('Checking...', color='blue')
    for label_idx in tqdm(range(len(cam0_2d_label))):
        for joint_idx in range(len(cam0_2d_label[label_idx]['label'])):
            refine_cam = ''
            unlableled_count = 0
            if cam0_2d_label[label_idx]['label'][joint_idx]['points'] is None:
                unlableled_count += 1
                refine_cam += 'C000  '
            if cam1_2d_label[label_idx]['label'][joint_idx]['points'] is None:
                unlableled_count += 1
                refine_cam += 'C001  '
            if cam2_2d_label[label_idx]['label'][joint_idx]['points'] is None:
                unlableled_count += 1
                refine_cam += 'C002  '
            if cam3_2d_label[label_idx]['label'][joint_idx]['points'] is None:
                unlableled_count += 1
                refine_cam += 'C003  '
            if cam4_2d_label[label_idx]['label'][joint_idx]['points'] is None:
                unlableled_count += 1
                refine_cam += 'C004  '

            if unlableled_count > max_unlabeled_view_point:
                info = cam0_2d_label[label_idx]['frame_idx'].replace('C000', 'C00*') + f'    joint idx:  {joint_idx:02d}    camera idx:  {refine_cam}'
                refine_joint.append(info)

    if len(refine_joint) != 0:
        cprint(f'{len(refine_joint)} joints need to be refined, '
               f'related information has been saved in {args.data_root}/refine_joint.txt',
               color='yellow', attrs=['bold'])
        with open(os.path.join(args.data_root, 'refine_joint.txt'), "w") as f:
            for refine_item in refine_joint:
                f.write(refine_item+'\n\n')
    else:
        cprint('Check over, all labels are qualified.', color='green')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../dataset/scut_key_frame', help='path to the root of labeled key frame dataset.')
    args = parser.parse_args()
    main(args)

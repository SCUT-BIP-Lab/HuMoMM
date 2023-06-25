import os
import pickle
import numpy as np
import json


def read_skeleton_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
        frame_infos = js['shapes']

    body_joint = {}
    for frame_info in frame_infos:
        body_joint[frame_info['label'].zfill(2)] = frame_info['points'][0]

    return body_joint  # 返回skeleton_sequence字典


def prepare_label(data_path, out_path, part='train'):
    prepare_annotations = []                                 # 建立prepare_annotations列表
    video_names = sorted(os.listdir(data_path))                # 返回路径下包含的文件或文件夹的名字的列表，并排序赋给filenames
    for i, video_name in enumerate(video_names):                 # 组合为一个索引序列,这一步工作是取出s001
        frame_names = sorted(os.listdir(os.path.join(data_path, video_name)))
        for frame_idx, frame_name in enumerate(frame_names):
            if part == 'train' and frame_idx != len(frame_names) - 1:
                issample = True
            elif part == 'val' and frame_idx == len(frame_names) - 1:
                issample = True
            else:
                issample = False
            if issample:
                skeleton_anno_path = os.path.join(data_path, video_name, frame_name)      # 将路径和文件名连起来
                # 调用之前写好的函数，将skeleton_sequence字典赋给skeleton_sequence
                skeleton_sequence = read_skeleton_file(skeleton_anno_path)
                annotation = {}
                annotation['img_paths'] = f'RGB_frame/{video_name}/{frame_name[:-5]}.jpg'
                annotation["img_id"] = int(frame_name[1:4] + frame_name[5:8] +
                                           frame_name[9:12] + frame_name[13:16] + frame_name[18:21])
                annotation['img_width'] = 640
                annotation['img_height'] = 480
                annotation['body_id'] = '001'
                keypoints = []
                visible_keypoints = []
                for label in range(21):
                    if skeleton_sequence.get(f'{label:02d}') == None:
                        keypoint = [0, 0, 0]  # 0: "invisible", 1: "occlude", 2: "visible"
                    elif skeleton_sequence[f'{label:02d}'][0] == 0 and skeleton_sequence[f'{label:02d}'][1] == 0:
                        keypoint = [0, 0, 0]  # 0: "invisible", 1: "occlude", 2: "visible"
                    else:
                        keypoint = [skeleton_sequence[f'{label:02d}'][0],
                                    skeleton_sequence[f'{label:02d}'][1], 2]
                        visible_keypoints.append(keypoint)
                    keypoints.append(keypoint)
                annotation['keypoints'] = keypoints
                if not visible_keypoints: # no visible keypoints
                    continue
                # gen bbox and person_center
                visible_keypoints = np.array(visible_keypoints)
                x1, x2, y1, y2 = np.min(visible_keypoints[:, 0]), np.max(visible_keypoints[:, 0]), np.min(
                    visible_keypoints[:, 1]), np.max(visible_keypoints[:, 1])
                w, h = x2 - x1, y2 - y1
                annotation['bbox'] = [x1, y1, w, h]
                annotation['num_keypoints'] = len(visible_keypoints)
                prepare_annotations.append(annotation)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump(prepare_annotations, f)


if __name__ == '__main__':
    skeleton_dir = '/home/zzj/dataset/scut_key_frame/Label_3d_ba'
    out_dir = 'tmp/2d_labels'
    # prepare_label(skeleton_dir, out_dir, benchmark='xview', part='train')
    prepare_label(skeleton_dir, out_dir, part='train')
    prepare_label(skeleton_dir, out_dir, part='val')

import os
import pickle
import numpy as np
import json

training_subjects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
training_cameras = [0, 1, 2]
training_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def read_skeleton_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
        frame_infos = js['shapes']

    body_joint = {}
    for frame_info in frame_infos:
        body_joint[frame_info['label'].zfill(2)] = frame_info['points'][0]

    return body_joint  # 返回skeleton_sequence字典


def prepare_label(data_path, out_path, part='train', benchmark='xsub', net_input_size=256):
    prepare_annotations = []                                 # 建立prepare_annotations列表
    video_names = sorted(os.listdir(data_path))                # 返回路径下包含的文件或文件夹的名字的列表，并排序赋给filenames
    for i, video_name in enumerate(video_names):                 # 组合为一个索引序列,这一步工作是取出s001
        if benchmark == 'xsub':
            subject_id = int(video_name[video_name.find('P') + 1:video_name.find('P') + 4])
            istraining = subject_id in training_subjects
        elif benchmark == 'xaction':
            action_id = int(video_name[video_name.find('A') + 1:video_name.find('A') + 4])
            istraining = action_id in training_actions
        elif benchmark == 'xview':
            camera_id = int(video_name[video_name.find('C') + 1:video_name.find('C') + 4])
            istraining = camera_id in training_cameras
        else:
            raise ValueError('The benchmark is invalid, please check!')
        if part == 'train':
            issample = istraining
        else:
            issample = not istraining
        frame_names = sorted(os.listdir(os.path.join(data_path, video_name)))
        for frame_idx, frame_name in enumerate(frame_names):
            if issample:
                skeleton_anno_path = os.path.join(data_path, video_name, frame_name)      # 将路径和文件名连起来
                # 调用之前写好的函数，将skeleton_sequence字典赋给skeleton_sequence
                skeleton_sequence = read_skeleton_file(skeleton_anno_path)
                annotation = {}
                annotation['img_paths'] = f'{video_name}/{frame_name[:-5]}.jpg'
                annotation["img_id"] = int(frame_name[1:4] + frame_name[5:8] +
                                           frame_name[9:12] + frame_name[13:16] + frame_name[18:21])
                annotation['img_width'] = 640
                annotation['img_height'] = 480
                annotation['body_id'] = '001'
                keypoints = []
                visible_keypoints = []
                for label in range(21):
                    if skeleton_sequence.get(f'{label:02d}') == None:
                        keypoint = [0, 0, 2]  # 1: "visible" 2: "occlude" or "invisible"
                    elif skeleton_sequence[f'{label:02d}'][0] == 0 and skeleton_sequence[f'{label:02d}'][1] == 0:
                        keypoint = [0, 0, 2]  # 1: "visible" 2: "occlude" or "invisible"
                    else:
                        keypoint = [skeleton_sequence[f'{label:02d}'][0],
                                    skeleton_sequence[f'{label:02d}'][1], 1]
                        visible_keypoints.append(keypoint)
                    keypoints.append(keypoint)
                annotation['keypoints'] = keypoints
                if not visible_keypoints:
                    continue
                # gen bbox and person_center
                visible_keypoints = np.array(visible_keypoints)
                x1, x2, y1, y2 = np.min(visible_keypoints[:, 0]), np.max(visible_keypoints[:, 0]), np.min(
                    visible_keypoints[:, 1]), np.max(visible_keypoints[:, 1])
                w, h = x2 - x1, y2 - y1
                annotation['bbox'] = [x1, y1, w, h]
                annotation['objpos'] = [x1 + w / 2, y1 + h / 2]
                annotation['scale_provided'] = h / net_input_size
                annotation['num_keypoints'] = len(visible_keypoints)
                prepare_annotations.append(annotation)
                # TODO: provide "processed_other_annotations" for multi person
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump(prepare_annotations, f)


if __name__ == '__main__':
    benchmark='xview'
    skeleton_dir = '/data/pose_datasets/scut_sp/Label_3d'
    out_dir = '/home/zx/exp_wmh/data/2d_pose_anno'
    prepare_label(skeleton_dir, out_dir, part='train', benchmark=benchmark, net_input_size=256)

import json
import os
import pickle
import random
import numpy as np

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]


train_freq = 10


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def gen_coco_style_json(data_path, out_path, ignored_sample_path=None, benchmark='xview'):
    val_annotations = {}
    val_annotations['info'] = {
        "description": "NTU RGBD Dataset",
        "url": "",
        "version": "",
        "year": 2017,
        "contributor": "",
        "date_created": ""
    },
    val_annotations['licences'] = []
    val_annotations['images'] = []
    val_annotations['annotations'] = []

    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    filenames = sorted(os.listdir(data_path))
    id_ = 0
    img_id = 0

    random.seed(2333)
    random.shuffle(filenames)
    val_num = 100

    for i, filename in enumerate(filenames):
        # use S001 for training
        if filename[:4] != 'S001':
            continue
        print(f'procressing video: {filename[:-9]}')
        if filename in ignored_samples:
            continue

        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        issample = not (istraining)
        flag = 0
        if issample:
            skeleton_anno_path = os.path.join(data_path, filename)
            skeleton_sequence = read_skeleton_filter(skeleton_anno_path)
            for frame_i in range(skeleton_sequence['numFrame']):
                if frame_i % train_freq != 0:
                    continue
                image_info = {
                    "license": 1,
                    "file_name": f'{filename[:-9]}_rgb/{frame_i:03d}.jpg',
                    "coco_url": "",
                    "height": 1080,
                    "width": 1920,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": img_id
                }
                val_annotations['images'].append(image_info)

                frame_info = skeleton_sequence['frameInfo'][frame_i]
                for body_info in frame_info['bodyInfo']:
                    annotation = {}
                    annotation['id'] = id_
                    annotation['image_id'] = img_id
                    annotation['category_id'] = 1
                    keypoints = []
                    visible_keypoints = []
                    for join_info in body_info['jointInfo']:
                        if np.isnan(join_info['colorX']) or np.isnan(join_info['colorY']):
                            keypoint = [0, 0, 0]
                        else:
                            keypoint = [join_info['colorX'], join_info['colorY'], 2]
                            visible_keypoints.append(keypoint)
                        keypoints.extend(keypoint)
                    visible_keypoints = np.array(visible_keypoints)
                    annotation['num_keypoints'] = len(visible_keypoints)
                    annotation['keypoints'] = keypoints
                    x1, x2, y1, y2 = np.min(visible_keypoints[:, 0]), np.max(visible_keypoints[:, 0]), np.min(
                        visible_keypoints[:, 1]), np.max(visible_keypoints[:, 1])
                    w, h = x2-x1, y2-y1
                    annotation['bbox'] = [x1, y1, w, h]
                    annotation['area'] = w*h
                    annotation['iscrowd'] = 1 if frame_info['numBody'] > 1 else 0
                    annotation['segmentation'] = []
                    val_annotations['annotations'].append(annotation)
                    id_ += 1
                img_id += 1
                if len(val_annotations['images']) > val_num:
                    flag = 1
                    break
        if flag == 1:
            break
    val_annotations["categories"] = [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": ["base of the spine", "middle of the spine", "neck", "head", "left shoulder", "left elbow", "left wrist",
                          "left hand", "right shoulder", "right elbow", "right wrist ", "right hand", "left hip", "left knee",
                          "left ankle", "left foot", "right hip", "right knee", "right ankle", "right foot", "spine",
                          "tip of the left hand", "left thumb", "tip of the right hand", "right thumb", ],
            "skeleton": [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
            ]
        }
    ]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open('{}/val_{}_label_subset100.json'.format(out_path, benchmark), 'w') as f:
    # with open('{}/val_{}_label.json'.format(out_path, benchmark), 'w') as f:
        json.dump(val_annotations, f, indent=4)


if __name__ == '__main__':
    skeleton_dir = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_skeletons'
    out_dir = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_2d_labels'
    # gen_coco_style_json(skeleton_dir, out_dir, benchmark='xview')
    gen_coco_style_json(skeleton_dir, out_dir, benchmark='xsub')

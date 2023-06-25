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


def gen_coco_style_json(data_path, out_path):
    val_annotations = {}
    val_annotations['info'] = {
        "description": "SCUT RGBD Dataset",
        "url": "",
        "version": "",
        "year": 2022,
        "contributor": "",
        "date_created": ""
    },
    val_annotations['licences'] = []
    val_annotations['images'] = []
    val_annotations['annotations'] = []

    id_ = 0
    video_names = sorted(os.listdir(data_path))                # 返回路径下包含的文件或文件夹的名字的列表，并排序赋给filenames
    for i, video_name in enumerate(video_names):                 # 组合为一个索引序列,这一步工作是取出s001
        frame_names = sorted(os.listdir(os.path.join(data_path, video_name)))
        for frame_idx, frame_name in enumerate(frame_names):
            if frame_idx == len(frame_names) - 1:
                issample = True
            else:
                issample = False
            if issample:
                skeleton_anno_path = os.path.join(data_path, video_name, frame_name)      # 将路径和文件名连起来
                # 调用之前写好的函数，将skeleton_sequence字典赋给skeleton_sequence
                skeleton_sequence = read_skeleton_file(skeleton_anno_path)
                image_info = {
                    "license": 1,
                    "file_name": f'{video_name}/{frame_name[:-5]}.jpg', 
                    "coco_url": "",
                    "height": 480,
                    "width": 640,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": int(frame_name[1:4] + frame_name[5:8] + frame_name[9:12] + frame_name[13:16] + frame_name[18:21])
                }
                val_annotations['images'].append(image_info)

                annotation = {}
                annotation['id'] = id_
                annotation['image_id'] = int(frame_name[1:4] + frame_name[5:8] +
                                                frame_name[9:12] + frame_name[13:16] + frame_name[18:21])
                annotation['category_id'] = 1
                keypoints = []
                visible_keypoints = []
                for label in range(21):
                    if skeleton_sequence.get(f'{label:02d}') == None:
                        keypoint = [0, 0, 0]  # 0: "invisible", 1: "occlude", 2: "visible"
                    elif skeleton_sequence[f'{label:02d}'][0]==0 and skeleton_sequence[f'{label:02d}'][1]==0:
                        keypoint = [0, 0, 0]  # 0: "invisible", 1: "occlude", 2: "visible"
                    else:
                        keypoint = [skeleton_sequence[f'{label:02d}'][0],
                                    skeleton_sequence[f'{label:02d}'][1], 2]
                        visible_keypoints.append(keypoint)
                    keypoints.extend(keypoint)
                annotation['keypoints'] = keypoints
                annotation['num_keypoints'] = len(visible_keypoints)
                if not visible_keypoints:
                    continue
                # gen bbox and person_center
                visible_keypoints = np.array(visible_keypoints)
                x1, x2, y1, y2 = np.min(visible_keypoints[:, 0]), np.max(visible_keypoints[:, 0]), np.min(
                    visible_keypoints[:, 1]), np.max(visible_keypoints[:, 1])
                w, h = x2 - x1, y2 - y1
                annotation['bbox'] = [x1, y1, w, h]
                annotation['area'] = w * h
                annotation['iscrowd'] = 0
                annotation['segmentation'] = []
                val_annotations['annotations'].append(annotation)
                id_ += 1
    val_annotations["categories"] = [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": ["hip", "spine", "neck", "chin", "head", "left shoulder", "left elbow", "left wrist",
                          "left hand", "right shoulder", "right elbow", "right wrist ", "right hand", "left hip", 
                          "left knee", "left ankle", "left foot", "right hip", "right knee", "right ankle", "right foot", ],
            "skeleton": [
                (3, 10), (3, 6), (10, 11), (11, 12), (12, 13), (6, 7), (7, 8), (8, 9), (3, 2), (2, 1),
                (1, 18), (1, 14), (18, 19), (19, 20), (20, 21), (14, 15), (15, 16), (16, 17), (3, 4), (4, 5)]
        }
    ]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open('{}/val_label.json'.format(out_path), 'w') as f:
        json.dump(val_annotations, f, indent=4)


if __name__ == '__main__':
    skeleton_dir = '../datasets/scut_rgbd_subset/Label'
    out_dir = '../datasets/scut_rgbd_subset/2d_labels'
    gen_coco_style_json(skeleton_dir, out_dir)

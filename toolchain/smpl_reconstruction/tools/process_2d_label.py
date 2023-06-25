import argparse
import json
import os
import os.path as osp
import numpy as np


def read_skeleton_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
        frame_infos = js['shapes']

    body_joint = {}
    for frame_info in frame_infos:
        if 'group_id' not in frame_info.keys() or not frame_info['group_id']:
            person_id=0
        else:
            person_id=int(frame_info['group_id'])
        single_body_joint=body_joint.setdefault(person_id, {})
        single_body_joint[frame_info['label'].zfill(2)] = frame_info['points'][0]

    return body_joint  


def convert_body21_to_smpl(keypoints):
    smpl_idx=[0, 13, 17, -1, 14, 18, 1, 15, 19, -1, 16, 20, 2, -1, -1, 3, 5, 9, 6, 10, 7, 11, 8, 12]
    smpl_keypoints=[]
    for i in range(24):
        if smpl_idx[i]==-1:
            new_keypoint=[0, 0, 0]
        else:
            new_keypoint=keypoints[smpl_idx[i]]
        smpl_keypoints.append(new_keypoint)
    return smpl_keypoints

def convert_body21_to_body25(keypoints):
    smpl_idx=[3, 2, 9, 10, 11, 5, 6, 7, 0, 17, 18, 19, 13, 14, 15, 4, 4, -1, -1, 16, -1, -1, 20, -1, -1]
    smpl_keypoints=[]
    for i in range(25):
        if smpl_idx[i]==-1:
            new_keypoint=[0, 0, 0]
        else:
            new_keypoint=keypoints[smpl_idx[i]]
        smpl_keypoints.append(new_keypoint)
    return smpl_keypoints


def process_2d_labels(seq_root, skel_type):
    ori_label_dir = osp.join(seq_root, 'annots_ori')
    out_annots_dir = osp.join(seq_root, 'annots')

    cam_names = os.listdir(ori_label_dir)

    for cam_name in cam_names:
        cam_anno_dir = osp.join(ori_label_dir, cam_name)
        anno_files = os.listdir(cam_anno_dir)
        for anno_file in anno_files:
            anno_path = osp.join(cam_anno_dir, anno_file)
            anno = {
                'filename': f'images/{cam_name}/{anno_file.replace(".json", ".jpg")}',
                'height': 480,
                'width': 640,
            }
            anno['annots'] = []
            all_skeleton_sequence = read_skeleton_file(anno_path)
            for person_id in range(len(all_skeleton_sequence)):
                keypoints = []
                visible_keypoints = []
                skeleton_sequence=all_skeleton_sequence[person_id]
                for label in range(21):
                    if skeleton_sequence.get(f'{label:02d}') == None:
                        keypoint = [0, 0, 0]  # 0 for unvisible
                    elif skeleton_sequence[f'{label:02d}'][0] == 0 and skeleton_sequence[f'{label:02d}'][1] == 0:
                        keypoint = [0, 0, 0]
                    else:
                        keypoint = [skeleton_sequence[f'{label:02d}'][0],
                                    skeleton_sequence[f'{label:02d}'][1], 1]
                        visible_keypoints.append(keypoint)
                    keypoints.append(keypoint)

                if not visible_keypoints:
                    continue
                # gen bbox and person_center
                visible_keypoints = np.array(visible_keypoints)
                x1, x2, y1, y2 = np.min(visible_keypoints[:, 0]), np.max(visible_keypoints[:, 0]), np.min(
                    visible_keypoints[:, 1]), np.max(visible_keypoints[:, 1])

                # convert to smpl
                if skel_type=='smpl':
                    keypoints=convert_body21_to_smpl(keypoints)
                elif skel_type=='body25':
                    keypoints=convert_body21_to_body25(keypoints)
                elif skel_type=='body21':
                    pass
                else:
                    raise ValueError('The skel type is invalid, please check!')


                w, h = x2 - x1, y2 - y1
                area = w * h
                anno['annots'].append({
                    'personID': person_id,
                    'bbox': [x1, y1, x2, y2, 1],
                    'keypoints': keypoints,
                    'area': area
                })

            out_json_path = osp.join(out_annots_dir, cam_name, anno_file)
            if not osp.isdir(osp.dirname(out_json_path)):
                os.makedirs(osp.dirname(out_json_path))
            with open(out_json_path, 'w') as f:
                json.dump(anno, f, indent=4)



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', type=str, default='tmp/seq_mp', help='path to the root of the processed sequence')
    parser.add_argument('--skel_type', type=str, default='smpl', help='skeleton style to the output label, must be smpl, body21 or body25')
    args = parser.parse_args()
    process_2d_labels(args.seq_dir, args.skel_type)
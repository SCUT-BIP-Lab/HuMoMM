import os
import sys
import argparse
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm

import _init_paths
from inference_api import PoseEstimation
from utils.utils import img_arr_to_b64


def check_dir(dir):
    if os.path.exists(dir) and len(os.listdir(dir)):
        print(f'{dir}')
        ans = input('exists and is not empty, delete it and continue?(y/n):')
        if ans == 'y':
            shutil.rmtree(dir)
        else:
            raise Exception(f'{dir} exists and is not empty.')
    os.makedirs(dir, exist_ok=True)

    return


def main(args):
    if not args.redet_fail:
        check_dir(args.auto_label_path)

    if args.redet_fail:
        fail_samples, fewer_samples=[], []
        if os.path.exists('tmp/det_fail_imgs.txt'):
            with open('tmp/det_fail_imgs.txt', 'r') as f:
                fail_samples=f.readlines()
                fail_samples=[s.strip() for s in fail_samples]
        if os.path.exists('tmp/det_fewer_imgs.txt'):
            with open('tmp/det_fewer_imgs.txt', 'r') as f:
                fewer_samples=f.readlines()
                fewer_samples=[s.strip() for s in fewer_samples]
    
    if os.path.exists('tmp/det_fail_imgs.txt'):
        os.remove('tmp/det_fail_imgs.txt')
    if os.path.exists('tmp/det_fewer_imgs.txt'):
        os.remove('tmp/det_fewer_imgs.txt')

    auto_labelling_rgb = os.path.join(args.data_root, 'RGB_frame')
    pose_model = PoseEstimation(args.model_file)

    for sample in tqdm(sorted(os.listdir(auto_labelling_rgb))):
        if not os.path.exists(os.path.join(args.auto_label_path, sample)):
            os.mkdir(os.path.join(args.auto_label_path, sample))

        manual_labelling_json_list = os.listdir(os.path.join(args.manual_label_path, sample)) if os.path.isdir(
            os.path.join(args.manual_label_path, sample)) else []
        manual_labelling_img_list = [json_path.replace('json', 'jpg') for json_path in manual_labelling_json_list]
        for img_path in os.listdir(os.path.join(auto_labelling_rgb, sample)):
            if args.redet_fail:
                if img_path not in fail_samples and img_path not in fewer_samples:
                    continue
            if img_path in manual_labelling_img_list:
                shutil.copy2(os.path.join(args.manual_label_path, sample, img_path.replace('jpg', 'json')),
                             os.path.join(args.auto_label_path, sample, img_path.replace('jpg', 'json')))
            else:
                img = cv2.imread(os.path.join(auto_labelling_rgb, sample, img_path))
                js_data = dict()
                js_data['shapes'] = []
                js_data['imagePath'] = img_path
                js_data['imageData'] = img_arr_to_b64(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).decode('utf-8')

                pred_pose, has_person = pose_model.inference(img, img_path)

                if not has_person:
                    with open('tmp/det_fail_imgs.txt', 'a') as f:
                        f.write(f'{img_path}\n')

                num_person=len(pred_pose)
                if num_person<args.num_person:
                    print(f'{img_path}, fewer person were detected than expected!')
                    with open('tmp/det_fewer_imgs.txt', 'a') as f:
                        f.write(f'{img_path}\n')
                elif num_person>args.num_person:
                    pred_pose=pred_pose[:args.num_person]

                for person_id, single_pose in enumerate(pred_pose):
                    for joint_idx, joint_2d in enumerate(single_pose):
                        joint_data = {
                            'label': f'{joint_idx:02d}',
                            'points': [joint_2d.tolist()],
                            'vis': '1', 
                            'group_id': f'{person_id}'
                        }
                        js_data['shapes'].append(joint_data)
                js_path = os.path.join(args.auto_label_path, sample, img_path.replace('jpg', 'json'))
                with open(js_path, 'w') as dump_f:
                    json.dump(js_data, dump_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/custom_dataset',
                        help='path to the root of the origin dataset.')
    parser.add_argument('--manual_label_path', type=str,
                        default='../dataset/custom_dataset_key_frame/Label', help='path to the manual labelling dataset.')
    parser.add_argument('--auto_label_path', type=str, default='../dataset/custom_dataset/Label',
                        help='path to the automatic labelling file.')
    parser.add_argument('--num_person', type=int, default=1, help='number of person to detect')
    parser.add_argument('--model_file', type=str, default='./tmp/auto_labeling_models/model1.pth',
                        help='path to the automatic labelling file.')
    parser.add_argument('--redet_fail', action='store_true', help='re-detect the fail person in det_fail_imgs.txt and det_fewer_imgs.txt')
    args = parser.parse_args()
    main(args)

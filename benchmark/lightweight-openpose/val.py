import os
import argparse
import cv2
import time
import numpy as np

import torch
from datasets import get_val_dataset
from utils.inference import infer, infer_fast
from models import get_pose_estimation_model

from utils.keypoints import extract_keypoints, extract_keypoints_fast, group_keypoints
from utils.load_state import load_state
from utils.pose import get_pose
from utils.utils import AverageMeter


def evaluate(dataset, net, output_name, input_height=256, stride=8, upsample_ratio=4, multiscale=False, visualize=False):
    net = net.cuda().eval()
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]

    coco_result = []
    infer_net_time_avg = AverageMeter()
    extract_kpt_time_avg = AverageMeter()
    group_kpt_time_avg = AverageMeter()
    all_inference_time_avg = AverageMeter()

    for i, sample in enumerate(dataset):
        start_time = time.time()
        file_name = sample['file_name']
        img = sample['img']

        ###################################################################################
        avg_heatmaps, avg_pafs = infer(net, img, scales, input_height, stride)

        infer_time = time.time()
        # print(f'infer network time: {(infer_time - start_time) * 1000:.0f}ms')
        infer_net_time_avg.update((infer_time - start_time) * 1000)
        all_keypoints_by_type = extract_keypoints(avg_heatmaps)

        extract_time = time.time()
        # print(f'extract keypoints time: {(extract_time - infer_time) * 1000:.0f}ms')
        extract_kpt_time_avg.update((extract_time - infer_time) * 1000)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs, dataset=dataset.dataset_name)

        end_time = time.time()
        # print(f'group keypoints time: {(end_time - extract_time) * 1000:.0f}ms')
        group_kpt_time_avg.update((end_time - extract_time) * 1000)

        # print(f'all inference time: {(end_time - start_time) * 1000:.0f}ms')
        all_inference_time_avg.update((end_time - start_time) * 1000)
        #############################################################################################
        # fast infer
        # heatmaps, pafs, scale, pad = infer_fast(net, img, input_height, stride, upsample_ratio, cpu=False)

        # infer_time = time.time()
        # # print(f'infer network time: {(infer_time - start_time) * 1000:.0f}ms')
        # infer_net_time_avg.update((infer_time - start_time) * 1000)

        # all_keypoints_by_type = extract_keypoints_fast(heatmaps)

        # extract_time = time.time()
        # # print(f'extract keypoints time: {(extract_time - infer_time) * 1000:.0f}ms')
        # extract_kpt_time_avg.update((extract_time - infer_time) * 1000)

        # pose_entries, all_keypoints = group_keypoints(
        #     all_keypoints_by_type, pafs, dataset=dataset.dataset_name)
        # for kpt_id in range(all_keypoints.shape[0]):
        #     all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        #     all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        # end_time = time.time()
        # # print(f'group keypoints time: {(end_time - extract_time) * 1000:.0f}ms')
        # group_kpt_time_avg.update((end_time - extract_time) * 1000)

        # # print(f'all inference time: {(end_time - start_time) * 1000:.0f}ms')
        # all_inference_time_avg.update((end_time - start_time) * 1000)
        ##########################################################################################

        coco_keypoints, scores = dataset.convert_to_coco_format(pose_entries, all_keypoints)

        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': sample['image_id'],
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((dataset.num_joints, 2), dtype=np.int32) * -1
                for kpt_id in range(dataset.num_joints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = get_pose(dataset.dataset_name, keypoints=pose_keypoints,
                                confidence=pose_entries[n][dataset.num_joints])
                current_poses.append(pose)

            orig_img = img.copy()
            for pose in current_poses:
                pose.draw(img)
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            vis_name = f'tmp/val_vis/{file_name}'
            vis_dir = os.path.dirname(vis_name)
            if not os.path.isdir(vis_dir):
                os.makedirs(vis_dir)
            cv2.imwrite(vis_name, img)

    print(f'infer network time: {infer_net_time_avg.avg:.0f}ms, extract keypoints time: {extract_kpt_time_avg.avg:.0f}ms, '
          f'group keypoints time: {group_kpt_time_avg.avg:.0f}ms, all inference time: {all_inference_time_avg.avg:.0f}ms.')
    dataset.evaluate(coco_result, output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='./results/detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--input-height', type=int, default=256, help='input height of net')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    parser.add_argument('--use-pruned', action='store_true', help='show keypoints')
    parser.add_argument('--arch', type=str, default='mobilenetv2',
                        help='backbone for pose estimation model, must be mobilenetv2 or hrnet')
    parser.add_argument('--dataset', type=str, default='ntu', help='dataset to train, must be ntu or coco')
    args = parser.parse_args()

    val_dataset = get_val_dataset(args.dataset, labels=args.labels, images_folder=args.images_folder)

    net = get_pose_estimation_model(args.arch, num_heatmaps=val_dataset.num_joints + 1, num_pafs=val_dataset.num_pafs)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    # set some configs
    if args.arch == 'mobilenetv2':
        stride, upsample_ratio = 8, 4
    elif 'hrnet' in args.arch:
        stride, upsample_ratio = 4, 2
    else:
        raise ValueError('The arch is incorrect, must be hrnet or mobilenet')
    with torch.no_grad():
        evaluate(val_dataset, net, args.output_name, args.input_height, stride, upsample_ratio)

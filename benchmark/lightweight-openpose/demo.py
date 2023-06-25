import argparse
import time

import cv2
import numpy as np
import torch
from models import get_pose_estimation_model

from configs.keypoints import get_keypoint_info
from utils.keypoints import extract_keypoints, extract_keypoints_fast, group_keypoints
from utils.load_state import load_state
from utils.pose import get_pose, track_poses
from utils.inference import infer_fast
from utils.utils import vis_heatmap, vis_paf


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def run_demo(net, image_provider, dataset, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = get_keypoint_info(dataset).NUM_JOINTS
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        vis_heatmap(img, heatmaps.transpose((2, 0, 1)), 'tmp/heatmaps.png')
        vis_paf(img, pafs.transpose((2, 0, 1)), 'tmp/pafs.png')

        all_keypoints_by_type = extract_keypoints_fast(heatmaps)


        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs, dataset='scut')
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = get_pose(dataset,keypoints=pose_keypoints, confidence=pose_entries[n][22])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        # for pose in current_poses:
        #     cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
        #                   (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        #     if track:
        #         cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
        #                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imwrite(f'tmp/demo.jpg', img)
        # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        # key = cv2.waitKey(delay)
        # if key == 27:  # esc
        #     return
        # elif key == 112:  # 'p'
        #     if delay == 1:
        #         delay = 0
        #     else:
        #         delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--dataset', type=str, default='ntu', help='dataset to the demo model')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    keypoint_info=get_keypoint_info(args.dataset)
    net = get_pose_estimation_model('mobilenetv2', num_heatmaps=keypoint_info.NUM_JOINTS + 1, num_pafs=keypoint_info.NUM_PAFS)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.dataset, args.height_size, args.cpu, args.track, args.smooth)

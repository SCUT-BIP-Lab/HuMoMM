import os
import pickle
import cv2
import numpy as np
from utils.pose import NTUPose

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


def vis_nturgbd(skeleton_path, img_dir):
    filenames = sorted(os.listdir(skeleton_dir))
    for i, filename in enumerate(filenames):
        # use S001 for training
        if filename[:4] != 'S001':
            continue
        print(f'procressing video: {filename[:-9]}')

        skeleton_anno_path = os.path.join(skeleton_dir, filename)
        skeleton_sequence = read_skeleton_filter(skeleton_anno_path)
        for frame_i in range(skeleton_sequence['numFrame']):
            if frame_i % train_freq != 0:
                continue
            frame_info = skeleton_sequence['frameInfo'][frame_i]

            img_name = f'{filename[:-9]}_rgb/{frame_i:03d}.jpg'
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            orig_img = img.copy()
            for body_info in frame_info['bodyInfo']:
                keypoints = []
                for join_info in body_info['jointInfo']:
                    keypoint = [join_info['colorX'], join_info['colorY'], 1]  # assume the keypoints are visible
                    keypoints.append(keypoint)
                # gen bbox and person_center
                keypoints_array = np.array(keypoints)[:, 0:2]
                # vis
                pose = NTUPose(keypoints_array, 0)
                pose.draw(img)
                img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            out_path=f'tmp/vis_nturgbd/{img_name}'
            out_dir=os.path.dirname(out_path)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(out_path, img)
                


if __name__ == '__main__':
    skeleton_dir = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_skeletons'
    img_dir = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_rgb_imgs'
    vis_nturgbd(skeleton_dir, img_dir)

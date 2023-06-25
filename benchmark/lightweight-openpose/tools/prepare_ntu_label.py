import os
import pickle
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


def prepare_label(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='train', net_input_size=368):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    prepare_annotations = []
    filenames = sorted(os.listdir(data_path))
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

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            skeleton_anno_path = os.path.join(data_path, filename)
            skeleton_sequence = read_skeleton_filter(skeleton_anno_path)
            for frame_i in range(skeleton_sequence['numFrame']):
                if frame_i % train_freq != 0:
                    continue
                frame_info = skeleton_sequence['frameInfo'][frame_i]
                for body_info in frame_info['bodyInfo']:
                    annotation = {}
                    annotation['img_paths'] = f'{filename[:-9]}_rgb/{frame_i:03d}.jpg'
                    annotation['img_width'] = 1920
                    annotation['img_height'] = 1080
                    annotation['body_id'] = body_info['bodyID']
                    keypoints = []
                    visible_keypoints = []
                    for join_info in body_info['jointInfo']:
                        if np.isnan(join_info['colorX']) or np.isnan(join_info['colorY']):
                            keypoint = [0, 0, 2]
                        else:
                            keypoint = [join_info['colorX'], join_info['colorY'], 1]
                            visible_keypoints.append(keypoint)
                        keypoints.append(keypoint)
                    annotation['keypoints'] = keypoints
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
    with open('{}/{}_{}_label_256.pkl'.format(out_path, part, benchmark), 'wb') as f:
        pickle.dump(prepare_annotations, f)


if __name__ == '__main__':
    skeleton_dir = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_skeletons'
    out_dir = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_2d_labels'
    # prepare_label(skeleton_dir, out_dir, benchmark='xview', part='train')
    prepare_label(skeleton_dir, out_dir, benchmark='xsub', part='train', net_input_size=256)

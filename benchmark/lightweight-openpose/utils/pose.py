import cv2
import numpy as np

from configs.keypoints import get_keypoint_info
from utils.one_euro_filter import OneEuroFilter


def get_pose(name, **kwargs):
    pose_dict = {
        'coco': Pose,
        'ntu': NTUPose,
        'scut': SCUTPose
    }
    return pose_dict[name](**kwargs)


class Pose:
    num_kpts = get_keypoint_info('coco').NUM_JOINTS
    kpt_names = get_keypoint_info('coco').KEYPOINT_NAMES
    sigmas = np.array(get_keypoint_info('coco').SIGMAS, dtype=np.float32) / 10.0

    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]
    BODY_PARTS_KPT_IDS = get_keypoint_info('coco').BODY_PARTS_KPT_IDS
    BODY_PARTS_PAF_IDS = get_keypoint_info('coco').BODY_PARTS_PAF_IDS

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(Pose.BODY_PARTS_PAF_IDS)):
            kpt_a_id = Pose.BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = Pose.BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)


class NTUPose(Pose):
    num_kpts = get_keypoint_info('ntu').NUM_JOINTS
    kpt_names = get_keypoint_info('ntu').KEYPOINT_NAMES
    sigmas = np.array(get_keypoint_info('ntu').SIGMAS, dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]
    BODY_PARTS_KPT_IDS = get_keypoint_info('ntu').BODY_PARTS_KPT_IDS
    BODY_PARTS_PAF_IDS = get_keypoint_info('ntu').BODY_PARTS_PAF_IDS

    def __init__(self, keypoints, confidence):
        Pose.num_kpts = NTUPose.num_kpts
        Pose.kpt_names = NTUPose.kpt_names
        # TODO: need modify
        Pose.sigmas = NTUPose.sigmas
        Pose.vars = (NTUPose.sigmas * 2) ** 2
        Pose.last_id = NTUPose.last_id
        Pose.color = NTUPose.color
        Pose.BODY_PARTS_KPT_IDS = NTUPose.BODY_PARTS_KPT_IDS
        Pose.BODY_PARTS_PAF_IDS = NTUPose.BODY_PARTS_PAF_IDS
        super().__init__(keypoints, confidence)

class SCUTPose(Pose):
    num_kpts = get_keypoint_info('scut').NUM_JOINTS
    kpt_names = get_keypoint_info('scut').KEYPOINT_NAMES
    sigmas = np.array(get_keypoint_info('scut').SIGMAS, dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]
    BODY_PARTS_KPT_IDS = get_keypoint_info('scut').BODY_PARTS_KPT_IDS
    BODY_PARTS_PAF_IDS = get_keypoint_info('scut').BODY_PARTS_PAF_IDS

    def __init__(self, keypoints, confidence):
        Pose.num_kpts = SCUTPose.num_kpts
        Pose.kpt_names = SCUTPose.kpt_names
        # TODO: need modify
        Pose.sigmas = SCUTPose.sigmas
        Pose.vars = (SCUTPose.sigmas * 2) ** 2
        Pose.last_id = SCUTPose.last_id
        Pose.color = SCUTPose.color
        Pose.BODY_PARTS_KPT_IDS = SCUTPose.BODY_PARTS_KPT_IDS
        Pose.BODY_PARTS_PAF_IDS = SCUTPose.BODY_PARTS_PAF_IDS
        super().__init__(keypoints, confidence)

def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)

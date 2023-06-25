import operator

import cv2
import numpy as np


def thres_diff(np_all_frames, key_frame_number=15):
    '''Using difference with threshold to extract key frames.

    Args:
        np_all_frames: np.ndarray(N, H, W, 3) in BGR, original video frames.
        key_frame_number: int, number of extracted key frames.
    
    Returns:
        key_frames: np.ndarray(K, H, W, 3) in BGR, key frames.
        key_frame_idx: np.ndarray(K, ), indices of key frames in original video frames.
    '''
    USE_TOP_ORDER = True
    NUM_TOP_FRAMES = key_frame_number
    curr_frame = None
    prev_frame = None

    frame_diffs = []
    frames = []

    i = 0
    for filename in range(0, np_all_frames.shape[0]):
        curr_frame = np_all_frames[filename]
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1

    keyframe_id_set = set()

    if USE_TOP_ORDER:
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)

    idx = 0
    key_frame_idx = []
    key_frames = []

    for filename in range(0, np_all_frames.shape[0]):
        if idx in keyframe_id_set:
            key_f = np_all_frames[filename]
            keyframe_id_set.remove(idx)
            key_frames.append(key_f)
            key_frame_idx.append(idx)
        idx = idx + 1
    key_frames = np.stack(key_frames, axis=0)
    key_frame_idx = np.asarray(key_frame_idx)

    return key_frames, key_frame_idx    


def accu_diff(np_all_frames):
    '''Using accumulated difference to extract key frames.

    Args:
        np_all_frames: list of np.ndarray(N, H, W, 3) in BGR, list of original video frames.

    Returns:
        key_frames: np.ndarray(K, H, W, 3) in BGR, key frames.
        key_frame_idx: np.ndarray(K, ), indices of key frames in original video frames.
    '''
    USE_THRESH = True
    THRESH = 0.20

    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    i = 0
    for filename in range(0, np_all_frames.shape[0]):
        curr_frame = np_all_frames[filename]
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1

    keyframe_id_set = set()

    if USE_THRESH:
        sum1 = 0
        for i in range(1, len(frames)):
            a1 = (frames[i].diff - frames[i-1].diff) / max(frames[i-1].diff, frames[i].diff)
            a_2 = abs(a1)
            sum1 = sum1 + a_2
            if (sum1 >= THRESH):
                keyframe_id_set.add(frames[i].id)
                sum1 = 0

    # save all keyframes as image
    key_frame_idx = []
    key_frames = []
    idx = 0
    for filename in range(0, np_all_frames.shape[0]):
        if idx in keyframe_id_set:
            key_f = np_all_frames[filename]
            keyframe_id_set.remove(idx)
            key_frames.append(key_f)
            key_frame_idx.append(idx)
        idx = idx + 1
    key_frames = np.stack(key_frames, axis=0)
    key_frame_idx = np.asarray(key_frame_idx)

    return key_frames, key_frame_idx


class Frame(object):
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

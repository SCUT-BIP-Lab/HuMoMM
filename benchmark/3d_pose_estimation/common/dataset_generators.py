from itertools import zip_longest
import numpy as np
import torch

from torch.utils.data import Dataset


class ChunkedGeneratorDataset(Dataset):
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(
            poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] +
                        chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)),
                         bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)),
                             bounds[:-1], bounds[1:], ~augment_vector)

        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        if shuffle:
            self.random = np.random.RandomState(random_seed)
            self.pairs = self.random.permutation(self.pairs)

    def augment_enabled(self):
        return self.augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        chunk = self.pairs[index]
        seq_i, start_3d, end_3d, flip = chunk

        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        # 2D poses
        seq_2d = self.poses_2d[seq_i]
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            single_2d = np.pad(seq_2d[low_2d:high_2d], ((
                pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            single_2d = seq_2d[low_2d:high_2d].copy()

        if flip:
            # Flip 2D keypoints
            single_2d[:, :, 0] *= -1
            single_2d[:, self.kps_left + self.kps_right] = single_2d[
                :, self.kps_right + self.kps_left]
        # 3D poses
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_i]
            low_3d = max(start_3d, 0)
            high_3d = min(end_3d, seq_3d.shape[0])
            pad_left_3d = low_3d - start_3d
            pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                single_3d = np.pad(seq_3d[low_3d:high_3d], ((
                    pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                single_3d = seq_3d[low_3d:high_3d].copy()

            if flip:
                # Flip 3D joints
                single_3d[:, :, 0] *= -1
                single_3d[:, self.joints_left + self.joints_right] = \
                    single_3d[:,
                              self.joints_right + self.joints_left]
        # Cameras
        if self.cameras is not None:
            single_cam = self.cameras[seq_i].copy()
            if flip:
                # Flip horizontal distortion coefficients
                single_cam[2] *= -1
                single_cam[7] *= -1

        if self.cameras is not None and self.poses_3d is not None:
            return single_cam, single_3d, single_2d
        elif self.poses_3d is not None:
            return single_3d, single_2d
        return single_2d


class UnchunkedGeneratorDataset(Dataset):
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.

    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.

    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d

        self.augment = augment

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def __len__(self):
        return len(self.poses_2d)

    def __getitem__(self, index):
        single_cam = self.cameras[index].copy() if len(
            self.cameras) > 0 else None
        single_3d = self.poses_3d[index].copy() if len(
            self.poses_3d) > 0 else None
        seq_2d = self.poses_2d[index].copy()
        single_2d = np.pad(seq_2d,
                           ((self.pad + self.causal_shift, self.pad -
                             self.causal_shift), (0, 0), (0, 0)),
                           'edge')
        if self.augment:
            # Append flipped version
            if single_cam is not None:
                single_cam = np.expand_dims(single_cam, axis=0)
                single_cam = np.concatenate((single_cam, single_cam), axis=0)
                single_cam[1, 2] *= -1
                single_cam[1, 7] *= -1

            if single_3d is not None:
                single_3d = np.expand_dims(single_3d, axis=0)
                single_3d = np.concatenate((single_3d, single_3d), axis=0)
                single_3d[1, :, :, 0] *= -1
                single_3d[1, :, self.joints_left + self.joints_right] = single_3d[1,
                                                                                  :, self.joints_right + self.joints_left]

            single_2d = np.expand_dims(single_2d, axis=0)
            single_2d = np.concatenate((single_2d, single_2d), axis=0)
            single_2d[1, :, :, 0] *= -1
            single_2d[1, :, self.kps_left + self.kps_right] = single_2d[1,
                                                                        :, self.kps_right + self.kps_left]

        if single_cam is not None and single_3d is not None:
            return single_cam, single_3d, single_2d
        elif single_3d is not None:
            return single_3d, single_2d
        return single_2d

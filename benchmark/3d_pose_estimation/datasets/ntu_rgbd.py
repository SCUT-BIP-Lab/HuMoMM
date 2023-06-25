import numpy as np
import copy
from common.skeleton import Skeleton
from .mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates

ntu_skeleton = Skeleton(parents=[-1,  0,  20,  2,  20,  4,  5,  6,  20,  8,  9,  10, 0, 12, 13, 14, 0,
                                 16, 17, 18, 1, 7, 7, 11, 11],
                        joints_left=[4, 5, 6, 7, 12, 13, 14, 15, 21, 22],
                        joints_right=[8, 9, 10, 11, 16, 17, 18, 19, 23, 24])

# Joints in H3.6M -- data has 32 joints, but only 17 retained; these are the indices.
NTU_NAMES = [''] * 25
NTU_NAMES[0] = 'Hip'
NTU_NAMES[16] = 'RHip'
NTU_NAMES[17] = 'RKnee'
NTU_NAMES[18] = 'RAnkle'
NTU_NAMES[12] = 'LHip'
NTU_NAMES[13] = 'LKnee'
NTU_NAMES[14] = 'LAnkle'
NTU_NAMES[1] = 'Spine'
NTU_NAMES[20] = 'Thorax'  # TODO 不一定正确
NTU_NAMES[2] = 'Neck/Nose'
NTU_NAMES[3] = 'Head'
NTU_NAMES[4] = 'LShoulder'
NTU_NAMES[5] = 'LElbow'
NTU_NAMES[6] = 'LWrist'
NTU_NAMES[8] = 'RShoulder'
NTU_NAMES[9] = 'RElbow'
NTU_NAMES[10] = 'RWrist'

valid_joints = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 4, 5, 6, 8, 9, 10]

ntu_remove_list = [7, 11, 15, 19, 21, 22, 23, 24]


class NTU_RGBD(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        super().__init__(fps=50, skeleton=copy.deepcopy(ntu_skeleton))
        print('Preparing NTU-RGBD Dataset...')

        # Load serialized dataset
        self._data = np.load(path, allow_pickle=True)['positions_3d'].item()

        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 25
            self.remove_joints_better(valid_joints=valid_joints)

    def supports_semi_supervised(self):
        return True

    def valid_indexes(self):
        return self.valid_joints

    def remove_joints_better(self, valid_joints):
        valid_indexes = self._skeleton.remove_joints_better(valid_joints)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                for cam in self._data[subject][action].keys():
                    for seg in self._data[subject][action][cam].keys():
                        s = self._data[subject][action][cam][seg]
                        self._data[subject][action][cam][seg] = s[:, valid_indexes]
        self.valid_joints = valid_indexes

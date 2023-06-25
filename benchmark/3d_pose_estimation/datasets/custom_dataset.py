import numpy as np
import copy
from common.skeleton import Skeleton
from .mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates

custom_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                           joints_left=[5, 6, 7, 8, 13, 14, 15, 16],
                           joints_right=[9, 10, 11, 12, 17, 18, 19, 20])

CUSTOM_NAMES = [''] * 21
CUSTOM_NAMES[0] = 'Hip'
CUSTOM_NAMES[17] = 'RHip'
CUSTOM_NAMES[18] = 'RKnee'
CUSTOM_NAMES[19] = 'RAnkle'
CUSTOM_NAMES[13] = 'LHip'
CUSTOM_NAMES[14] = 'LKnee'
CUSTOM_NAMES[15] = 'LAnkle'
CUSTOM_NAMES[1] = 'Spine'
CUSTOM_NAMES[2] = 'Thorax'  # TODO 不一定正确
CUSTOM_NAMES[3] = 'Neck/Nose'
CUSTOM_NAMES[4] = 'Head'
CUSTOM_NAMES[5] = 'LShoulder'
CUSTOM_NAMES[6] = 'LElbow'
CUSTOM_NAMES[7] = 'LWrist'
CUSTOM_NAMES[9] = 'RShoulder'
CUSTOM_NAMES[10] = 'RElbow'
CUSTOM_NAMES[11] = 'RWrist'

valid_joints = [0, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

custom_remove_list = [8, 12, 16, 20]


class Custom_dataset(MocapDataset):
    def __init__(self, path, remove_static_joints=False):
        super().__init__(fps=50, skeleton=copy.deepcopy(custom_skeleton))
        print('Preparing Custom Dataset...')

        # Load serialized dataset
        self._data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self.valid_joints = [i for i in range(21)]

        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 25
            self.remove_joints_better(valid_joints=valid_joints)

    def supports_semi_supervised(self):
        return True

    def valid_indexes(self):
        return self.valid_joints

    def remove_joints_better(self, valid_joints):
        valid_indexes = self._skeleton.remove_joints_better(valid_joints)
        self._data = self._data[:, :, valid_indexes]
        self.valid_joints = valid_indexes

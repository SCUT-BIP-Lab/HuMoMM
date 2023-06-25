# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from posixpath import join
import numpy as np


class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints_better(self, valid_joints):
        idx_map = {}
        new_parents = []
        new_joints_left = []
        new_joints_right = []
        for new_idx, old_idx in enumerate(valid_joints):
            idx_map[old_idx] = new_idx

        for joint in valid_joints:
            while self._parents[joint] != -1 and self._parents[joint] not in valid_joints:
                self._parents[joint] = self._parents[self._parents[joint]]
            parent = self._parents[joint]
            if parent == -1:
                new_parents.append(parent)
            else:
                new_parents.append(idx_map[parent])

            if self._joints_left is not None and joint in self._joints_left:
                new_joints_left.append(idx_map[joint])

            if self._joints_right is not None and joint in self._joints_right:
                new_joints_right.append(idx_map[joint])

        self._parents = new_parents
        if self._joints_left is not None:
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            self._joints_right = new_joints_right

        self._compute_metadata()
        return valid_joints

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        # if the parent of a joint is to_remove, change the parent to parent of parent.
        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

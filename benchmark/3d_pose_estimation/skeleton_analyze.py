from common.skeleton import Skeleton
from datasets.h36m import h36m_skeleton, H36M_NAMES, h36m_remove_list
from datasets.ntu_rgbd import ntu_skeleton, NTU_NAMES, ntu_remove_list, valid_joints


def skeleton_analyze(skeleton_names, skeleton, remove_list=None, valid_list=None):

    if remove_list is not None:
        joint_parents = []
        joint_index = {}
        for i, joint in enumerate(skeleton_names):
            if joint == '':
                continue
            joint_index[joint] = i
            parent = skeleton._parents[i]
            parent_name = ''
            if parent < 0:
                parent_name = 'None'
            else:
                parent_name = skeleton_names[parent]
            joint_parents.append((joint, parent_name))

        skeleton.remove_joints(
            remove_list)

        for i, p in enumerate(skeleton._parents):
            joint, origin_parent = joint_parents[i]
            if p == -1:
                parent = 'None'
            else:
                parent, _ = joint_parents[p]
            print("{}-{}-{}-{}({})-{}".format(
                i, joint_index[joint], joint, parent, p, origin_parent))
    else:
        parents_dict = {}
        for old_idx, name in enumerate(skeleton_names):
            parent = skeleton._parents[old_idx]
            if parent == -1:
                parents_dict[name] = 'None'
            else:
                parents_dict[name] = skeleton_names[parent]
        skeleton.remove_joints_better(valid_list)
        for i, p in enumerate(skeleton._parents):
            old_idx = valid_joints[i]
            origin_parent = parents_dict[skeleton_names[old_idx]]
            joint = skeleton_names[old_idx]
            parent = skeleton_names[valid_joints[p]]
            print("{}-{}-{}-{}({})-{}".format(
                i, old_idx, joint, parent, p, origin_parent))


skeleton_analyze(H36M_NAMES, h36m_skeleton, remove_list=h36m_remove_list)
print("==============")
skeleton_analyze(NTU_NAMES, ntu_skeleton, valid_list=valid_joints)

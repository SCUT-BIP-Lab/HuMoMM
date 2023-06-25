#!/usr/bin/env python
# coding=utf-8

'''
transform the skeleton data in NTU RGB+D dataset into the numpy arrays for a more efficient data loading
'''

from posixpath import join
import numpy as np
import os
from glob import glob
import sys
from tqdm import tqdm
import cv2

save_npz_path = '/home/wt/py_projects/HPE-3d/data/'
output_filename = save_npz_path + 'data_3d_ntu'
output_filename_2d = save_npz_path + 'data_2d_ntu_gt'
output_filename_dep = save_npz_path + 'data_dep_ntu'
skeletons_path = '/home/data/hpe_3d/ntu-rgbd/nturgb+d_skeletons/'
missing_file_path = '/home/data/hpe_3d/ntu-rgbd/NTU_RGBD_samples_with_missing_skeletons.txt'
mask_path = '/home/data/hpe_3d/ntu-rgbd/nturgb+d_depth_masked/'
# just parse range, for the purpose of paralle running.
# step_ranges = list(range(0, 100))

target_subjects = ['S001', 'S002', 'S003']
target_cameras = ['C001', 'C002', 'C003']
target_actions = ['A%03d' % i for i in range(1, 6)]
max_body_num = 1
frames_thresh = 25


def get_skeletons(subjects, cameras, actions, miss_files):
    # e.g S001C003P008R002A058.skeleton
    total_skeletons = {}
    num = 0
    for subject in subjects:
        total_skeletons[subject] = {}
        for action in actions:
            total_skeletons[subject][action] = {}
            for cam in cameras:
                match = subject + cam + '*' + action + '.skeleton'
                print("finding %s ..." % match)
                files = glob(skeletons_path + match)
                print("finding missing...")
                for f in files:
                    if f in miss_files:
                        print("remove {}".format(f))
                        files.remove(f)
                files.sort()
                total_skeletons[subject][action][cam] = files
                num += len(files)
    return total_skeletons, num


def load_missing_file(skt_dir, path):
    missing_files = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files.append(skt_dir + line + '.skeleton')
    return missing_files


def read_skeleton(file_path):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    bodymat = dict()
    nframe = int(datas[0][:-1])
    bodymat['nframe'] = nframe
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = []
    bodymat['origin_frame'] = []
    bodymat['njoints'] = njoints
    bodymat['valid'] = True
    for body in range(max_body):
        bodymat['skel_body{}'.format(body)] = np.zeros(
            (nframe, njoints, 3), dtype=np.float32)
        bodymat['rgb_body{}'.format(body)] = np.zeros(
            (nframe, njoints, 2), dtype=np.float32)
        bodymat['depth_body{}'.format(body)] = np.zeros(
            (nframe, njoints, 2), dtype=np.float32)
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount == 0:
            # skip the empty frame
            continue
        # TODO 目前只保留那些只含有一个人的帧
        if bodycount == 1:
            bodymat['nbodys'].append(bodycount)
            bodymat['origin_frame'].append(frame)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)

            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                bodymat[skel_body][frame, joint] = jointinfo[:3]
                bodymat[depth_body][frame, joint] = jointinfo[3:5]
                bodymat[rgb_body][frame, joint] = jointinfo[5:7]

    if len(bodymat['nbodys']) < frames_thresh:
        bodymat['valid'] = False
        return bodymat

    # prune the abundant bodys
    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            del bodymat['skel_body{}'.format(each)]
            del bodymat['rgb_body{}'.format(each)]
            del bodymat['depth_body{}'.format(each)]
    return bodymat


def run():
    missing_files = load_missing_file(skeletons_path, missing_file_path)
    skeletons, num_skeletons = get_skeletons(target_subjects, target_cameras,
                                             target_actions, missing_files)
    print("get %d skeletons." % num_skeletons)

    output_3d = {}
    output_2d = {}
    output_dep = {}

    wrong_depth_file = []

    for subject in skeletons.keys():
        output_3d[subject] = {}
        output_2d[subject] = {}
        output_dep[subject] = {}

        for action in skeletons[subject].keys():
            output_3d[subject][action] = {}
            output_2d[subject][action] = {}
            output_dep[subject][action] = {}

            for cam in skeletons[subject][action].keys():
                output_3d[subject][action][cam] = {}
                output_2d[subject][action][cam] = {}
                output_dep[subject][action][cam] = {}

                files = skeletons[subject][action][cam]
                seg = 0
                print("process {}-{}-{}".format(subject, action, cam))
                for f in tqdm(files):
                    basename = os.path.basename(f)
                    name = basename.split('.')[-2]

                    bodymat = read_skeleton(f)
                    valid = bodymat['valid']
                    if not valid:
                        print("find invalid file: {}".format(name))
                        continue

                    nbodys = bodymat['nbodys']
                    origin_frames = bodymat['origin_frame']
                    njoints = bodymat['njoints']

                    body_num = min(max_body_num, max(nbodys))

                    assert(len(nbodys) == len(origin_frames))

                    nframe = len(nbodys)

                    depth_vecs = [np.zeros((nframe, njoints, 1), dtype=np.float32)]
                    joints_vecs = [np.zeros((nframe,njoints,3),dtype=np.float32)]
                    keypoint_vecs = [np.zeros((nframe,njoints,2),dtype=np.float32)]
                    depth_vecs *= body_num
                    joints_vecs *= body_num
                    keypoint_vecs *= body_num
                    # read depth map
                    for frame, origin_frame in enumerate(origin_frames):
                        depth_path = os.path.join(
                            mask_path, name, 'MDepth-%08d.png' % (origin_frame + 1))
                        if not os.path.exists(depth_path):
                            print("ERROR: can not find {}".format(depth_path))
                            exit(-1)
                        depth_img = cv2.imread(depth_path)
                        for body in range(min(body_num, nbodys[frame])):
                            skel_body = 'skel_body{}'.format(body)
                            rgb_body = 'rgb_body{}'.format(body)
                            depth_body = 'depth_body{}'.format(body)

                            joints_vecs[body][frame] = bodymat[skel_body][origin_frame]
                            keypoint_vecs[body][frame] = bodymat[rgb_body][origin_frame]
                            for joint in range(njoints):
                                depth_x, depth_y = int(bodymat[depth_body][origin_frame, joint][0]), int(
                                    bodymat[depth_body][origin_frame, joint][1])
                                depth_x = max(0, depth_x)
                                depth_x = min(512 - 1, depth_x)
                                depth_y = max(0, depth_y)
                                depth_y = min(424 - 1, depth_y)
                                # 注意x,y是像素坐标
                                depth = depth_img[depth_y][depth_x][0]
                                if np.isnan(depth) or depth == 0:
                                    wrong_file_name = os.path.join(
                                        name, 'MDepth-%08d.png' % (origin_frame + 1))
                                    xmin = max(0, depth_y - 2)
                                    xmax = min(424 - 1, depth_y + 3)
                                    ymin = max(0, depth_x - 2)
                                    ymax = min(512 - 1, depth_x + 3)
                                    sub_depth = np.mean(
                                        depth_img[xmin:xmax, ymin:ymax, 0])
                                    print("find wrong depth: {} body: {} joint: {} depth: {} relace by: {}".format(
                                        wrong_file_name, body, joint, depth, sub_depth))
                                    wrong_depth_file.append(wrong_file_name)
                                    depth = sub_depth
                                    # exit(-1)
                                depth_vecs[body][frame, joint] = depth
                            # 将为0的深度值用平均值代替
                            m = np.mean(depth_vecs[body][depth_vecs[body] > 0])
                            depth_vecs[body][depth_vecs[body] == 0] = m
                    for body in range(body_num):
                        output_3d[subject][action][cam][seg] = joints_vecs[body].astype(
                            'float32')
                        output_2d[subject][action][cam][seg] = keypoint_vecs[body].astype(
                            'float32')
                        output_dep[subject][action][cam][seg] = depth_vecs[body].astype(
                            'float32')
                        seg += 1
    print("Finish reading all files.\nSaving...")
    np.savez_compressed(output_filename, positions_3d=output_3d)
    np.savez_compressed(output_filename_2d, positions_2d=output_2d)
    np.savez_compressed(output_filename_dep, depths=output_dep)
    print("Done.")


if __name__ == '__main__':
    run()
    

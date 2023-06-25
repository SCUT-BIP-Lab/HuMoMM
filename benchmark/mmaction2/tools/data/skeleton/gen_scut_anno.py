# -- coding: utf-8 --**

import json
import pickle
import numpy as np
import os
import os.path as osp


def scut_anno_generation(ori_label_dirs, out_train_path, out_val_path, benchmark):
    ori_label_list = os.listdir(ori_label_dirs)
    scut_train = []
    scut_val = []
    ori_label_list = sorted(ori_label_list)
    for i, video in enumerate(ori_label_list):
        print(i,'/', len(ori_label_list))
        anno = dict()
        videoframe_dir = osp.join(ori_label_dirs,video)
        ori_anno_list = os.listdir(videoframe_dir)
        total_frames = len(ori_anno_list)
        keypoint = []
        keypoint_score = []
        person_id = 0
        ori_anno_list = sorted(ori_anno_list)
        for j, frame in enumerate(ori_anno_list):
            kpts = [[0.,0.] for x in range(0,21)]
            vis = [0 for x in range(0,21)]
            videoframe_path = osp.join(videoframe_dir,frame)
            with open(videoframe_path) as f:
                load_data = json.load(f)
                keypoints = load_data['shapes']
                 
                label = videoframe_dir.split('A')[1][:3]
                person_id = int(load_data['imagePath'].split('P')[1][:3])
                view_id = int(load_data['imagePath'].split('C')[1][:3])
                frame_dir = videoframe_dir
                for k,kpt in enumerate(keypoints):
                    idx = int(kpt['label'])
                    kpts[idx] = (kpt['points'][0])
                    vis[idx] = 1
                # print(frame,len(kpts))
                keypoint.append(kpts)
                keypoint_score.append(vis)
                
        anno['keypoint'] = np.expand_dims(np.array(keypoint,dtype=np.float16),axis=0)
        anno['keypoint_score'] = np.expand_dims(np.array(keypoint_score,dtype=np.float16),axis=0)
        anno['frame_dir'] = video
        anno['img_shape'] = (480, 640)
        anno['original_shape'] = (480, 640)
        anno['total_frames'] = total_frames
        anno['label'] = int(label)
        # print(anno)
        if benchmark=='xsub':
            if person_id <= 9:
                scut_train.append(anno)
            else:
                scut_val.append(anno)
        if benchmark=='xview':
            if view_id<=2:
                scut_train.append(anno)
            else:
                scut_val.append(anno)
        # if i == 100:
        #     break
    out_train = open(out_train_path, 'wb')
    pickle.dump(scut_train, out_train)
    out_train.close()
    val_train = open(out_val_path, 'wb')
    pickle.dump(scut_val, val_train)
    val_train.close()

    return anno

if __name__ == '__main__':
    benchmark='xview'
    ori_label_dirs = '/data/pose_datasets/scut_sp/Label'
    out_train_path = '/data/pose_datasets/scut_sp/scut_sp_train_xview.pkl'
    out_val_path = '/data/pose_datasets/scut_sp/scut_sp_val_xview.pkl'
    scut_anno_generation(ori_label_dirs, out_train_path, out_val_path, benchmark)
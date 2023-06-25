'''
  @ Date: 2021-01-13 17:15:46
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-05 19:29:33
  @ FilePath: /EasyMocap/easymocap/dataset/mvmpmf.py
'''
import os.path as osp
import numpy as np
from .base import MVBase
from ..mytools.file_utils import get_bbox_from_pose
from easymocap.smplmodel.body_param import select_nf
from easymocap.mytools.reconstruction import projectN3

class MVMPMF(MVBase):
    """ Dataset for multi-view, multiperson, multiframe.
    This class is compatible with single-view, multiperson, multiframe if use specify only one `cams`
    """
    def __init__(self, root, cams=[], out=None, config={}, 
        image_root='images', annot_root='annots', kpts_type='body25',
        undis=False, no_img=False, filter2d=None) -> None:
        super().__init__(root, cams, out, config, image_root, annot_root, 
            kpts_type=kpts_type, undis=undis, no_img=no_img, filter2d=filter2d)
    
    def write_keypoints3d(self, peopleDict, nf):
        results = []
        for pid, people in peopleDict.items():
            result = {'id': pid, 'keypoints3d': people.keypoints3d}
            results.append(result)
        super().write_keypoints3d(results, nf)
    
    def vis_smpl(self, body_model, images, results, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True):

        render_data = {}
        for pid, result in results.items():
            frames = result['frames']
            if nf in frames:
                nnf = frames.index(nf)
                params = select_nf(result['body_params'], nnf)
                vertices = body_model(return_verts=True, return_tensor=False, **params)[0]
                assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
                render_data[pid] = {'vertices': vertices, 'faces': body_model.faces, 
                    'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        
        outname = osp.join(self.out, 'smpl', '{:06d}.jpg'.format(nf))
        cameras = {'K': [], 'R':[], 'T':[]}
        if len(sub_vis) == 0:
            sub_vis = self.cams
        for key in cameras.keys():
            cameras[key] = np.stack([self.cameras[cam][key] for cam in sub_vis])
        images = [images[self.cams.index(cam)] for cam in sub_vis]
        self.writer.vis_smpl(render_data, images, cameras, outname, add_back=add_back)

    def vis_repro(self, images, peopleDict, nf, sub_vis=[]):
        lDetections = []
        for nv in range(len(images)):
            res = []
            for pid, people in peopleDict.items():
                det = {
                    'id': people.id,
                    'keypoints2d': people.kptsRepro[nv],
                    'bbox': get_bbox_from_pose(people.kptsRepro[nv], images[nv])
                }
                res.append(det)
            lDetections.append(res)
        super().vis_detections(images, lDetections, nf, mode='repro', sub_vis=sub_vis)

    def vis_track_repro(self, images, results, nf, sub_vis=[]):
        people_dict={}
        for pid, result in results.items():
            frames = result['frames']
            if nf in frames:
                nnf = frames.index(nf)
                keypoints3d = result['keypoints3d'][nnf,:, :3]
                kptsRepro = projectN3(keypoints3d, self.Pall)
                people_dict[pid]={
                    'id': pid, 
                    'keypoints3d': keypoints3d,
                    'kptsRepro': kptsRepro
                }
        lDetections = []
        for nv in range(len(images)):
            res = []
            for pid, people in people_dict.items():
                det = {
                    'id': people['id'],
                    'keypoints2d': people['kptsRepro'][nv],
                    'bbox': get_bbox_from_pose(people['kptsRepro'][nv], images[nv])
                }
                res.append(det)
            lDetections.append(res)
        super().vis_detections(images, lDetections, nf, mode='repro', sub_vis=sub_vis)


    def save_track_repro(self, images, results, nf):
        people_dict={}
        for pid, result in results.items():
            frames = result['frames']
            if nf in frames:
                nnf = frames.index(nf)
                keypoints3d = result['keypoints3d'][nnf,:, :3]
                kptsRepro = projectN3(keypoints3d, self.Pall)
                people_dict[pid]={
                    'id': pid, 
                    'keypoints3d': keypoints3d,
                    'kptsRepro': kptsRepro
                }
        lDetections = []
        for nv in range(len(images)):
            res = []
            for pid, people in people_dict.items():
                det = {
                    'id': people['id'],
                    'keypoints': people['kptsRepro'][nv],
                    'bbox': get_bbox_from_pose(people['kptsRepro'][nv], images[nv])
                }
                res.append(det)
            lDetections.append(res)
        
        super().write_keypoints2d(lDetections, nf)
        

    def __getitem__(self, index: int):
        images, annots_all = super().__getitem__(index)
        # 筛除不需要的2d
        return images, annots_all

    def __len__(self) -> int:
        return self.nFrames
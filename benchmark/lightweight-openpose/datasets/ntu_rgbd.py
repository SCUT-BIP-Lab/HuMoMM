import copy
import json
import math
import os
import pickle

import cv2
import numpy as np
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from datasets.transformations import Scale, Rotate, CropPad, Flip
from .coco import CocoTrainDataset, CocoValDataset

from configs.keypoints import get_keypoint_info


def get_mask(segmentations, mask):
    for segmentation in segmentations:
        rle = pycocotools.mask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        mask[pycocotools.mask.decode(rle) > 0.5] = 0
    return mask


class NTUTrainDataset(CocoTrainDataset):
    def __init__(self, labels, images_folder, stride, sigma, paf_thickness, input_size):
        super().__init__(labels, images_folder, stride, sigma, paf_thickness, input_size)
        self.dataset_name = 'ntu'
        keypoint_info = get_keypoint_info('ntu')
        self.num_joints = keypoint_info.NUM_JOINTS
        self.num_pafs = keypoint_info.NUM_PAFS
        self.left = keypoint_info.LEFT
        self.right = keypoint_info.RIGHT
        self.BODY_PARTS_KPT_IDS = keypoint_info.BODY_PARTS_KPT_IDS

        self._transform = transforms.Compose([
            Scale(),
            Rotate(pad=(128, 128, 128)),
            CropPad(crop_x=input_size, crop_y=input_size, pad=(128, 128, 128)),
            Flip(self.left, self.right)])

        # self.add_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        label = copy.deepcopy(self._labels[idx])  # label modified in transform
        image = cv2.imread(os.path.join(self._images_folder, label['img_paths']), cv2.IMREAD_COLOR)
        mask = np.ones(shape=(label['img_height'], label['img_width']), dtype=np.float32)
        # mask = get_mask(label['segmentations'], mask)
        sample = {
            'label': label,
            'image': image,
            'mask': mask
        }
        if self._transform:
            sample = self._transform(sample)

        mask = cv2.resize(sample['mask'], dsize=None, fx=1 / self._stride,
                          fy=1 / self._stride, interpolation=cv2.INTER_AREA)
        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps
        keypoint_mask = np.zeros(shape=keypoint_maps.shape, dtype=np.float32)
        for idx in range(keypoint_mask.shape[0]):
            keypoint_mask[idx] = mask
        sample['keypoint_mask'] = keypoint_mask

        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = paf_maps
        paf_mask = np.zeros(shape=paf_maps.shape, dtype=np.float32)
        for idx in range(paf_mask.shape[0]):
            paf_mask[idx] = mask
        sample['paf_mask'] = paf_mask

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256
        sample['image'] = image.transpose((2, 0, 1))

        # sample['image'] = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
        # sample['image'] = self.add_transform(sample['image'])
        del sample['label']
        return sample

    def _generate_keypoint_maps(self, sample):
        n_keypoints = self.num_joints
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(n_keypoints + 1,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        label = sample['label']
        for keypoint_idx in range(n_keypoints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps

    def _generate_paf_maps(self, sample):
        n_pafs = len(self.BODY_PARTS_KPT_IDS)
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)

        label = sample['label']
        for paf_idx in range(n_pafs):
            keypoint_a = label['keypoints'][self.BODY_PARTS_KPT_IDS[paf_idx][0]]
            keypoint_b = label['keypoints'][self.BODY_PARTS_KPT_IDS[paf_idx][1]]
            if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                              keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                              self._stride, self._paf_thickness)
        return paf_maps


class NTUValDataset(CocoValDataset):
    def __init__(self, labels, images_folder):
        super().__init__(labels, images_folder)
        self.dataset_name = 'ntu'
        keypoint_info = get_keypoint_info('ntu')
        self.num_joints = keypoint_info.NUM_JOINTS
        self.num_pafs = keypoint_info.NUM_PAFS
        self.left = keypoint_info.LEFT
        self.right = keypoint_info.RIGHT

    def __getitem__(self, idx):
        file_name = self._labels['images'][idx]['file_name']
        img_id = self._labels['images'][idx]['id']
        img = cv2.imread(os.path.join(self._images_folder, file_name), cv2.IMREAD_COLOR)

        return {
            'img': img,
            'file_name': file_name,
            'image_id': img_id
        }

    def evaluate(self, pred_results, pred_results_path):
        result_dir = os.path.dirname(pred_results_path)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        with open(pred_results_path, 'w') as f:
            json.dump(pred_results, f, indent=4)

        self.run_coco_eval(self.label_path, pred_results_path)

    def run_coco_eval(self, gt_file_path, dt_file_path):
        annotation_type = 'keypoints'
        print('Running test for {} results.'.format(annotation_type))

        coco_gt = COCO(gt_file_path)
        coco_dt = coco_gt.loadRes(dt_file_path)

        result = NTUEval(coco_gt, coco_dt, annotation_type)
        result.evaluate()
        result.accumulate()
        result.summarize()

    def convert_to_coco_format(self, pose_entries, all_keypoints):
        coco_keypoints = []
        scores = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            keypoints = [0] * self.num_joints * 3
            person_score = pose_entries[n][-2]
            position_id = -1
            for keypoint_id in pose_entries[n][:-2]:
                position_id += 1

                cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                    cx = cx + 0.5
                    cy = cy + 0.5
                    visibility = 1
                keypoints[position_id * 3 + 0] = cx
                keypoints[position_id * 3 + 1] = cy
                keypoints[position_id * 3 + 2] = visibility
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
        return coco_keypoints, scores


class NTUEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array(get_keypoint_info('ntu').SIGMAS) / 10.0
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from datasets.transformations import Scale, Rotate, CropPad, Flip
from .ntu_rgbd import NTUTrainDataset, NTUValDataset
from configs.keypoints import get_keypoint_info


class SCUTTrainDataset(NTUTrainDataset):
    def __init__(self, labels, images_folder, stride, sigma, paf_thickness, input_size):
        super().__init__(labels, images_folder, stride, sigma, paf_thickness, input_size)
        self.dataset_name = 'scut'
        keypoint_info = get_keypoint_info('scut')
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


class SCUTValDataset(NTUValDataset):
    def __init__(self, labels, images_folder):
        super().__init__(labels, images_folder)
        self.dataset_name = 'scut'
        keypoint_info = get_keypoint_info('scut')
        self.num_joints = keypoint_info.NUM_JOINTS
        self.num_pafs = keypoint_info.NUM_PAFS
        self.left = keypoint_info.LEFT
        self.right = keypoint_info.RIGHT

    def run_coco_eval(self, gt_file_path, dt_file_path):
        annotation_type = 'keypoints'
        print('Running test for {} results.'.format(annotation_type))

        coco_gt = COCO(gt_file_path)
        coco_dt = coco_gt.loadRes(dt_file_path)

        result = SCUTEval(coco_gt, coco_dt, annotation_type)
        result.evaluate()
        result.accumulate()
        result.summarize()

class SCUTEval(COCOeval):
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
        sigmas = np.array(get_keypoint_info('scut').SIGMAS) / 10.0
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
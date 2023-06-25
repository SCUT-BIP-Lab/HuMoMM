import logging
from typing import Iterator
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import numpy as np
import torch

from common.camera import normalize_screen_coordinates
from common.loss import mpjpe
from trainval import create_model

config_path = '/home/wt/py_projects/Human-Pose-Estimation-3D/outputs/2022-01-08/20-44-58/.hydra/'
output_filename = 'predict'

log = logging.getLogger('PREDICTION')

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def load_dataset_raw(data_dir: str, dataset_type: str, keypoints_type: str, use_depth: bool):
    print('Loading dataset...')
    dataset_path = data_dir + 'data_3d_' + dataset_type + '.npz'

    from datasets.custom_dataset import Custom_dataset
    dataset = Custom_dataset(dataset_path, remove_static_joints=False)

    print('Preparing data {}'.format(dataset_type))
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_3d = []
            for cam in anim.keys():
                for seg in anim[cam].keys():
                    pos_3d = anim[cam][seg]
                    pos_3d /= 1000  # mm to m
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    anim[cam][seg] = pos_3d

    print('Loading 2D detections...')

    keypoints = np.load(data_dir + 'data_2d_' + dataset_type +
                        '_' + keypoints_type + '.npz', allow_pickle=True)
    # keypoints_metadata = keypoints['metadata'].item()
    # keypoints_metadata = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(dataset.skeleton().joints_left()), list(
        dataset.skeleton().joints_right())
    keypoints_metadata = [kps_left, kps_right]  # not use
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(
        dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()
    depth_vecs = {}

    if use_depth:
        print("Loading depth vec...")
        depth_vecs = np.load(data_dir+'data_dep'+'_'+dataset_type
                             + '.npz', allow_pickle=True)
        depth_vecs = depth_vecs['depths'].item()

    valid_indexes = dataset.valid_indexes()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(
            subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
                action, subject)

            for cam in keypoints[subject][action].keys():
                for seg in keypoints[subject][action][cam].keys():
                    kps = keypoints[subject][action][cam][seg][:,
                                                               valid_indexes]
                    if use_depth:
                        d_vec = depth_vecs[subject][action][cam][seg][:,
                                                                      valid_indexes]
                        kps = np.concatenate((kps, d_vec), -1)
                        assert kps.shape[-1] == 3
                    kps[..., :2] = normalize_screen_coordinates(
                        kps[..., :2], w=640, h=480)
                    if use_depth:
                        assert kps.shape[-1] == 3, "No depth dimentions with tensor shape: {}".format(
                            kps.shape)
                        kps[..., 2] = kps[..., 2] / 10000  # TODO: better norm
                    keypoints[subject][action][cam][seg] = kps

    return dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right


def gen_outputs(model_pos, dataset, keypoints, pad, causal_shift, actions_test, subjects_test):
    N = 0
    total_loss = 0.0
    output = {}
    with torch.no_grad():
        model_pos.eval()
        model_pos.cuda()
        for subject in subjects_test:
            output[subject] = {}
            for action in actions_test:
                output[subject][action] = {}
                for cam in dataset[subject][action].keys():
                    output[subject][action][cam] = {}
                    for seg in dataset[subject][action][cam].keys():
                        gt_3d = dataset[subject][action][cam][seg]

                        input_2d = keypoints[subject][action][cam][seg]

                        input_2d = np.pad(input_2d,
                                          ((pad + causal_shift, pad -
                                            causal_shift), (0, 0), (0, 0)),
                                          'edge')

                        input_2d = torch.from_numpy(
                            input_2d).unsqueeze(0).cuda()
                        gt_3d = torch.from_numpy(gt_3d).unsqueeze(0).cuda()

                        gt_3d[:, :, 0] = 0

                        predict_3d = model_pos(input_2d)

                        error = mpjpe(predict_3d, gt_3d)

                        total_loss += gt_3d.shape[0] * \
                            gt_3d.shape[1] * error.item()
                        N += gt_3d.shape[0] * gt_3d.shape[1]

                        output[subject][action][cam][seg] = predict_3d.cpu(
                        ).squeeze().numpy()
    return output, total_loss / N


@hydra.main(config_path=config_path, config_name="config")
def main(cfg):
    log.info("Config path: {}".format(config_path + 'config.yaml'))

    dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right = load_dataset_raw(cfg.data_dir,
                                                                                                              cfg.dataset, cfg.keypoints, cfg.depth_map)
    actions_test = cfg.actions_test.split(',')
    subjects_test = cfg.subjects_test.split(',')
    log.info("actions test: {}".format(actions_test))

    njoints = 21
    features = 2 + cfg.depth_map

    poses_valid_2d = [np.ones((njoints, features))]

    model_pos_train, model_pos, pad, causal_shift = create_model(
        cfg, dataset, poses_valid_2d)

    chk_filename = config_path.replace(
        '.hydra', 'checkpoint') + 'epoch_best.bin'
    checkpoint = torch.load(chk_filename)
    model_pos.load_state_dict(checkpoint["model_pos"])
    log.info("Begin prediction...")

    output, total_loss = gen_outputs(model_pos, dataset, keypoints, pad,
                                     causal_shift, actions_test, subjects_test)

    np.savez_compressed(output_filename, predict_3d=output)

    log.info("Finish with loss: {}".format(total_loss * 1000))


if __name__ == '__main__':
    main()

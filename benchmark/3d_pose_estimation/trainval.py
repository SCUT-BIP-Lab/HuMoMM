import os

from alive_progress import alive_bar
import numpy as np
import torch

from common.camera import normalize_screen_coordinates, world_to_camera
from common.loss import mpjpe, p_mpjpe
from common.utils import deterministic_random
from model.VideoPose3D import TemporalModel, TemporalModelOptimized1f
from model.Gastnet.gast_net import SpatioTemporalModel, SpatioTemporalModelOptimized1f
from graph_utils import adj_mx_from_skeleton
from common.skeleton import Skeleton


def load_dataset(data_dir: str, dataset_type: str, keypoints_type: str):
    print('Loading dataset...')
    dataset_path = data_dir + 'data_3d_' + dataset_type + '.npz'

    if dataset_type == "h36m":
        from datasets.h36m import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')

    print('Preparing data')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(
                        anim['positions'], R=cam['orientation'], t=cam['translation'])
                    # Remove global offset, but keep trajectory in first position
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints = np.load(data_dir + 'data_2d_' + dataset_type +
                        '_' + keypoints_type + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(
        keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(
        dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(
            subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
                action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(
                dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps
    return dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right


def load_dataset_ntu(data_dir: str, dataset_type: str, keypoints_type: str, use_depth: bool):
    print('Loading dataset...')
    dataset_path = data_dir + 'data_3d_' + dataset_type + '.npz'

    if dataset_type == "ntu":
        from datasets.ntu_rgbd import NTU_RGBD
        dataset = NTU_RGBD(dataset_path)
    elif dataset_type == "custom" or dataset_type == "custom_ba":
        from datasets.custom_dataset import Custom_dataset
        dataset = Custom_dataset(dataset_path, remove_static_joints=False)
    else:
        raise KeyError('Invalid dataset')

    print('Preparing data {}'.format(dataset_type))
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_3d = []
            for cam in anim.keys():
                for seg in anim[cam].keys():
                    pos_3d = anim[cam][seg]
                    if dataset_type == "custom" or dataset_type == "custom_ba":
                        pos_3d /= 1000  # mm to m
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

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
            if 'positions_3d' not in dataset[subject][action]:
                continue

            keypoints_2d = []
            for cam in keypoints[subject][action].keys():
                for seg in keypoints[subject][action][cam].keys():
                    kpt_2d = keypoints[subject][action][cam][seg][:,
                                                                  valid_indexes]
                    if use_depth:
                        d_vec = depth_vecs[subject][action][cam][seg][:,
                                                                      valid_indexes]
                        kpt_2d = np.concatenate((kpt_2d, d_vec), -1)
                        assert kpt_2d.shape[-1] == 3
                    keypoints_2d.append(kpt_2d)

            keypoints[subject][action] = keypoints_2d
            assert len(keypoints[subject][action]) == len(
                dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            if len(keypoints[subject][action])!=10:
                print('haha')
            for seg_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=640, h=480)
                if use_depth:
                    assert kps.shape[-1] == 3, "No depth dimentions with tensor shape: {}".format(
                        kps.shape)
                    kps[..., 2] = kps[..., 2] / 10000  # TODO: better norm

                keypoints[subject][action][seg_idx] = kps

    return dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right


def fetch(subjects, dataset, keypoints, action_filter=None, downsample=5, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(
                0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    return out_camera_params, out_poses_3d, out_poses_2d


def fetch_ntu(subjects, dataset, keypoints, action_filter=None, view=None, downsample=5, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across segs
                if str(i//2) in view:
                    out_poses_2d.append(poses_2d[i])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'seg count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    if str(i//2) in view:
                        out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(
                0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    return out_camera_params, out_poses_3d, out_poses_2d


def create_model(cfg, dataset, poses_valid_2d):
    filter_widths = [int(x) for x in cfg.arch.split(",")]
    if cfg.model_name == 'videopose3d':
        if not cfg.disable_optimizations and not cfg.dense and cfg.stride == 1:
            # Use optimized model for single-frame predictions
            model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                                    filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels)
        else:
            # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
            model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                            filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels,
                                            dense=cfg.dense)

        model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels,
                                dense=cfg.dense)
    elif cfg.model_name == 'gastnet':
        scut_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                           joints_left=[5, 6, 7, 8, 13, 14, 15, 16],
                           joints_right=[9, 10, 11, 12, 17, 18, 19, 20])
        adj = adj_mx_from_skeleton(scut_skeleton)
        if not cfg.disable_optimizations and not cfg.dense and cfg.stride == 1:
            # Use optimized model for single-frame predictions
            model_pos_train = SpatioTemporalModelOptimized1f(adj,poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                                    filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels)
        else:
            # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
            model_pos_train = SpatioTemporalModel(adj,poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                            filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels,
                                            dense=cfg.dense)

        model_pos = SpatioTemporalModel(adj, poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels,
                                dense=cfg.dense)
        

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # padding on each side
    if cfg.causal:
        causal_shift = pad
    else:
        causal_shift = 0

    return model_pos_train, model_pos, pad, causal_shift


def load_weight(cfg, model_pos_train, model_pos):
    checkpoint = dict()
    if cfg.resume or cfg.evaluate:
        chk_filename = cfg.resume if cfg.resume else cfg.evaluate
        print("Loading checkpoint", chk_filename)
        checkpoint = torch.load(chk_filename)
        # print("This model was trained for {} epochs".format(checkpoint["epoch"]))
        model_pos_train.load_state_dict(checkpoint["model_pos"])
        model_pos.load_state_dict(checkpoint["model_pos"])

    return model_pos_train, model_pos, checkpoint


def train(accelerator, model_pos_train, train_loader, optimizer):
    epoch_loss_3d_train = 0
    N = 0

    # TODO dataloader and tqdm
    total = len(train_loader)
    with alive_bar(total, title='Train', spinner='elements') as bar:
        for batch_data in train_loader:
            inputs_3d, inputs_2d = batch_data[-2], batch_data[-1]
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_train += inputs_3d.shape[0] * \
                inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos

            # accelerator backward
            accelerator.backward(loss_total)

            optimizer.step()

            bar()

    epoch_losses_eva = epoch_loss_3d_train / N

    return epoch_losses_eva


def eval(model_train_dict, model_pos, test_loader, train_loader_eval):
    N = 0
    epoch_loss_3d_valid = 0
    epoch_loss_3d_train_eval = 0

    with torch.no_grad():
        model_pos.load_state_dict(model_train_dict)
        model_pos.eval()

        # Evaluate on test set
        total_test = len(test_loader)
        with alive_bar(total_test, title='Test ', spinner='flowers') as bar:
            for batch_data in test_loader:
                inputs_3d, inputs_2d = batch_data[-2], batch_data[-1]

                inputs_3d[:, :, 0] = 0

                # Predict 3D poses
                predicted_3d_pos = model_pos(inputs_2d)
                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_valid += inputs_3d.shape[0] * \
                    inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                bar()

        losses_3d_valid_ave = epoch_loss_3d_valid / N

        # Evaluate on training set, this time in evaluation mode
        N = 0
        total_eval = len(train_loader_eval)
        with alive_bar(total_eval, title='Eval ', spinner='flowers') as bar:
            for batch_data in train_loader_eval:
                inputs_3d, inputs_2d = batch_data[-2], batch_data[-1]
                if inputs_2d.shape[1] == 0:
                    # This happens only when downsampling the dataset
                    continue

                inputs_3d[:, :, 0] = 0

                # Compute 3D poses
                predicted_3d_pos = model_pos(inputs_2d)
                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_train_eval += inputs_3d.shape[0] * \
                    inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                bar()

        losses_3d_train_eval_ave = epoch_loss_3d_train_eval / N

    return losses_3d_valid_ave, losses_3d_train_eval_ave


def prepare_actions(subjects_test, dataset):
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append(
                (subject, action))
    return all_actions, all_actions_by_subject


def fetch_actions(actions, keypoints, dataset, view_filters, downsample=1):
    out_poses_3d = []
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)):  # Iterate across camera
            if str(i//2) in view_filters:
                out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)):  # Iterate across cameras
            if str(i//2) in view_filters:
                out_poses_3d.append(poses_3d[i])

    stride = downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d


def evaluate(test_loader, model_pos, action=None, log=None, joints_left=None, joints_right=None, test_augment=True):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for batch_data in test_loader:
            inputs_3d, inputs_2d = batch_data[-2], batch_data[-1]

            if test_augment:
                inputs_2d = torch.squeeze(inputs_2d, 0)
                inputs_3d = torch.squeeze(inputs_3d, 0)

            inputs_3d[:, :, 0] = 0
            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            if test_augment:
                assert joints_left is not None and joints_right is not None
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left +
                                 joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(
                    predicted_3d_pos, dim=0, keepdim=True)

                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos += inputs_3d.shape[0] * \
                inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1,
                                                     inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy(
            ).reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * \
                inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000

    if log is not None:
        if action is None:
            log.info('----------')
        else:
            log.info('----{}----'.format(action))

        log.info('Protocol #1 Error (MPJPE): {} mm'.format(e1))
        log.info('Protocol #2 Error (P-MPJPE): {} mm'.format(e2))
        log.info('----------')

    return e1, e2


def predict(test_generator, model_pos):
    with torch.no_grad():
        model_pos.eval()
        batch_2d = next(test_generator.next_epoch())[-1]
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()
        predicted_3d_pos = model_pos(inputs_2d)
    return predicted_3d_pos.squeeze(0).cpu().numpy()

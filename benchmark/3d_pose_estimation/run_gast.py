import logging
from typing import Iterator
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import sys
import errno
import random
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from common.utils import summary
from common.dataset_generators import UnchunkedGeneratorDataset, ChunkedGeneratorDataset
from trainval import create_model, fetch_ntu, load_dataset, fetch, load_dataset_ntu, load_weight, train, eval, prepare_actions, fetch_actions, evaluate

log = logging.getLogger('hpe-3d')


@hydra.main(config_path="config/", config_name="gast_conf")
def main(cfg: DictConfig):
    mpjpe = []
    p_mpjpe = []
    log.info('Config:\n' + OmegaConf.to_yaml(cfg))

    if cfg.resume and cfg.evaluate:
        log.error(
            'Invlid Config: resume and evaluate can not be set at the same time')
        exit(-1)

    if cfg.dataset == 'h36m' and cfg.depth_map:
        log.error('Cannot use depth map when using h36m dataset')
        exit(-1)

    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(cfg.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                'Unable to create checkpoint directory:', cfg.checkpoint)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if cfg.dataset != 'h36m':
        dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right = load_dataset_ntu(cfg.data_dir,
                                                                                                                  cfg.dataset, cfg.keypoints, cfg.depth_map)
    else:
        dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right = load_dataset(cfg.data_dir,
                                                                                                              cfg.dataset, cfg.keypoints)

    subjects_train = cfg.subjects_train.split(',')
    subjects_test = cfg.subjects_test.split(',')

    actions_test = cfg.actions_test.split(',')
    actions_train = cfg.actions_train.split(',')



    view_train=cfg.view_train.split(',')
    view_test=cfg.view_test.split(',')
    # for i in range(0, 20):
    #     action = 'A%03d' % i
    #     if action not in actions_test:
    #         actions_train.append(action)

    action_filter = None
    if actions_test is not None:
        log.info('Selected train actions:{}'.format(actions_train))
        log.info('Selected test actions:{}'.format(actions_test))
    if cfg.dataset != 'h36m':
        cameras_valid, poses_valid, poses_valid_2d = fetch_ntu(
            subjects_test, dataset, keypoints, actions_test, view_test, cfg.downsample, cfg.subset)
    else:
        cameras_valid, poses_valid, poses_valid_2d = fetch(
            subjects_test, dataset, keypoints, action_filter, cfg.downsample, cfg.subset)

    model_pos_train, model_pos, pad, causal_shift = create_model(
        cfg, dataset, poses_valid_2d)
    receptive_field = model_pos.receptive_field()
    log.info("Receptive field: {} frames".format(receptive_field))
    if cfg.causal:
        log.info("Using causal convolutions")

    # Loading weight
    model_pos_train, model_pos, checkpoint = load_weight(
        cfg, model_pos_train, model_pos)

    test_dataset = UnchunkedGeneratorDataset(cameras_valid, poses_valid, poses_valid_2d,
                                             pad=pad, causal_shift=causal_shift, augment=False,
                                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=cfg.num_workers)
    log.info("Testing on {} frames".format(test_dataset.num_frames()))

    if not cfg.evaluate:
        if cfg.dataset != 'h36m':
            cameras_train, poses_train, poses_train_2d = fetch_ntu(subjects_train,  dataset, keypoints, actions_train, view_train, 
                                                                   cfg.downsample, subset=cfg.subset)
        else:
            cameras_train, poses_train, poses_train_2d = fetch(subjects_train,  dataset, keypoints, action_filter,
                                                               cfg.downsample, subset=cfg.subset)
        lr = cfg.learning_rate
        optimizer = torch.optim.Adam(
            model_pos_train.parameters(), lr=lr, amsgrad=True)
        lr_decay = cfg.lr_decay

        losses_3d_train = []
        losses_3d_train_eval = []
        losses_3d_valid = []

        epoch = 0
        initial_momentum = 0.1
        final_momentum = 0.001

        train_dataset = ChunkedGeneratorDataset(cameras_train, poses_train, poses_train_2d,
                                                cfg.stride,
                                                pad=pad, causal_shift=causal_shift, shuffle=True, augment=cfg.data_augmentation,
                                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                                joints_right=joints_right)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        train_dataset_eval = UnchunkedGeneratorDataset(cameras_train, poses_train, poses_train_2d,
                                                       pad=pad, causal_shift=causal_shift, augment=False)
        train_loader_eval = DataLoader(
            train_dataset_eval, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

        log.info('Training on {} frames'.format(
            train_dataset_eval.num_frames()))
        sample_inputs_2d = train_dataset[0][-1]
        input_shape = [cfg.batch_size]
        input_shape += list(sample_inputs_2d.shape)
        log.info('Input 2d shape: {}'.format(
            input_shape))
        summary(log, model_pos,
                sample_inputs_2d.shape, cfg.batch_size, device='cpu')

        if cfg.resume:
            epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                log.info(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            lr = checkpoint['lr']

        log.info(
            '** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
        log.info(
            '** The final evaluation will be carried out after the last training epoch.')

        # Prepare everything for gpu and fp16
        accelerator = Accelerator(device_placement=True)
        # accelerator = Accelerator(device_placement=True, fp16=cfg.fp16)
        model_pos_train, model_pos, optimizer, train_loader, train_loader_eval, test_loader = accelerator.prepare(
            model_pos_train, model_pos, optimizer, train_loader, train_loader_eval, test_loader)
        log.info("Training on device: {}".format(accelerator.device))

        loss_min = 49.5

        # Pos model only
        while epoch < cfg.epochs:
            start_time = time()
            model_pos_train.train()

            # Regular supervised scenario
            epoch_loss_3d = train(
                accelerator, model_pos_train, train_loader, optimizer)
            losses_3d_train.append(epoch_loss_3d)

            # After training an epoch, whether to evaluate the loss of the training and validation set
            if not cfg.no_eval:
                model_train_dict = model_pos_train.state_dict()
                losses_3d_valid_ave, losses_3d_train_eval_ave = eval(
                    model_train_dict, model_pos, test_loader, train_loader_eval)
                losses_3d_valid.append(losses_3d_valid_ave)
                losses_3d_train_eval.append(losses_3d_train_eval_ave)

            elapsed = (time() - start_time) / 60

            if cfg.no_eval:
                log.info('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
            else:
                log.info('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1] * 1000))

                # Saving the best result
                if losses_3d_valid[-1]*1000 < loss_min:
                    chk_path = os.path.join(cfg.checkpoint, 'epoch_best.bin')
                    log.info('Saving checkpoint to {}'.format(chk_path))

                    torch.save({
                        'epoch': epoch,
                        'lr': lr,
                        'optimizer': optimizer.state_dict(),
                        'model_pos': model_pos_train.state_dict()
                    }, chk_path)

                    loss_min = losses_3d_valid[-1]*1000

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            epoch += 1

            # Decay BatchNorm momentum
            momentum = initial_momentum * \
                np.exp(-epoch/cfg.epochs *
                       np.log(initial_momentum/final_momentum))
            model_pos_train.set_bn_momentum(momentum)

            # Save checkpoint if necessary
            if epoch % cfg.checkpoint_frequency == 0:
                chk_path = os.path.join(
                    cfg.checkpoint, 'epoch_{}.bin'.format(epoch))
                log.info('Saving checkpoint to {}'.format(chk_path))

                torch.save({
                    'epoch': epoch,
                    'lr': lr,
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos_train.state_dict()
                }, chk_path)

            # Save training curves after every epoch, as .png images (if requested)
            if cfg.export_training_curves and epoch > 3:
                if 'matplotlib' not in sys.modules:
                    import matplotlib

                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                plt.figure()
                epoch_x = np.arange(3, len(losses_3d_train)) + 1
                plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
                plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
                plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
                plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
                plt.ylabel('MPJPE (m)')
                plt.xlabel('Epoch')
                plt.xlim((3, epoch))
                plt.savefig(os.path.join(cfg.checkpoint, 'loss_3d.png'))
                plt.close('all')
    # Evaluate
    log.info('Evaluating...')

    all_actions, all_actions_by_subject = prepare_actions(
        subjects_test, dataset)

    def run_evaluation(actions, action_filter, view_filter):
        nonlocal model_pos
        errors_p1 = []
        errors_p2 = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(
                actions[action_key], keypoints, dataset, view_filter, cfg.downsample)
            _dataset = UnchunkedGeneratorDataset(None, poses_act, poses_2d_act,
                                                 pad=pad, causal_shift=causal_shift, augment=cfg.test_time_augment, kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                                 joints_right=joints_right)
            action_loader = DataLoader(_dataset, 1, shuffle=False)
            if cfg.evaluate:
                accelerator = Accelerator(device_placement=True)
                action_loader, poses_act, poses_2d_act,model_pos = accelerator.prepare(action_loader, poses_act, poses_2d_act, model_pos)
            else:
                action_loader = accelerator.prepare_data_loader(action_loader)

            e1, e2 = evaluate(action_loader, model_pos,
                              action=action_key, log=log, joints_left=joints_left, joints_right=joints_right, test_augment=cfg.test_time_augment)
            errors_p1.append(e1)
            errors_p2.append(e2)
        erp1 = round(np.mean(errors_p1), 1)
        erp2 = round(np.mean(errors_p2), 1)
        log.info('Protocol #1   (MPJPE) action-wise average: {} mm'.format(erp1))
        log.info('Protocol #2 (P-MPJPE) action-wise average: {} mm'.format(erp2))
        return erp1, erp2

    if not cfg.by_subject:
        run_evaluation(all_actions, actions_test, view_test)
    else:
        for subject in all_actions_by_subject.keys():
            log.info('Evaluating on subject: {}'.format(subject))
            erp1,erp2= run_evaluation(all_actions_by_subject[subject], actions_test, view_test)
            mpjpe.append(erp1)
            p_mpjpe.append(erp2)
            log.info('')
    print('MPJPE:', np.mean(mpjpe))
    print('P_MPJPE:', np.mean(p_mpjpe))
    log.info('Protocol #1   (MPJPE) average: {} mm'.format(np.mean(mpjpe)))
    log.info('Protocol #2 (P-MPJPE) average: {} mm'.format(np.mean(p_mpjpe)))


if __name__ == '__main__':
    main()

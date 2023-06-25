from torchvision.models.resnet import resnet18
import torch
import torch.nn as nn

from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import L2FilterPruner

import os, sys, time


'''
pip install nni
'''

def prune_light_openpose_model(model):
    # 指定剪枝层名
    op_names = ['model.0.0']
    op_names += ['model.{}.0'.format(i) for i in range(1, 12)]
    op_names += ['model.{}.3'.format(i) for i in range(1, 12)]
    op_names += ['initial_stage.trunk.{}.0'.format(i) for i in range(3)]
    op_names += ['initial_stage.heatmaps.{}.0'.format(i) for i in range(1)] # 最后一层输出不剪枝
    op_names += ['initial_stage.pafs.{}.0'.format(i) for i in range(1)]
    # op_names += ['refinement_stages.0.trunk.{}.initial.0'.format(i) for i in range(5)]
    # op_names += ['refinement_stages.0.trunk.{}.trunk.0.0'.format(i) for i in range(5)]
    # op_names += ['refinement_stages.0.trunk.{}.trunk.1.0'.format(i) for i in range(5)]
    # op_names += ['refinement_stages.0.heatmaps.{}.0'.format(i) for i in range(1)]
    # op_names += ['refinement_stages.0.pafs.{}.0'.format(i) for i in range(1)]

    config_list = [{
        'sparsity': 0.2,  # 修剪率
        'op_types': ['Conv2d'],  # op类型
        'op_names': op_names,
    }]
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    input_size = [1, 3, 256, 216]
    dummy_input = torch.randn(input_size).to(device)

    tic = time.time()
    _ = model(dummy_input)
    inf_time = time.time() - tic

    flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
    print(f"Model FLOPs {flops/1e6:.2f}M, Params {params/1e6:.2f}M, Time {inf_time}S")


    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')


    # 采用L2剪枝器测试，后续可更换其他剪枝器
    pruner = L2FilterPruner(model, config_list)
    pruner.compress()
    pruner.export_model(model_path="./tmp/prune_model.pth", mask_path="./tmp/prune_model_mask.pth")
    # 合并模型和剪枝mask，完成模型加速
    pruner._unwrap_model()
    m_speedup = ModelSpeedup(model, dummy_input, "./tmp/prune_model_mask.pth", device)
    m_speedup.speedup_model()
    tic = time.time()
    _ = model(dummy_input)
    inf_time = time.time() - tic

    flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
    print(f"Model FLOPs {flops/1e6:.2f}M, Params {params/1e6:.2f}M, Time {inf_time}S")




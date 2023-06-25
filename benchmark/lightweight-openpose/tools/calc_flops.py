import sys
sys.path.append('.')

import torch
from models import get_pose_estimation_model
from nni.compression.pytorch.utils.counter import count_flops_params
from thop import profile
from thop import clever_format

model = get_pose_estimation_model('hrnet', num_heatmaps=22, num_pafs=48)
device = 'cpu'
input_size = [1, 3, 256, 256]
dummy_input = torch.randn(input_size).to(device)

# _ = model(dummy_input)

# flops, params, _ = count_flops_params(model, dummy_input, verbose=True)


flops_op, params_op = profile(model, inputs=(dummy_input,))
flops_op, params_op = clever_format([flops_op, params_op], "%.3f")

print('FLOPs = ' + str(flops_op))
print('Params = ' + str(params_op))

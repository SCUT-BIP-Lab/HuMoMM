import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('.')
import time
import torch
from thop import profile
from thop import clever_format
from models.hrnet import PoseHighResolutionNet
from models.small_hrnet import SmallHighResolutionNet
from utils.utils import AverageMeter

import yaml
from yaml.loader import SafeLoader

from addict import Dict

# Open the file and load the file
# with open('models/hrnet_l3_hc.yaml') as f:
#     cfg = yaml.load(f, Loader=SafeLoader)

# cfg =Dict(cfg)
# hrnet =PoseHighResolutionNet(cfg)
with open('models/small_hrnet_v1.yaml') as f:
    cfg = yaml.load(f, Loader=SafeLoader)

cfg =Dict(cfg)
hrnet =SmallHighResolutionNet(cfg)
hrnet=hrnet.cuda()
hrnet.eval()

inputs = torch.rand(1, 3, 256, 480).cuda()

flops, params = profile(hrnet, inputs=(inputs,))
flops, params = clever_format([flops, params], "%.3f")
print('FLOPs = ' + str(flops))
print('Params = ' + str(params))


infer_time_avg = AverageMeter()
for i in range(50):
    start_time=time.time()
    out=hrnet(inputs)
    infer_time=time.time()
    # print(f'infer network time: {(infer_time - start_time) * 1000:.0f}ms')
    infer_time_avg.update((infer_time - start_time) * 1000)
print(f'avg infer network time: {infer_time_avg.avg:.0f}ms')




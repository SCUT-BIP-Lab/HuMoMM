import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch
from thop import profile
from thop import clever_format
from models.litehrnet import LiteHRNet
from mmpose.models.heads import TopdownHeatmapSimpleHead

from utils.utils import AverageMeter


extra = dict(
    stem=dict(
        stem_channels=32,
        out_channels=32,
        expand_ratio=1),
    num_stages=3,
    stages_spec=dict(
        num_modules=(3, 8, 3),
        num_branches=(2, 3, 4),
        num_blocks=(2, 2, 2),
        module_type=('LITE', 'LITE', 'LITE'),
        with_fuse=(True, True, True),
        reduce_ratios=(8, 8, 8),
        num_channels=(
            (40, 80),
            (40, 80, 160),
            (40, 80, 160, 320),
        )),
    with_head=True,
)
# extra=dict(
#     stem=dict(  
#         stem_channels=32,
#         out_channels=32,
#         expand_ratio=1),
#     num_stages=3,
#     stages_spec=dict(
#         num_modules=(2, 4, 2),
#         num_branches=(2, 3, 4),
#         num_blocks=(2, 2, 2),
#         module_type=('LITE', 'LITE', 'LITE'),
#         with_fuse=(True, True, True),
#         reduce_ratios=(8, 8, 8),
#         num_channels=(
#             (40, 80),
#             (40, 80, 160),
#             (40, 80, 160, 320),
#         )),
#     with_head=True,)

lite_backbone = LiteHRNet(extra)

# head
head_cfg = dict(
    in_channels=40,
    out_channels=21,
    num_deconv_layers=0,
    extra=dict(final_conv_kernel=1, ),
    loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

simple_head=TopdownHeatmapSimpleHead(**head_cfg)



# inputs = torch.rand(1, 3, 256, 256)
# flops, params = profile(lite_backbone, inputs=(inputs,))
# flops, params = clever_format([flops, params], "%.3f")
# print('FLOPs = ' + str(flops))
# print('Params = ' + str(params))

inputs = torch.rand(1, 3, 256, 480).cuda()
lite_backbone=lite_backbone.cuda()
simple_head=simple_head.cuda()
lite_backbone.eval()
simple_head.eval()

infer_time_avg = AverageMeter()
for i in range(100):
    start_time=time.time()
    level_outputs = lite_backbone(inputs)

    out=simple_head(level_outputs)
    infer_time=time.time()
    print(f'infer network time: {(infer_time - start_time) * 1000:.0f}ms')
    infer_time_avg.update((infer_time - start_time) * 1000)
print(f'avg infer network time: {infer_time_avg.avg:.0f}ms')


# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     lite_backbone(inputs)
# print(prof)
# prof.export_chrome_trace('profiles')

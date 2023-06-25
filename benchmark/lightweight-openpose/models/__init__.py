import yaml
from yaml import SafeLoader
from addict import Dict
from .with_mobilenet import PoseEstimationWithMobileNet
from .hrnet import PoseHighResolutionNet
from .small_hrnet import SmallHighResolutionNet



def get_pose_estimation_model(model_name, **kwargs):
    if model_name == 'mobilenetv2':
        return PoseEstimationWithMobileNet(**kwargs)
    elif model_name == 'hrnet':
        with open('models/hrnet.yaml') as f:
            cfg = yaml.load(f, Loader=SafeLoader)
        cfg = Dict(cfg)
        cfg.MODEL.NUM_JOINTS=kwargs['num_heatmaps']
        cfg.MODEL.NUM_PAFS=kwargs['num_pafs']
        return PoseHighResolutionNet(cfg)
    elif model_name == 'small_hrnet_v1':
        with open('models/small_hrnet_v1.yaml') as f:
            cfg = yaml.load(f, Loader=SafeLoader)
        cfg = Dict(cfg)
        cfg.MODEL.NUM_JOINTS=kwargs['num_heatmaps']
        cfg.MODEL.NUM_PAFS=kwargs['num_pafs']
        return SmallHighResolutionNet(cfg)
    else:
        raise ValueError('The model name is incorrect, please check.')
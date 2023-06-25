from .coco import CocoTrainDataset, CocoValDataset
from .ntu_rgbd import NTUTrainDataset, NTUValDataset
from .scut_rgbd import SCUTTrainDataset, SCUTValDataset


def get_train_dataset(dataset_name, **kwargs):
    if dataset_name == 'ntu':
        return NTUTrainDataset(**kwargs)
    elif dataset_name == 'coco':
        return CocoTrainDataset(**kwargs)
    elif dataset_name == 'scut':
        return SCUTTrainDataset(**kwargs)
    else:
        raise ValueError('the dataset name is incorrect.')


def get_val_dataset(dataset_name, **kwargs):
    if dataset_name == 'ntu':
        return NTUValDataset(**kwargs)
    elif dataset_name == 'coco':
        return CocoValDataset(**kwargs)
    elif dataset_name == 'scut':
        return SCUTValDataset(**kwargs)
    else:
        raise ValueError('the dataset name is incorrect.')

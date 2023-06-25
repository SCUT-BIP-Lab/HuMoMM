# RGBD三维姿态估计：自动标注

本项目采用HRNet作为自动标注模型，将人工标注的关键帧作为训练数据，训练后的HRNet模型在剩余帧推理得到自动标注的2D姿态标签。


## 目录
- [RGBD三维姿态估计：自动标注](#rgbd三维姿态估计自动标注)
  - [目录](#目录)
  - [直接标注](#直接标注)
  - [自动训练+标注](#自动训练标注)
  - [手动训练+标注](#手动训练标注)


## 直接标注
可以直接使用我们提供的自动标注模型完成自动标注，[自动标注模型](), 命令如下：
```
CUDA_VISIBLE_DEVICES=1  python tools/auto_labeling.py \
--data_root /path/to/dataset \
--manual_label_path /path/to/manual/label \
--auto_label_path /path/to/auto/label \
--model_file /path/to/model
```
参数说明：
- data_root: 数据集的路径
- manual_label_path: 人工标注的关键帧标签路径
- auto_label_path: 输出的自动标注标签路径
- model_file: 训练得到的自动标注模型路径


## 自动训练+标注
下载coco上的预训练模型，并放入`./prepared_models`下，下载地址[HRNet](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)。
再使用[tools/auto_labeling_with_training.py](tools/auto_labeling_with_training.py)脚本训练自动标注模型，并采用该模型完成自动标注。命令如下：
```
CUDA_VISIBLE_DEVICES=1 python tools/auto_labeling_with_training.py \
--data_root path/to/dataset \
--manual_label_path path/to/manual/label \
--auto_label_path /path/to/auto/label \
--loop_steps 1
```
参数说明：
- data_root: 数据集的路径
- manual_label_path: 人工标注的关键帧标签路径
- auto_label_path: 输出的自动标注标签路径
- loop_steps: 自动标注的迭代次数，默认1，即使用关键帧数据训练，一次性预测其他所有剩余帧。


## 手动训练+标注
手动训练模型，并用训练得到的模型完成自动标注。

1. 准备2D训练标签

使用[tools/prepare_scut_rgbd.py](./tools/prepare_scut_rgbd.py)和[tools/prepare_scut_val.py](./tools/prepare_scut_val.py)完成2D标签生成。

2. 训练

下载coco上的预训练模型，并放入`./prepared_models`下，下载地址[HRNet](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)。

再使用[tools/train.py](./tools/train.py)完成模型训练。命令如下：
```
python tools/train.py --cfg  experiments/scut_rgbd/hrnet/w48_256x192_adam_lr1e-3.yaml \
                        DATASET.LABEL_DIR path/to/2d_label \
                        DATASET.ROOT path/to/data
```
3. 自动标注
   
使用[tools/auto_labeling.py](./tools/auto_labeling.py)完成自动标注，命令如下：
```
CUDA_VISIBLE_DEVICES=1  python tools/auto_labeling.py \
--data_root /path/to/dataset \
--manual_label_path /path/to/manual/label \
--auto_label_path /path/to/auto/label \
--model_file /path/to/model
```
参数说明：
- data_root: 数据集的路径
- manual_label_path: 人工标注的关键帧标签路径
- auto_label_path: 输出的自动标注标签路径
- model_file: 训练得到的自动标注模型路径




   


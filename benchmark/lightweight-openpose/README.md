# RGBD三维人体姿态估计：2D姿态估计训练代码

本仓库主要包含RGBD三维人体姿态估计项目中的2D人体姿态部分，采用lightweight-openpose作为基础算法，并做了针对性改进。

## 目录

- [RGBD三维人体姿态估计：2D姿态估计训练代码](#rgbd三维人体姿态估计2d姿态估计训练代码)
  - [目录](#目录)
  - [环境](#环境)
  - [训练](#训练)
  - [验证](#验证)
  - [演示](#演示)

## 环境

* Ubuntu >=18.04
* Python >=3.8
* PyTorch >=1.7
* 使用 `pip install -r requirements.txt`完成依赖库的安装

## 训练

1. 准备训练数据集

使用[tools/prepare_scut_label.py](./tools/prepare_scut_label.py)和[tools/prepare_scut_val.py](./tools/prepare_scut_val.py)完成标签的生成。

2. 训练

使用以下命令完成SCUT数据集的训练：
```
exp_name="scut_xsub"
if [ ! -d logs/${exp_name} ];then
   mkdir -p logs/${exp_name}
fi

PYTHONPATH=".":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 python -u train.py --arch mobilenetv2 --dataset scut \
--prepared-train-labels   ./tmp/2d_labels/train_label.pkl \
--train-images-folder  /home/zzj/dataset/scut_key_frame/RGB_frame \
--val-images-folder /home/zzj/dataset/scut_key_frame/RGB_frame \
--val-labels ./tmp/2d_labels/val_label.json \
--checkpoint-path  ./pretrained_models/checkpoint_iter_370000.pth \
--weights-only --batch-size 32 --num-workers 8 --val-after 1000 \
--experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log
```

参数说明：
- arch：使用的backbone模型，可选`mobilenetv2`, `hrnet`, `small_hrnet_v1`.
- dataset: 使用的训练数据集：可选`scut`, `ntu`, `coco`.
- prepared-train-labels: 使用[tools/prepare_scut_label.py](./tools/prepare_scut_label.py)生成的训练标签文件
- train-images-folder: 训练数据集RGB图片路径
- val-images-folder: 验证数据集RGB图片路径
- val-labels: 使用[tools/prepare_scut_val.py](./tools/prepare_scut_val.py)生成的验证json文件
- checkpoint-path: 预训练模型文件，训练scut和ntu数据集时，加载coco数据集上预训练的模型；训练coco数据集时，加载imagenet的预训练模型

PS：可以直接运行脚本[scripts/train_scut.sh](./scripts/train_scut.sh)完成训练，其他数据集同理。

预训练模型下载地址：
- coco
  - [mobilenetv2](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)
  - [hrnet](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC)
  - [small_hrnet_v1]()
- imagenet
  - [mobilenetv2](https://github.com/marvis/pytorch-mobilenet)
  - [hrnet](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC)
  - [small_hrnet_v1](https://onedrive.live.com/?authkey=%21APY8jW-MnKfaDsY&id=F7FD0B7F26543CEB%21155&cid=f7fd0b7f26543ceb)

## 验证

使用以下命令完成验证：
```
PYTHONPATH=".":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 \
python val.py --arch small_hrnet_v1 --dataset scut \
--label ./tmp/2d_labels/val_label.json \
--images-folder /home/zzj/dataset/scut_key_frame/RGB_frame \
--checkpoint-path logs/scut_xsub/checkpoints/checkpoint_iter_45000.pth \
```
参数说明：
- images-folder: 验证数据集RGB图片路径
- label: 使用[tools/prepare_scut_val.py](./tools/prepare_scut_val.py)生成的验证json文件
- checkpoint-path: 上一步训练得到的模型文件

PS： 可以使用[scripts/val_scut.py](./scripts/val_scut.sh)完成验证，其他数据集同理。

## 演示

使用[demo.py](./demo.py)可以完成图片或视频的推理演示，命令如下：
```
python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video 0
```

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python val.py --arch mobilenetv2 --dataset scut \
--label /home/zx/exp_wmh/data/2d_pose_anno/val_label_xview.json \
--images-folder /data/pose_datasets/scut_sp/RGB_frame \
--checkpoint-path  logs/scut_mobilenetv2_xview/checkpoints/checkpoint_iter_50000.pth
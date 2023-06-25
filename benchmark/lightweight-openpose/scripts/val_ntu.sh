PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 \
python val.py --arch mobilenetv2 --dataset coco \
--label ../datasets/ntu_rgbd/nturgb+d_2d_labels/val_xsub_label_subset100.json \
--images-folder ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
--checkpoint-path ./logs/ft_coco_256/checkpoints/checkpoint_iter_5000.pth \
--input-height 256
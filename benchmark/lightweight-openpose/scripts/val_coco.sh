# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 \
# python val.py --arch mobilenetv2 --dataset coco \
# --label ./coco_data/val_subset.json \
# --images-folder ../datasets/coco2017/val2017 \
# --checkpoint-path ./logs/train_coco_mobilenetv2/checkpoints/checkpoint_iter_100000.pth \
# --input-height 256 



# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 \
# python val.py --arch hrnet --dataset coco \
# --label ./coco_data/val_subset.json \
# --images-folder ../datasets/coco2017/val2017 \
# --checkpoint-path logs/coco_hrnet/checkpoints/checkpoint_iter_210000.pth \
# --input-height 256 



PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 \
python val.py --arch small_hrnet_v1 --dataset coco \
--label ./coco_data/val_subset.json \
--images-folder ../datasets/coco2017/val2017 \
--checkpoint-path logs/train_coco_small_hrnet/checkpoints/checkpoint_iter_200000.pth \
--input-height 256 
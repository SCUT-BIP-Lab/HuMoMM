# exp_name="test"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch mobilenetv2 --dataset ntu \
# --prepared-train-labels   ../datasets/ntu_rgbd/nturgb+d_2d_labels/train_xsub_label_256.pkl \
# --train-images-folder  ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-images-folder ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-labels ../datasets/ntu_rgbd/nturgb+d_2d_labels/val_xsub_label_subset100.json \
# --checkpoint-path  ./pretrained_models/checkpoint_iter_370000.pth \
# --weights-only --batch-size 32 --num-workers 8 --val-after 1000 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

# exp_name="test"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch small_hrnet_v1 --dataset ntu \
# --prepared-train-labels   ../datasets/ntu_rgbd/nturgb+d_2d_labels/train_xsub_label_256.pkl \
# --train-images-folder  ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-images-folder ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-labels ../datasets/ntu_rgbd/nturgb+d_2d_labels/val_xsub_label_subset100.json \
# --weights-only --batch-size 32 --num-workers 8 --val-after 1000 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

# exp_name="ntu_hrnet_l3"
# if [ ! -d logs/${exp_name} ];then
#     mkdir -p logs/${exp_name}
# fi

# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch hrnet --dataset ntu \
# --prepared-train-labels   ../datasets/ntu_rgbd/nturgb+d_2d_labels/train_xsub_label_256.pkl \
# --train-images-folder  ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-images-folder ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-labels ../datasets/ntu_rgbd/nturgb+d_2d_labels/val_xsub_label_subset100.json \
# --batch-size 32 --num-workers  12 --val-after 3000  --base-lr 0.001 \
# --checkpoint-path ./pretrained_models/pose_hrnet_w32_256x192.pth --weights-only \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

exp_name="ntu_small_hrnet"
if [ ! -d logs/${exp_name} ];then
    mkdir -p logs/${exp_name}
fi

CUDA_VISIBLE_DEVICES=1 python -u train.py --arch small_hrnet_v1 --dataset ntu \
--prepared-train-labels   ../datasets/ntu_rgbd/nturgb+d_2d_labels/train_xsub_label_256.pkl \
--train-images-folder  ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
--val-images-folder ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
--val-labels ../datasets/ntu_rgbd/nturgb+d_2d_labels/val_xsub_label_subset100.json \
--batch-size 32 --num-workers  12 --val-after 3000  --base-lr 0.001 \
--checkpoint-path logs/train_coco_small_hrnet/checkpoints/checkpoint_iter_200000.pth --weights-only \
--experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

# finetune with pruning

# CUDA_VISIBLE_DEVICES=1 python train_nturgbd.py \
# --prepared-train-labels ../datasets/ntu_rgbd/nturgb+d_2d_labels/train_xsub_label.pkl \
# --train-images-folder ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-images-folder ../datasets/ntu_rgbd/nturgb+d_rgb_imgs \
# --val-labels ../datasets/ntu_rgbd/nturgb+d_2d_labels/val_xsub_label_subset100.json \
# --checkpoint-path logs/ft_coco_xsub/checkpoints/checkpoint_iter_35000.pth \
# --batch-size 16 --num-workers 8 --val-after 100 --base-lr 4e-6 \
# --experiment-name ft_pruning \
# --prune --weights-only

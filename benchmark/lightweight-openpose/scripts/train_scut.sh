# exp_name="scut_xsub_test"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch mobilenetv2 --dataset scut \
# --prepared-train-labels   ./tmp/2d_labels/train_label.pkl \
# --train-images-folder  /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-images-folder /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-labels ./tmp/2d_labels/val_label.json \
# --checkpoint-path  ./pretrained_models/checkpoint_iter_370000.pth \
# --weights-only --batch-size 32 --num-workers 8 --val-after 1000 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

# exp_name="scut_hrnet_xsub"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch hrnet --dataset scut \
# --prepared-train-labels   ./tmp/2d_labels/train_label.pkl \
# --train-images-folder  /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-images-folder /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-labels ./tmp/2d_labels/val_label.json \
# --checkpoint-path  logs/coco_hrnet/checkpoints/checkpoint_iter_210000.pth \
# --weights-only --batch-size 32 --num-workers 8 --val-after 5000 --base-lr 0.00005 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

# exp_name="scut_hrnet_l3_xsub"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch hrnet --dataset scut \
# --prepared-train-labels   ./tmp/2d_labels/train_label.pkl \
# --train-images-folder  /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-images-folder /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-labels ./tmp/2d_labels/val_label.json \
# --checkpoint-path  ./pretrained_models/pose_hrnet_w32_256x192.pth \
# --weights-only --batch-size 32 --num-workers 8 --val-after 5000 --base-lr 0.0005 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log




# exp_name="scut_small_hrnet_xsub"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch small_hrnet_v1 --dataset scut \
# --prepared-train-labels   ./tmp/2d_labels/train_label.pkl \
# --train-images-folder  /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-images-folder /home/zzj/dataset/scut_key_frame/RGB_frame \
# --val-labels ./tmp/2d_labels/val_label.json \
# --checkpoint-path  logs/train_coco_small_hrnet/checkpoints/checkpoint_iter_200000.pth \
# --weights-only --batch-size 32 --num-workers 8 --val-after 5000 --base-lr 0.0001 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

# exp_name="scut_mobilenetv2_xsub"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH=".":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch mobilenetv2 --dataset scut \
# --prepared-train-labels   /home/zx/exp_wmh/data/2d_pose_anno/train_label_xsub.pkl \
# --train-images-folder  /data/pose_datasets/scut_sp/RGB_frame \
# --val-images-folder /data/pose_datasets/scut_sp/RGB_frame \
# --val-labels /home/zx/exp_wmh/data/2d_pose_anno/val_label_xsub.json \
# --checkpoint-path  ./pretrained_models/checkpoint_iter_370000.pth \
# --weights-only --batch-size 32 --num-workers 8 --val-after 5000 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log



# exp_name="scut_mobilenetv2_xaction"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH=".":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch mobilenetv2 --dataset scut \
# --prepared-train-labels   /home/zx/exp_wmh/data/2d_pose_anno/train_label_xaction.pkl \
# --train-images-folder  /data/pose_datasets/scut_sp/RGB_frame \
# --val-images-folder /data/pose_datasets/scut_sp/RGB_frame \
# --val-labels /home/zx/exp_wmh/data/2d_pose_anno/val_label_xaction.json \
# --checkpoint-path  ./pretrained_models/checkpoint_iter_370000.pth \
# --weights-only --batch-size 32 --num-workers 8 --val-after 5000 \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log



exp_name="scut_mobilenetv2_xview"
if [ ! -d logs/${exp_name} ];then
   mkdir -p logs/${exp_name}
fi

PYTHONPATH=".":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 python -u train.py --arch mobilenetv2 --dataset scut \
--prepared-train-labels   /home/zx/exp_wmh/data/2d_pose_anno/train_label_xview.pkl \
--train-images-folder  /data/pose_datasets/scut_sp/RGB_frame \
--val-images-folder /data/pose_datasets/scut_sp/RGB_frame \
--val-labels /home/zx/exp_wmh/data/2d_pose_anno/val_label_xview.json \
--checkpoint-path  ./pretrained_models/checkpoint_iter_370000.pth \
--weights-only --batch-size 32 --num-workers 8 --val-after 5000 \
--experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log
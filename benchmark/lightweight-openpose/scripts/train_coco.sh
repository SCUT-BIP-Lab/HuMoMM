# exp_name="train_coco_mobilenetv2_368_paf"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch mobilenetv2 --dataset coco \
# --prepared-train-labels   ./coco_data/prepared_train_annotation_368.pkl \
# --train-images-folder  ../datasets/coco2017/train2017 \
# --val-images-folder ../datasets/coco2017/val2017 \
# --val-labels ./coco_data/val_subset.json \
# --batch-size 32 --num-workers 12 --val-after 5000 \
# --input-size 368 \
# --checkpoint-path pretrained_models/mobilenet_sgd_68.848.pth.tar --weights-only --from-mobilenet \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log

exp_name="coco_small_hrnet"
if [ ! -d logs/${exp_name} ];then
   mkdir -p logs/${exp_name}
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 python -u train.py --arch small_hrnet_v1 --dataset coco \
--prepared-train-labels   ./coco_data/prepared_train_annotation.pkl \
--train-images-folder  ../datasets/coco2017/train2017 \
--val-images-folder ../datasets/coco2017/val2017 \
--val-labels ./coco_data/val_subset.json \
--batch-size 64 --num-workers 12 --val-after 5000 \
--input-size 256 \
--base-lr 0.001 \
--checkpoint-path pretrained_models/hrnet_w18_small_model_v1.pth --weights-only \
--experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log



# exp_name="coco_hrnet"
# if [ ! -d logs/${exp_name} ];then
#    mkdir -p logs/${exp_name}
# fi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 python -u train.py --arch hrnet --dataset coco \
# --prepared-train-labels   ./coco_data/prepared_train_annotation.pkl \
# --train-images-folder  ../datasets/coco2017/train2017 \
# --val-images-folder ../datasets/coco2017/val2017 \
# --val-labels ./coco_data/val_subset.json \
# --batch-size 64 --num-workers 12 --val-after 10000 \
# --input-size 256 \
# --base-lr 0.001 \
# --checkpoint-path pretrained_models/imagenet_hrnet_w32.pth --weights-only \
# --experiment-name ${exp_name}  2>&1 | tee logs/${exp_name}/train.log
CUDA_VISIBLE_DEVICES=1 python valid.py \
--cfg experiments/scut/mobilenet/supermobile_xaction.yaml \
--superconfig mobile_configs/search-M.json \
TEST.MODEL_FILE output_xaction/scut_kpt/pose_supermobilenet/supermobile_xaction/model_best.pth.tar
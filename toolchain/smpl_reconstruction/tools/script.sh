python tools/auto_labeling.py --data_root /data/pose_datasets/scut --manual_label_path /home/zzj/dataset/scut_key_frame/Label --auto_label_path /data/pose_datasets/scut/auto_label_1

python update_bundle_adjusted_result.py --colmap_txt_root /data/pose_datasets/scut2/Colmap_txt/ --update_root /data/pose_datasets/scut2/Label_3d/ --output_root /data/pose_datasets/scut2/Label_3d_ba/ --cam_param_path ./cam_param/cam_param_2/cam_param_pkg.npz --vis

CUDA_VISIBLE_DEVICES=0 python tools/generate_smpl_label.py --data_root /data/pose_datasets/scut2 --output_root /data/pose_datasets/scut2/SMPL --cam_param_path ~/human_pose_estimation/rgbd_3d_human_pe/dataset_label_op_code/cam_param/cam_param_2/cam_param_pkg.npz



python tools/process_2d_label.py --seq_dir tmp/seq_mp2
python tools/process_cam_param.py --seq_dir tmp/seq_mp2 --seq_name P015R000A001

还存在的问题：
kptsRepro=np.nan_to_num(kptsRepro)
limblength的过滤判断
躯干的选择
人存在消失、跳变的问题

set PYTHONPATH=./
python apps/vis/vis_server.py --cfg config/vis3d/o3d_scene_smpl_my_mv.yml
python apps/vis/vis_client.py --path tmp/seq_mp/output-track/smpl --smpl

python apps/demo/mvmp.py tmp/seq_mp --out tmp/seq_mp/output --cfg config/exp/my_mvmp1f.yml --annot annots --body smpl --undis --vis_repro --vis_det

python apps/demo/auto_track.py tmp/seq_mp/output tmp/seq_mp/output-track --track3d --body smpl

python apps/demo/smpl_from_keypoints.py tmp/seq_mp --skel tmp/seq_mp/output-track/keypoints3d --out tmp/seq_mp/output-track/smpl --body smpl --verbose --opt smooth_poses 1e1

python tools/generate_smpl_label_mp.py --data_root /data/pose_datasets/scut2_mp --output_smpl_root /data/pose_datasets/scut2_mp/SMPL --cam_param_path ~/human_pose_estimation/rgbd_3d_human_pe/dataset_label_op_code/cam_param/cam_param_2/cam_param_pkg.npz


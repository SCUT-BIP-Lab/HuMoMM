# Benchmark
We provide a benchmark on HuMoMM to test the performance
of popular methods in several related task, including action
recognition, 2D keypoint detection, 3D pose estimation, and human mesh
recovery.


## Action Recognition
We evaluate popular skeleton-based action recognition methods (STGCN and PoseC3D) on HuMoMM. The code can be found in [mmaction2](./mmaction2/).


## 2D Keypoint Detection

We evaluate the effectiveness of two popular lightweight methods on HuMoMM, including Liteweight-OpenPose and LitePose. The code can be found in [liteweight-openpose](./lightweight-openpose/) and [litepose](./litepose/).

## 3D Human Pose Estimation

We evaluate popular 2D-to-3D lifting methods: Videopose3D and GastNet. The code are in [3d_pose_estimation](./3d_pose_estimation/).


## Human Mesh Recovery

As for human mesh recovery, we evaluate the latest method ROMP on HuMoMM. The code are in [ROMP](./ROMP/).

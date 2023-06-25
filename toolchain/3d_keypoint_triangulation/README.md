# RGB-D三维人体姿态估计：数据集采集处理说明文档

# 一、相机标定
相机阵列安装好之后需要标定，标定结束后需要处理标定结果。

## 1、相机顺序
相机顺序如下，所有相机两两标定时以2号相机（最下方）为主相机。
```
    3
0   4   1
    2
```

## 2、相机标定
### （1）同型号相机对[2, 0]，[2, 1]，[2, 3]的标定
针对同型号相机对的标定，使用matlab的双目相机标定工具stereoCameraCalibrator进行标定，三组都选择2号相机作为主相机cam0。

标定结束后，在matlab工作区导出标定结果，并将变量命名为cam{主相机编号2}_{副相机编号0，1，3}， 如cam2_0，使用[save_cam_calibr_result.m](save_cam_calibr_result.m)脚本处理保存相机参数。

在确保matlab的工作路径为`rgbd_3d_human_pe/dataset_label_op_code`的前提下，使用如下指令：
```
save_cam_calibr_result(cam2_0, 0)
```

### （2）异型号相机对[2, 4]的标定
使用ROS下的Kalibr标定工具，将得到相机参数yaml文件命名为cam2_4_param.yaml，将其复制到`./cam_param`目录下。

#### a. 安装kalibr
采用源码编译的方式安装kalibr，安装步骤参考https://zhuanlan.zhihu.com/p/361624995

#### b. 重命名2、4号相机图片
首先需要将图片文件夹以时间戳的方式重命名成如下格式：

- dataset-dir
    - cam0
        - 1385030208726607500.png
        - ...
    - cam1
        - 1385030208726607500.png
        - ...

其中，不同摄像头的图片文件夹必须命名为cam{i}，图片必须采用时间戳命名，如文件名1385030208726607500.png，其中1385030208726607500为19位ns级时间戳。重命名步骤可以使用如下脚本完成：
```
import os 
import time
root_path = "dataset-dir"

FPS=15
time_interval=int((1/15)*10**9)

cam_names=os.listdir(root_path)
if len(cam_names)!=2:
    print('the num of cam is invalid!')
for cam_idx, cam_name in enumerate(cam_names):
    cam_dir=os.path.join(root_path, cam_name)
    filelist = os.listdir(cam_dir)
    time_ns=1639969095631850658
    for file in filelist:
        old_file=os.path.join(cam_dir,file)
        if os.path.isdir(old_file):
            continue
        filename=os.path.splitext(file)[0]
        filetype=os.path.splitext(file)[1]
        new_file=os.path.join(cam_dir, f'{time_ns}'+filetype)
        os.rename(old_file,new_file)
        time_ns+=time_interval
    os.rename(cam_dir, os.path.join(root_path,f'cam{cam_idx}'))
```

#### c. 制作ROS bag
完成重命名后，使用kalibr提供的命令制作ROS bag，命令为：
```
kalibr_bagcreater --folder dataset-dir/. --output-bag output.bag
```

制作完成后，最好检查生成的.bag文件的大小，或用rosbag info output.bag确保生成的bag文件包含了所有图片。

#### d. 相机标定
使用步骤c中得到的bag文件进行标定，命令如下：
```
kalibr_calibrate_cameras --target ../april_6x6_50x50cm.yaml --bag 3_6.bag --models pinhole-radtan pinhole-radtan  --topics /cam0/image_raw /cam1/image_raw
```

参数含义如下：
```
--target 标定板的配置文件，根据标定板尺寸从kalibr官方github下载
--bag 步骤c中得到的bag文件
--models 相机模型
--topics 每个相机的的topic，一般写法为/cam{i}/image_raw
```

标定过程中可能出现某个相机无法确定初始焦距的问题，导致标定结果全为nan。解决方法为先使用以下命令设定环境变量：
```
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
```
再在标定过程中手动输入初始焦距，RealSense D455的初始焦距可以设定为 [fx, fy] = [380, 380]。

#### e. 处理标定结果
标定完成后，会得到.yaml，.pdf和.txt三个文件，将得到相机参数yaml文件命名为cam2_4_param.yaml，将其复制到[./cam_param](./cam_param)目录下。

另外，标注的电脑需要可以显示图形界面，或使用远程x11 forword，否则无法生成report.pdf（包含标定误差）。

## 3、相机标定结果打包
使用[create_cam_param_npz.py](create_cam_param_npz.py)文件打包相机参数。


# 二、数据集处理
数据采集可参考文档《RGBD三维姿态估计项目数据采集及数据标注说明》。

在采集完数据之后，需要使用该工程对数据集进行清洗、检查、重建以及二次标注等处理，具体如下。

### 1、数据集清洗
采集结束后的数据集结构如下：

- custom_dataset_ori/
    - sample_0/
        - 0_0/
            - realsense0_depth/
                - 000.png
                - ...
            - realsense1_rgb/
                - 000.jpg
                - ...
        - 0_1/
        - 1_0/
        - ...
    - sample_1/
    - ...

此时使用[dataset_rename.py](dataset_rename.py)对数据集结构进行重构，新结构如下：

- custom_dataset/
	- RGB_frame/
		- C000P000R000A000/
			- C000P000R000A000RF000.jpg
			- ...
		- C000P000R000A001/
		- ...
	- Depth_frame/
		- C000P000R000A000/
			- C000P000R000A000DF000.png
			- ...
		- ...

### 2、提取关键帧
使用[key_frame_extract.py](key_frame_extract.py)提取关键帧，获得关键帧数据集`custom_dataset_key_frame`。

### 3、基于关键帧数据集进行人工标注
标注方法参考文档《RGBD三维姿态估计项目数据标注说明》。

### 4、检查标签
使用[check_data_label.py](check_data_label.py)分离标签文件并检查是否有漏标注的帧，以及所有关键点是否满足至少标注2个视角以上，用于三维重建，不满足的帧记录下来重新标注，此时数据集结构如下：

- custom_dataset_key_frame/
	- RGB_frame/
		- C000P000R000A000/
			- C000P000R000A000RF000.jpg
			- ...
		- C000P000R000A001/
		- ...
	- Label/
		- C000P000R000A000/
			- C000P000R000A000RF000.json
			- ...
		- C000P000R000A001/
		- ...
	- ( lost_jsons.txt )
	- ( refine_frame.txt )
	
之后，将`lost_jsons.txt`和`refine_frame.txt`（如有出现）中的帧进行重新标注，再次使用[check_data_label.py](check_data_label.py)检查即可。

### 5、三维坐标重建及二维重投影
使用[reconstruct_3d_joints.py](reconstruct_3d_joints.py)计算每个关键点的三维坐标，并重投影回缺失的视角补全二维关键点。

使用[utils/data_labelling.py](utils/data_labelling.py)可以简单可视化重投影效果。

### 6、最小化三维坐标重投影误差
基于colmap的光束平差法优化三维坐标，步骤如下：

（1）使用[generate_colmap_txt.py](generate_colmap_txt.py)生成Colmap_txt文件夹，包含适用于colmap的txt文件。

（2）提前安装好colmap后，运行[colmap_bundleAdjustment.sh](colmap_bundleAdjustment.sh)进行光束平差，手动输入步骤（1）中得到的Colmap_txt文件夹路径。

（3）运行[update_bundle_adjusted_result.py](update_bundle_adjusted_result.py)更新Label_3d路径下的标签文件中的三维坐标以及重投影补充的二维关键点，得到新的标签文件夹Label_3d_ba。

### 7、自动标注
使用光束平差后得到的关键帧二维数据Label_3d_ba作为训练集训练一个二维姿态估计模型，本项目采用Deep-HRNet作为自动标注模型，具体训练方法参考[../Deep-HRNet/README.md](../Deep-HRNet/README.md)。

训练完Deep-HRNet后，使用[../Deep-HRNet/tools/auto_labelling.py](../Deep-HRNet/tools/auto_labelling.py)对custom_dataset中剩余数据进行自动标注，并对自动标注的二维关键点进行第5、6步。最后有效的数据集结构应如下：

- custom_dataset/
	- RGB_frame/
		- C000P000R000A000/
			- C000P000R000A000RF000.jpg
			- ...
		- C000P000R000A001/
		- ...
	- Depth_frame/
	- Label_3d_ba/
		- C000P000R000A000/
			- C000P000R000A000RF000.json
			- ...
		- C000P000R000A001/
		- ...

## 三、生成模型训练数据包
使用[generate_3d_training_gt_npz.py](generate_3d_training_gt_npz.py)生成训练三维姿态估计模型的数据集npz文件。

如果需要同步打包深度图，激活--depmap参数即可。


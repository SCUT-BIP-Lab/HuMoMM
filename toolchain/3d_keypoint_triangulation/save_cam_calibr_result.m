function save_cam_calibr_result(cam_param_mat, vice_cam_idx)
% 用于标定完相机导出参数后，处理相机参数矩阵
% 使用方法：提取标定相机参数为 cam{主相机编号2}_{副相机编号0，1，3}， 如cam2_0
%           然后在命令行窗口输入 save_cam_calibr_result(cam2_0, 0) 即可

    if ~exist('./cam_param','dir') == 1
        mkdir './cam_param';
    end
    
    main_cam_param = cam_param_mat.CameraParameters1.IntrinsicMatrix;
    vice_cam_param = cam_param_mat.CameraParameters2.IntrinsicMatrix;
    vice_cam_rot = cam_param_mat.RotationOfCamera2;
    vice_cam_trans = cam_param_mat.TranslationOfCamera2;
    
    save(['./cam_param/cam2_',num2str(vice_cam_idx),'_ori.mat'], 'cam_param_mat');
    save(['./cam_param/cam2_',num2str(vice_cam_idx),'_param.mat'], 'main_cam_param', 'vice_cam_param', 'vice_cam_rot', 'vice_cam_trans')
end
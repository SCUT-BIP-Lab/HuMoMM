function save_cam_calibr_result(cam_param_mat, vice_cam_idx)
% ���ڱ궨��������������󣬴��������������
% ʹ�÷�������ȡ�궨�������Ϊ cam{��������2}_{��������0��1��3}�� ��cam2_0
%           Ȼ���������д������� save_cam_calibr_result(cam2_0, 0) ����

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
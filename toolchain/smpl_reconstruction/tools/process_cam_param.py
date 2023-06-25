import argparse
import cv2
import numpy as np
import os.path as osp
from easymocap.mytools.camera_utils import write_intri, write_extri


def process_cam_param(sequence_dir, sequence_name):
    # sequence_dir='tmp/seq2'
    # sequence_name='P010R000A011'
    cam_param_pkg = np.load(f'{sequence_dir}/cam_param_pkg.npz', allow_pickle=True)['cam_param_pkg']
    print(cam_param_pkg)

    intri_param={}
    extri_param={}
    for param in cam_param_pkg:
        cam=str(list(param.keys())[0])
        vice_cam=cam.split('_')[1]
        vice_cam=f'C00{vice_cam}{sequence_name}'
        intri_param[vice_cam]={}
        intri_param[vice_cam]['K']=param[cam]['vice_cam_param']
        intri_param[vice_cam]['dist']=np.zeros((1, 5))

        extri_param[vice_cam]={}
        extri_param[vice_cam]['Rvec']=cv2.Rodrigues(param[cam]['vice_cam_rot'])[0]
        extri_param[vice_cam]['R']=param[cam]['vice_cam_rot']
        extri_param[vice_cam]['T']=param[cam]['vice_cam_trans']/1000.0
    
    # 2 is the main cam
    main_cam='2'
    main_cam=f'C00{main_cam}{sequence_name}'
    intri_param[main_cam]={}
    intri_param[main_cam]['K']=param['cam2_4']['main_cam_param']
    intri_param[main_cam]['dist']=np.zeros((1, 5))

    extri_param[main_cam]={}
    extri_param[main_cam]['Rvec']=cv2.Rodrigues(np.eye(3))[0]
    extri_param[main_cam]['R']=np.eye(3)
    extri_param[main_cam]['T']=np.zeros((3, 1))

    # sorted the cam names
    intri_param={key: intri_param[key] for key in sorted(intri_param.keys())}
    extri_param={key: extri_param[key] for key in sorted(extri_param.keys())}

    intri_file=f'{sequence_dir}/intri.yml'
    write_intri(intri_file, intri_param)
    extri_file=f'{sequence_dir}/extri.yml'
    write_extri(extri_file, extri_param)



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', type=str, default='tmp/sequence', help='path to the root of the processed sequence')
    parser.add_argument('--seq_name', type=str, default='P000R000A000', help='name of the process seq, e.g. P*R*A*')
    args = parser.parse_args()
    process_cam_param(args.seq_dir, args.seq_name)





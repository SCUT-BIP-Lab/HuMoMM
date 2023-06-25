import re
from config import args
from collections import OrderedDict
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs

default_mode = args().image_loading_mode


def SCUT_3DHP(base_class=default_mode):
    class SCUT_3DHP(Base_Classes[base_class]):
        def __init__(self, train_flag=True, split='train', **kwargs):
            super(SCUT_3DHP, self).__init__(train_flag, regress_smpl=True)
            self.data_folder = os.path.join(self.data_folder, 'scut_sp')
            annots_file_path = os.path.join(self.data_folder, 'annots.npz')
            self.image_folder = os.path.join(self.data_folder, 'RGB_frame')
            self.scale_range = [1.3, 1.9]
            if os.path.exists(annots_file_path):
                self.annots = np.load(annots_file_path, allow_pickle=True)['annots'][()]
            else:
                self.pack_data(annots_file_path)

            self.file_paths = list(self.annots.keys())

            # split train validation and test
            # train_action_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
            # val_action_id=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # test_action_id=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # if split=='train':
            #     self.file_paths=[file for file in self.file_paths if int(file[file.find('A') + 1:file.find('A') + 4]) in train_action_id]
            # elif split=='val':
            #     self.file_paths=[file for file in self.file_paths if int(file[file.find('A') + 1:file.find('A') + 4]) in val_action_id]
            # elif split=='test':
            #     self.file_paths=[file for file in self.file_paths if int(file[file.find('A') + 1:file.find('A') + 4]) in test_action_id]
            
            train_sub_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
            val_sub_id=[10, 11, 12, 13, 14]
            test_sub_id=[10, 11, 12, 13, 14]
            if split=='train':
                self.file_paths=[file for file in self.file_paths if int(file[file.find('P') + 1:file.find('P') + 4]) in train_sub_id]
            elif split=='val':
                self.file_paths=[file for file in self.file_paths if int(file[file.find('P') + 1:file.find('P') + 4]) in val_sub_id]
            elif split=='test':
                self.file_paths=[file for file in self.file_paths if int(file[file.find('P') + 1:file.find('P') + 4]) in test_sub_id]


            # train_view_id=[0, 1, 2]
            # val_view_id=[3, 4]
            # test_view_id=[3, 4]
            # if split=='train':
            #     self.file_paths=[file for file in self.file_paths if int(file[file.find('C') + 1:file.find('C') + 4]) in train_view_id]
            # elif split=='val':
            #     self.file_paths=[file for file in self.file_paths if int(file[file.find('C') + 1:file.find('C') + 4]) in val_view_id]
            # elif split=='test':
            #     self.file_paths=[file for file in self.file_paths if int(file[file.find('C') + 1:file.find('C') + 4]) in test_view_id]

            self.kp2d_mapper = constants.joint_mapping(constants.SCUT_21, constants.SMPL_ALL_54)
            self.kp3d_mapper = constants.joint_mapping(constants.SCUT_21, constants.SMPL_ALL_54)
            self.root_inds = [constants.SMPL_ALL_54['Pelvis']]
            # self.compress_length = 3
            self.shuffle_mode = args().shuffle_crop_mode
            self.shuffle_ratio = args().shuffle_crop_ratio_3d

            self.sample_num = len(self.file_paths)

            logging.info('Loaded SCUT-3DHP {} set,total {} samples'.format(split, self.__len__()))

        def exclude_subjects(self, file_paths, subjects=['S8']):
            file_path_left = []
            for inds, file_path in enumerate(file_paths):
                subject_id = os.path.basename(file_path).split('_')[0]
                if subject_id not in subjects:
                    file_path_left.append(file_path)
            return file_path_left

        def __len__(self):
            return self.sample_num

        def get_image_info(self, index):
            imgpath = self.file_paths[index % len(self.file_paths)]
            image = cv2.imread(imgpath)[:, :, ::-1]
            R, T = self.annots[imgpath]['extrinsics']
            fx, fy, cx, cy = self.annots[imgpath]['intrinsics']
            camMats = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            kp2ds = self.map_kps(self.annots[imgpath]['kp2d'], maps=self.kp2d_mapper)
            kp3ds = self.map_kps(self.annots[imgpath]['kp3d'], maps=self.kp3d_mapper)[None]
            vis_mask = _check_visible(kp2ds, get_mask=True)
            kp2ds = np.concatenate([kp2ds, vis_mask[:, None]], 1)[None]

            root_trans = kp3ds[:, self.root_inds].mean(1)

            valid_masks = np.array([self._check_kp3d_visible_parts_(kp3d) for kp3d in kp3ds])
            kp3ds -= root_trans[:,None]  # kp3ds 做归一化
            kp3ds[~valid_masks] = -2.


            # read smpl params
            smpl_params = self.annots[imgpath]['smpl_params']
            global_orient = np.array(smpl_params[0]['Rh'])
            poses = np.array(smpl_params[0]['poses'])
            betas = np.array(smpl_params[0]['shapes'])

            poses[0, :3] = global_orient
            params = np.concatenate((poses, betas), axis=-1)

            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,
                        'vmask_2d': np.array([[True, False, True]]), 'vmask_3d': np.array([[True, True, True, True, False, True]]),
                        'kp3ds': kp3ds, 'params': params, 'root_trans': root_trans, 'verts': None,
                        'camMats': camMats, 'img_size': image.shape[:2], 'ds': 'scut_sp'}

            return img_info

        def pack_data(self, annots_file_path):
            print('SCUT_3DHP data annotations is packing...')
            self.annots = {}
            video_names = os.listdir(self.image_folder)
            for i, video_name in enumerate(video_names):
                video_dir = os.path.join(self.image_folder, video_name)
                frame_names = os.listdir(video_dir)
                opt_cam_dir = video_dir.replace('RGB_frame', 'Opt_cam')
                for frame_name in frame_names:
                    img_path = os.path.join(video_dir, frame_name)
                    label_2d_path = img_path.replace('RGB_frame', 'Label').replace('jpg', 'json')
                    label_3d_path = img_path.replace('RGB_frame', 'Label_3d').replace('jpg', 'json')
                    label_smpl_path = img_path.replace('RGB_frame', 'SMPL').replace('jpg', 'json')
                    kp2d = read_2d_keypoints(label_2d_path)
                    kp3d = read_3d_keypoints(label_3d_path)/1000.
                    smpl_param = read_smpl_param(label_smpl_path)
                    # read cam params
                    cam_name = frame_name.split('.')[0].split('RF')[0]
                    cam_params = read_camera(os.path.join(opt_cam_dir, 'intri_opt.yml'),
                                             os.path.join(opt_cam_dir, 'extri_opt.yml'))
                    K, R, T = cam_params[cam_name]['K'], cam_params[cam_name]['R'], cam_params[cam_name]['T']
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    intrinsics = np.array([fx, fy, cx, cy])
                    self.annots[img_path] = {'kp2d': kp2d, 'kp3d': kp3d,
                                             'intrinsics': intrinsics, 'extrinsics': [R, T], 'smpl_params': smpl_param}
                    # frame_info[video_name].append(frame_id)

            np.savez(annots_file_path, annots=self.annots)
            print('SCUT_3DHP data annotations packed')

        def extract_frames(self, frame_info):
            os.makedirs(self.image_folder, exist_ok=True)
            for video_name, frame_ids in frame_info.items():
                video_path = os.path.join(self.data_folder, video_name)
                print('Extracting {}'.format(video_path))
                vidcap = cv2.VideoCapture(video_path)
                frame_id = 0
                while 1:
                    success, image = vidcap.read()
                    if not success:
                        break

                    if frame_id in frame_ids:
                        img_name = self.get_image_name(video_name, frame_id)
                        cv2.imwrite(os.path.join(self.image_folder, img_name), image)
                    frame_id += 1

        def get_image_name(self, video_name, frame_id):
            return video_name.strip('.avi').replace('/imageSequence', '').replace('/', '_') + '_F{:06d}.jpg'.format(frame_id)
    return SCUT_3DHP


def _check_visible(joints, w=2048, h=2048, get_mask=False):
    visibility = True
    # check that all joints are visible
    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
    ok_pts = np.logical_and(x_in, y_in)
    if np.sum(ok_pts) < len(joints):
        visibility = False
    if get_mask:
        return ok_pts
    return visibility


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def read_skeleton_file(file_path, keypoint_type='2d'):
    with open(file_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
        frame_infos = js['shapes']

    body_joint = {}
    for frame_info in frame_infos:
        if 'group_id' not in frame_info.keys():
            person_id = 0
        else:
            person_id = int(frame_info['group_id'] if frame_info['group_id'] else 0)
        single_body_joint = body_joint.setdefault(person_id, {})
        if keypoint_type == '2d':
            single_body_joint[frame_info['label'].zfill(2)] = frame_info['points'][0]
        elif keypoint_type == '3d':
            single_body_joint[frame_info['label'].zfill(2)] = frame_info['points_3d'][0]

    return body_joint


def read_2d_keypoints(anno_path):
    all_skeleton_sequence = read_skeleton_file(anno_path, keypoint_type='2d')
    for person_id in range(len(all_skeleton_sequence)):
        keypoints = []
        visible_keypoints = []
        skeleton_sequence = all_skeleton_sequence[person_id]
        for label in range(21):
            if skeleton_sequence.get(f'{label:02d}') == None:
                keypoint = [-1, -1]  
            elif skeleton_sequence[f'{label:02d}'][0] == 0 and skeleton_sequence[f'{label:02d}'][1] == 0:
                keypoint = [-1, -1]
            else:
                keypoint = [skeleton_sequence[f'{label:02d}'][0],
                            skeleton_sequence[f'{label:02d}'][1]]
            keypoints.append(keypoint)

    keypoints = np.array(keypoints)
    return keypoints


def read_3d_keypoints(anno_path):
    all_skeleton_sequence = read_skeleton_file(anno_path, keypoint_type='3d')
    for person_id in range(len(all_skeleton_sequence)):
        keypoints = []
        skeleton_sequence = all_skeleton_sequence[person_id]
        for label in range(21):
            if skeleton_sequence.get(f'{label:02d}') == None:
                keypoint = [-1, -1, -1]  # 0 for unvisible
            elif skeleton_sequence[f'{label:02d}'][0] == 0 and skeleton_sequence[f'{label:02d}'][1] == 0:
                keypoint = [-1, -1, -1]
            else:
                keypoint = [skeleton_sequence[f'{label:02d}'][0],
                            skeleton_sequence[f'{label:02d}'][1],
                            skeleton_sequence[f'{label:02d}'][2]]
            keypoints.append(keypoint)

    keypoints = np.array(keypoints)
    return keypoints


def read_smpl_param(anno_path):
    with open(anno_path) as f:
        smpl_params = json.load(f)
    return smpl_params


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out + '\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.9f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format(cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        assert Rvec is not None, cam
        # R = extri.read('Rot_{}'.format(cam)) # 直接读入导致R不完美的情况，即不满足外参旋转的cos、sin值等限制
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['Rvec'] = Rvec
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = - Rvec.T @ Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams


if __name__ == '__main__':
    dataset = SCUT_3DHP(base_class=default_mode)(train_flag=True, regress_smpl=True)
    Test_Funcs[default_mode](dataset, with_smpl=True)
    print('Done')

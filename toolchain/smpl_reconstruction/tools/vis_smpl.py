import os
import numpy as np
from tqdm import tqdm
from easymocap.dataset.mv1pmf import MV1PMF
from easymocap.mytools.file_utils import read_json, select_nf
from easymocap.smplmodel.body_param import load_model
from easymocap.mytools import Timer
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


if __name__ == '__main__':
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    parser.add_argument('--opt_cam', action='store_true')
    args = parse_parser(parser)

    cam='C003P000R000A000'
    dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, verbose=args.verbose)
    
    dataset.cameras[cam]['R']=np.eye(3)
    dataset.cameras[cam]['Rvec']=np.array([[0.0, 0.0, 0.0]]).T
    dataset.cameras[cam]['T']=np.array([[0.0, 0.0, 0.0]]).T
    dataset.cameras[cam]['RT']=np.hstack((dataset.cameras[cam]['R'], dataset.cameras[cam]['T']))
    dataset.cameras[cam]['center']=np.array([[0.,]])
    dataset.cameras[cam]['P']=dataset.cameras[cam]['K']@dataset.cameras[cam]['RT']

    
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    start, end=0, 75
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]

        param=read_json(os.path.join('/data/pose_datasets/scut', 'SMPL', f'{cam}', f'{cam}RF{nf:03d}.json'))[0]
        # param=read_json(os.path.join(args.path, 'output', 'smpl', f'{nf:06d}.json'))[0]
        vertices = body_model(return_verts=True, return_tensor=False, **param)
        dataset.vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)
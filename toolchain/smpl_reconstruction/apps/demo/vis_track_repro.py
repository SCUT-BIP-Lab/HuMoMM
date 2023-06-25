import os
from easymocap.dataset import CONFIG
from easymocap.dataset import MVMPMF
from tqdm import tqdm
from easymocap.config.mvmp1f import Config
from easymocap.mytools.reader import read_keypoints3d_all
from easymocap.mytools import load_parser, parse_parser

if __name__ =='__main__':
    parser = load_parser()
    parser.add_argument('--vis_match', action='store_true')
    args = parse_parser(parser)
    from easymocap.config.mvmp1f import Config
    cfg = Config.load(args.cfg, args.cfg_opts)


    vis_cfg = Config.load(args.cfg, args.cfg_opts)
    dataset = MVMPMF(args.path, cams=args.sub, annot_root=args.annot,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, out=args.out, filter2d=vis_cfg.dataset)
    results3d, filenames = read_keypoints3d_all(os.path.join(args.out, 'keypoints3d'))

    for nf, skelname in enumerate(tqdm(filenames, desc='visulize repro')):
        images, annots = dataset[nf]
        dataset.vis_track_repro(images, results3d,nf,)
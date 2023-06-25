import os
import argparse
import sys
import shutil
import cv2
import json
import _init_paths
from subprocess import Popen, PIPE, STDOUT
from tools.prepare_scut_rgbd import prepare_label
from tools.prepare_scut_val import gen_coco_style_json
from inference_api import PoseEstimation
from utils.utils import img_arr_to_b64


def check_dir(dir):
    if os.path.exists(dir) and len(os.listdir(dir)):
        print(f'{dir}')
        ans = input('exists and is not empty, delete it and continue?(y/n):')
        if ans == 'y':
            shutil.rmtree(dir)
        else:
            raise Exception(f'{dir} exists and is not empty.')
    os.makedirs(dir, exist_ok=True)

    return


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def gen_pose_annotation(img, img_name, pose):
    annotation = {
        "version": "0.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": img_name,
        'imageData': img_arr_to_b64(img).decode('utf-8')
    }
    for i, keypoint in enumerate(pose):  # only one person
        annotation['shapes'].append({
            "label": f'{i:02d}',
            "points": [[float(keypoint[0]), float(keypoint[1])]],
            "group_id": None,
            "shape_type": "point",
            "flags": {},
            "vis": '1'
        })
    return annotation


def copy_and_overwrite(from_path, to_path, copy_function=shutil.copy2):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path, copy_function=copy_function)


def exe_command(command):
    """
    执行 shell 命令并实时打印输出
    :param command: shell 命令
    :return: process, exitcode
    """
    print(command)
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip())
    exitcode = process.wait()
    return process, exitcode


def main(args):
    label_dir = args.manual_label_path
    image_dir = os.path.join(args.data_root, 'RGB_frame')

    print(f'=> Start auto labeling, it contains {args.loop_steps} steps.')
    # calc the num of labeling img in every step
    video_names = sorted(os.listdir(label_dir))
    img_labeled = sorted(os.listdir(os.path.join(label_dir, video_names[0])))
    img_labeled = set([img_name[:-5] for img_name in img_labeled])  # remove .json
    img_all = sorted(os.listdir(os.path.join(image_dir, video_names[0])))
    img_all = set([img_name[:-4] for img_name in img_all])  # remove .jpg
    img_unlabeled = img_all - img_labeled
    num_unlabeled_subset = len(img_unlabeled) // args.loop_steps
    print(f'=> Use {num_unlabeled_subset} imgs to auto labeling in every steps')

    for step in range(1, args.loop_steps + 1):
        print(f'=> Step {step}: start auto labeling')
        if step > 1:
            label_dir = os.path.join(args.tmp_dir, f'Label{step-1}')
        new_label_dir = os.path.join(args.tmp_dir, f'Label{step}')

        copy_and_overwrite(label_dir, new_label_dir)

        # prepare label file for hrnet training
        out_dir = os.path.join(args.tmp_dir, f'2d_label{step}')
        prepare_label(label_dir, out_dir, part='train')
        prepare_label(label_dir, out_dir, part='val')
        gen_coco_style_json(label_dir, out_dir)
        print(f'=> Step {step}: finish generate label for training pose model')

        # train hrnet using the subset label and last model
        if step == 1:
            exe_command(
                f'python tools/train.py --cfg  experiments/scut_rgbd/hrnet/w48_256x192_adam_lr1e-3.yaml '
                f'DATASET.LABEL_DIR {args.tmp_dir}/2d_label{step} '
                f'DATASET.ROOT {args.data_root}')
        else:
            # use last model to initialize the model
            last_model_path = os.path.join(args.tmp_dir, 'auto_labeling_models', f'model{step-1}.pth')
            exe_command(
                f'python tools/train.py --cfg  experiments/scut_rgbd/hrnet/w48_256x192_adam_lr1e-3.yaml '
                f'DATASET.LABEL_DIR {args.tmp_dir}/2d_label{step} '
                f'MODEL.PRETRAINED {last_model_path} '
                f'DATASET.ROOT {args.data_root}')
        print(f'=> Step {step}: finish training pose model')

        model_path = os.path.join(args.tmp_dir, 'auto_labeling_models', f'model{step}.pth')
        make_dir(os.path.dirname(model_path))
        shutil.copy('output/scut_rgbd/pose_hrnet/w48_256x192_adam_lr1e-3/model_best.pth', model_path)

        # init the model using the model trained in subset
        print(f'=> Step {step}: start predicting label')
        pose_model = PoseEstimation(model_file=model_path)
        video_names = sorted(os.listdir(label_dir))
        for video_name in video_names:
            img_labeled = sorted(os.listdir(os.path.join(label_dir, video_name)))
            img_labeled = set([img_name[:-5] for img_name in img_labeled])  # remove .json
            img_all = sorted(os.listdir(os.path.join(image_dir, video_name)))
            img_all = set([img_name[:-4] for img_name in img_all])  # remove .jpg
            img_unlabeled = img_all - img_labeled

            if num_unlabeled_subset < len(img_unlabeled) < 2 * num_unlabeled_subset:
                img_unlabeled_subset = img_unlabeled
            else:
                img_unlabeled_subset = list(img_unlabeled)[:num_unlabeled_subset]

            for img_name in img_unlabeled_subset:
                img = cv2.imread(os.path.join(image_dir, video_name, f'{img_name}.jpg'))

                pred_joints, has_person = pose_model.inference(img, img_name)

                if not has_person:
                    with open('tmp/det_fail_imgs.txt', 'a') as f:
                        f.write(f'{img_name}\n')

                annotation = gen_pose_annotation(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f'{img_name}.jpg', pred_joints)
                new_label_path = os.path.join(new_label_dir, video_name, f'{img_name}.json')
                with open(new_label_path, 'w') as f:
                    json.dump(annotation, f, indent=4)
        print(f'=> Save predict labels to {new_label_dir}')
        print(f'=> Step {step}: finish auto labeling')
    print(f'=> Copy final label to {args.auto_label_path}')
    copy_and_overwrite(os.path.join(args.tmp_dir, f'Label{step}'), args.auto_label_path)
    print(f'=> Finish all steps of auto labeling')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../datasets/custom_dataset',
                        help='path to the root of the origin dataset.')
    parser.add_argument('--manual_label_path', type=str,
                        default='../datasets/custom_dataset_key_frame/Label', help='path to the manual labelling file.')
    parser.add_argument('--auto_label_path', type=str, default='../datasets/custom_dataset/Label',
                        help='path to the automatic labelling file.')
    parser.add_argument('--tmp_dir', type=str, default='./tmp', help='path to save the intermediate files.')
    parser.add_argument('--loop_steps', type=int, default=1, help='loop steps to the auto labeling')
    args = parser.parse_args()
    main(args)

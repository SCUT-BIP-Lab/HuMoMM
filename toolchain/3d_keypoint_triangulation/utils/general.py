import os
import shutil
from termcolor import cprint
import cv2


def check_dir(dir):
    if os.path.exists(dir) and len(os.listdir(dir)):
        cprint(f'{dir}', color='yellow', attrs=['bold'], end=' ')
        ans = input('exists and is not empty, delete it and continue?(y/n):')
        if ans == 'y':
            shutil.rmtree(dir)
        else:
            raise Exception(f'{dir} exists and is not empty.')
    os.makedirs(dir, exist_ok=True)

    return


def ske_vis(joints_2d, img):
    '''
    Skeleton visualization.

    :param joints_2d: input 2d joints of n*2
    :param img: src img.
    :return: img with skeleton.
    '''
    skeleton_parents = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    if len(joints_2d) != len(skeleton_parents):
        print('Joint lost, skeleton visualization failed.')
        return img

    for i in range(len(joints_2d)):
        cv2.circle(img, (int(joints_2d[i][0]), int(joints_2d[i][1])), 1, (0, 255, 0), 4)
        if skeleton_parents[i] != -1:
            cv2.line(img,
                     (int(joints_2d[i][0]), int(joints_2d[i][1])),
                     (int(joints_2d[skeleton_parents[i]][0]), int(joints_2d[skeleton_parents[i]][1])),
                     (0, 255, 0), 1)
    return img

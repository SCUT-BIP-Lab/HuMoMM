import os
import cv2

if __name__ == '__main__':
    video_path = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_rgb'
    out_img_path = '/home/data/human_pose_estimation/ntu_rgbd/nturgb+d_rgb_imgs'
    videos = os.listdir(video_path)
    for i, video_name in enumerate(videos):
        print(f'[{i+1}/{len(videos)}] processing video: {video_name}')
        file_name = video_name.split('.')[0]
        folder_name = os.path.join(out_img_path, file_name)
        os.makedirs(folder_name, exist_ok=True)
        cap = cv2.VideoCapture(os.path.join(video_path, video_name))
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(folder_name, f'{cnt:03d}.jpg'), frame)
                cnt += 1
            else:
                break
        cap.release()

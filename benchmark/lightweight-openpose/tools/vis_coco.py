import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils

dataset_dir = '../datasets/coco2017/val2017/'
json_file = './coco_data/val_subset.json'

coco = COCO(json_file)
catIds = coco.getCatIds(catNms=['person']) # catIds=1 表示人这一类
imgIds = coco.getImgIds(catIds=catIds ) # 图片id，许多值
for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    I = io.imread(dataset_dir + img['file_name'])
    plt.axis('off')
    plt.figure()
    plt.imshow(I) #绘制图像，显示交给plt.show()处理
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show() #显示图像
    plt.savefig('tmp/vis_coco.jpg')
    print(f'visualize figure {i} finished.')




# with open(json_file) as anno_:
#     annotations = json.load(anno_)

# def apply_mask(image, segmentation):
#     alpha = 0.5
#     color = (0, 0.6, 0.6)
#     threshold = 0.5
#     mask = maskUtils.decode(segmentation) # 分割解码
#     mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)
#     for c in range(3): # 3个通道
#         # mask=1执行前一个，否则后一个
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image

# results = []
# for i in range(len(annotations)):
#     annotation = annotations[i]
#     image_id = annotation['image_id']
#     # 包含size:图片高度宽度  counts:压缩后的mask  通过mask = maskUtils.decode(encoded_mask)解码，得到mask,需要导入from pycocotools import mask as maskUtils
#     segmentation = annotation['segmentation'] 
#     full_path = os.path.join(dataset_dir, str(image_id).zfill(12) + '.jpg')
#     image = cv2.imread(full_path)
#     mask_image = apply_mask(image, segmentation)
#     cv2.imshow('demo', mask_image)
#     cv2.waitKey(5000)




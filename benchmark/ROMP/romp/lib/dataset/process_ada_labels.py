import math
import os
import cv2
import numpy as np


def process_ada_labels(img, kpts2d):
    output_res = 128
    num_joints = 54
    num_classes = 1
    max_objs = 64
    num_objs = min(len(kpts2d), max_objs)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    trans_input = get_affine_transform(c, s, rot, [height, width])
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])

    hm = np.zeros((num_classes, output_res, output_res), dtype=np.float32)  # center heatmap

    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res),
                         dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res),
                              dtype=np.float32)
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    kps = np.zeros((max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)
    kps_mask = np.zeros((max_objs, num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((max_objs * num_joints), dtype=np.int64)

    # draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
    # 								draw_umich_gaussian
    draw_gaussian = draw_msra_gaussian

    gt_det = []
    for k in range(num_objs):
        # ann = anns[k]
        # bbox = self._coco_box_to_bbox(ann['bbox'])
        # cls_id = int(ann['category_id']) - 1
        cls_id = 0
        pts = np.array(kpts2d[k], np.float32).reshape(num_joints, 3)
        # pts=np.hstack((pts, np.ones((num_joints, 1))))
        # print(bbox)
        # if flipped:
        # 	bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        # 	pts[:, 0] = width - pts[:, 0] - 1
        # 	for e in self.flip_idx:
        # 		pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

        valid_kps_num = 0
        pts_tmp = np.zeros((1, 2))

        #######################################################################################################################
        for ind_ in range(num_joints):
            if pts[ind_, 2] > 0:
                pts[ind_, :2] = affine_transform(pts[ind_, :2], trans_output_rot)
                if pts[ind_, 0] >= 0 and pts[ind_, 0] < output_res and \
                        pts[ind_, 1] >= 0 and pts[ind_, 1] < output_res:
                    valid_kps_num += 1
                    pts_tmp = np.concatenate((pts_tmp, pts[ind_, :2][None]), axis=0)
                else:
                    pts[ind_, 2] = 0
            # import pudb; pudb.set_trace()
        if valid_kps_num == 0:
            continue
        else:
            ct = np.array(
                [pts_tmp[:, 0].sum() / valid_kps_num, pts_tmp[:, 1].sum() / valid_kps_num], dtype=np.float32)  # the average of all valid keypoints

        ####################################### generate the pseudo-box according to the visiable keypoints#####################
        pts_tmp_wo_zero = pts_tmp[1:, :]
        assert len(pts_tmp_wo_zero) == valid_kps_num
        tl = np.min(pts_tmp_wo_zero, axis=0)
        rd = np.max(pts_tmp_wo_zero, axis=0)

        h, w = rd[1] - tl[1], rd[0] - tl[0]
        ###################################################################################################################

        ct_int = ct.astype(np.int32)

        if ct_int[0] >= 0 and ct_int[0] < output_res and \
                ct_int[1] >= 0 and ct_int[1] < output_res and (h > 0 and w > 0) and valid_kps_num > 0:

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            # radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
            radius = max(0, int(radius))

            wh[k] = 1. * w / output_res, 1. * h / output_res  # normalize to (0,1)

            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))

            # hp_radius = self.opt.hm_gauss \
            #                         if self.opt.mse_loss else max(0, int(hp_radius))
            hp_radius = max(0, int(hp_radius))
            ind[k] = ct_int[1] * output_res + ct_int[0]
            reg_mask[k] = 1

            for j in range(num_joints):
                if pts[j, 2] > 0:
                    kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                    kps_mask[k, j * 2: j * 2 + 2] = 1
                    pt_int = pts[j, :2].astype(np.int32)
                    hp_offset[k * num_joints + j] = pts[j, :2] - pt_int  # kpt的小数偏差
                    hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                    hp_mask[k * num_joints + j] = 1
                    # if self.opt.dense_hp:
                    #     # must be before draw center hm gaussian
                    #     draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                    #                    pts[j, :2] - ct_int, radius, is_offset=True)  # 这个是offset的heatmap
                    #     draw_gaussian(dense_kps_mask[j], ct_int, radius)
                    draw_gaussian(hm_hp[j], pt_int, hp_radius)  # joint 的heatmap,辅助用的

            draw_gaussian(hm[cls_id], ct_int, radius)
            gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                           ct[0] + w / 2, ct[1] + h / 2, 1] +
                          pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
    # import pudb;pudb.set_trace()
    # cv2.imwrite('/data/yabo.xiao/coco_vis_center/'+file_name,inp_)

    ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'hps': kps, 'hps_mask': kps_mask}
    ret.update({'hm_hp': hm_hp})
    return ret  # hm: centermap hps: joints offset, maxobj*17*2 # hm_hp: joint的heatmap,辅助训练，dense_hps：在centermap生成heatmap的offset,但没用到


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans




def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2 

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2 
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)
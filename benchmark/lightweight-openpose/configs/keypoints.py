from addict import Dict


coco_keypoint_info = Dict(
    KEYPOINT_NAMES=['nose', 'neck',
                    'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                    'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                    'r_eye', 'l_eye',
                    'r_ear', 'l_ear'],
    BODY_PARTS_KPT_IDS=[[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                        [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]],
    BODY_PARTS_PAF_IDS=([0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17],
                        [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37]),
    RIGHT=[2, 3, 4, 8, 9, 10, 14, 16],
    LEFT=[5, 6, 7, 11, 12, 13, 15, 17],
    SIGMAS=[.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
    NUM_JOINTS=18,
    NUM_PAFS=38,
)

# ntu rgbd
ntu_keypoint_info = Dict(
    KEYPOINT_NAMES=["base of the spine", "middle of the spine", "neck", "head", "left shoulder", "left elbow", "left wrist",
                    "left hand", "right shoulder", "right elbow", "right wrist ", "right hand", "left hip", "left knee",
                    "left ankle", "left foot", "right hip", "right knee", "right ankle", "right foot",
                    "spine", "tip of the left hand", "left thumb", "tip of the right hand", "right thumb"],

    BODY_PARTS_KPT_IDS=[(20, 8), (20, 4), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), (4, 5), (5, 6), (6, 7),
                        (7, 21), (7, 22), (20, 1), (1, 0), (0, 16), (0, 12), (16, 17), (17, 18), (18, 19),
                        (12, 13), (13, 14), (14, 15), (20, 2), (2, 3)],
    BODY_PARTS_PAF_IDS=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19),
                        (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37),
                        (38, 39), (40, 41), (42, 43), (44, 45), (46, 47)],
    RIGHT=[8, 9, 10, 11, 23, 24, 16, 17, 18, 19],
    LEFT=[4, 5, 6, 7, 21, 22, 12, 13, 14, 15],
    SIGMAS=[1.07, 1.07, .72, .72, .79, .72, .62, .35, .79, .72, .62, .35,
            1.07, .87, .89, .89, 1.07, .87, .89, .89, 1.07, .35, .35, .35, .35, ],
    NUM_JOINTS=25,
    NUM_PAFS=48,
)


# scut rgbd
scut_keypoint_info = Dict(
    KEYPOINT_NAMES=["hip", "spine", "neck", "chin", "head", "left shoulder", "left elbow", "left wrist",
                    "left hand", "right shoulder", "right elbow", "right wrist ", "right hand", "left hip",
                    "left knee", "left ankle", "left foot", "right hip", "right knee", "right ankle", "right foot"],

    BODY_PARTS_KPT_IDS=[(2, 9), (2, 5), (9, 10), (10, 11), (11, 12), (5, 6), (6, 7), (7, 8), (2, 1), (1, 0),
                        (0, 17), (0, 13), (17, 18), (18, 19), (19, 20), (13, 14), (14, 15), (15, 16), (2, 3), (3, 4)],

    BODY_PARTS_PAF_IDS=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19),
                        (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37), (38, 39)],

    RIGHT=[9, 10, 11, 12, 17, 18, 19, 20],
    LEFT=[5, 6, 7, 8, 13, 14, 15, 16],

    SIGMAS=[1.07, 1.07, 1.07, .72, .72, .79, .72, .62, .35, .79, .72, .62, .35, 1.07, .87, .89, .89, 1.07, .87, .89, .89],
    NUM_JOINTS=21,
    NUM_PAFS=40,
)


def get_keypoint_info(dataset_name):
    infos = {
        'coco': coco_keypoint_info,
        'ntu': ntu_keypoint_info,
        'scut': scut_keypoint_info
    }
    return infos[dataset_name]

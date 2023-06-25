# DATASETS

More infomation: 
https://github.com/facebookresearch/VideoPose3D/issues/39.
https://github.com/facebookresearch/VideoPose3D/issues/45

The model predicts the pose in camera space, meaning that XY correspond to the screen coordinates and Z is the depth. This is the only unambiguous way a pose can be predicted without knowing the camera position and rotation with respect to the world reference frame.
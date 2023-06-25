# Human Pose Estimation
My projects for all experiments of 3d human pose estimation.

## Requirement
- pytorch
- hydra-core
- opencv-python
- matplotlib
- accelerate

## Usage
### Train
修改config/conf.yaml的配置，然后：
```bash
python run.py
```

### Predict
在`predict.py`中修改`config_path`为先前训练输出.hydra路径，然后：
```bash
python predict.py
```
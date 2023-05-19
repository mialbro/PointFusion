# PointFusion
 - Unofficial implementation of PointFusion Neural Network: https://arxiv.org/abs/1711.10871
 - Regresses spatial offsets from object points and its 3D bounding box

## Dependencies:
* docker 
* nvidia-container-runtime
* docker-compose >= v1.28

## Installation (Docker)
```
git clone https://github.com/mialbro/PointFusion.git
cd PointFusion
docker-compose build pointfusion
```

## LINMEMOD
```
cd PointFusion/datasets
```
* Download https://drive.google.com/file/d/11YzXNEyeQY7DcNZZZ6SVn732_EqMSopv/view?usp=sharing

## Training
```python
import pointfusion

trainer = pointfusion.Trainer()
trainer.model = pointfusion.models.PointFusion()
trainer.dataset = pointfusion.datasets.LINEMOD()
trainer.fit()
```

## Inference
```python
import pointfusion

inference = pointfusion.Inference()
inference.camera = pointfusion.D455()

for (color, depth, point_cloud) in inference.camera:
    corners = pf.predict(color, depth)

```

## Devices
* Currently support IntelRealsense cameras (D4**)
* To expand library to support additional cameras create child class from camera.Camera() as done with d455.D455()

```python
class ZED(pointfusion.Camera):
    def __init__(self, width=1280, height=720, fps=30):
        ...
        
    def next(self):
    ...
    return color, depth, point_cloud
```

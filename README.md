# PointFusion
 - Unofficial implementation of PointFusion Neural Network: https://arxiv.org/abs/1711.10871
 - Regresses spatial offsets from object points and its 3D bounding box

## Dependencies:
* CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)
* docker (https://docs.docker.com/engine/install/ubuntu/)
* docker-compose >= v1.28 (https://docs.docker.com/compose/install/linux/)
* nvidia-container-toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* nvidia-docker2 (https://www.ibm.com/docs/en/masv-and-l/maximo-vi/continuous-delivery?topic=planning-installing-docker-nvidia-docker2)

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

model_name = pointfusion.ModelName.DenseFusion
modalities = [ pointfusion.Modality.RGB, pointfusion.Modality.POINT_CLOUD ]

model = pointfusion.DenseFusion(point_count=400, modalities=modalities)
dataset = pointfusion.LINEMOD(point_count=400, model_name=model_name)
loss_fcn = pointfusion.loss.dense_fusion

trainer = pointfusion.Trainer()
trainer.batch_size = 5
trainer.lr = 0.01
trainer.weight_decay = 0.001
trainer.model = model
trainer.loss_fcn = loss_fcn
trainer.dataset = dataset

trainer.fit()
```

## Inference
```python
import pointfusion

camera = pointfusion.D455()
inference = pointfusion.Inference()

for (color, depth, point_cloud) in camera:
    corners = pf(color, depth)
```

## Devices
* Currently support IntelRealsense cameras (D4**)

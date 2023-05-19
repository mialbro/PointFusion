# PointFusion
 - Unofficial implementation of PointFusion Neural Network: https://arxiv.org/abs/1711.10871
 - Regresses spatial offsets from object points and its 3D bounding box

## Installation (Docker)
git clone https://github.com/mialbro/PointFusion.git
cd PointFusion
docker-compose build pointfusion

## Usage (Training)
```
import pointfusion

trainer = pointfusion.Trainer()
trainer.model = pointfusion.models.PointFusion()
trainer.dataset = pointfusion.datasets.LINEMOD()
trainer.fit()
```

## Usage (Inference)
```
import pointfusion

inference = pointfusion.Inference()
inference.camera = pointfusion.D455()

for (color, depth, point_cloud) in inference.camera:
    corners = pf.predict(color, depth)

```

## Datasets
* LINEMOD (https://drive.google.com/file/d/11YzXNEyeQY7DcNZZZ6SVn732_EqMSopv/view?usp=sharing)

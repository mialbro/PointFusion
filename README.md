# PointFusion
 - Unofficial implementation of PointFusion Neural Network: https://arxiv.org/abs/1711.10871
 - Regresses spatial offsets from object points and its 3D bounding box

## Datasets
* LINEMOD (https://drive.google.com/file/d/11YzXNEyeQY7DcNZZZ6SVn732_EqMSopv/view?usp=sharing)
```
import pointfusion

pf = pointfusion.PointFusion()

pf.dataset = LINEMOD('/home/linemod/LINEMOD')
pf.mode = pointfusion.Mode.TRAIN

pf.train()
```

```
import pointfusion

pf = pointfusion.PointFusion()

for (frame, cloud) in zip(frames, clouds):
  
```

```
conda deactivate
conda remove -n pointfusion --all

conda env create -f environment.yml
```
# PointFusion
 - Unofficial implementation of PointFusion Neural Network: https://arxiv.org/abs/1711.10871
 - Regresses spatial offsets from object points and its 3D bounding box

# Steps:
 - Extract preprocessed linemod dataset into the "dataset" folder
 - Run train script
   - Model checkpoints will be placed into "models" folder
 
### To-do:
- [x] Linemod dataset (https://drive.google.com/file/d/18C_MKYG3a01bgBGo15-Syj1kAlix3rGO/view?usp=sharing)
- [ ] Objectron dataset script
- [x] Custom PoinNet Network
- [x] PointFusion Network
- [x] ResNet50 Feature Extraction Network
- [x] Unsupervised Loss Function
- [x] Train

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

conda deactivate
conda remove -n pointfusion --all

conda env create -f environment.yml
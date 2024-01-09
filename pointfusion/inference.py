import open3d as o3d

import cv2
import torch
import torchvision
import numpy as np

import pointfusion
from pointfusion.d455 import D455
from pointfusion.models import GlobalFusion, DenseFusion

class Inference:
    def __init__(self, model_name: pointfusion.ModelName, filepath: str) -> None:   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
        self.frcn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        self.frcn.eval()
        self.frcn.to(self.device)

        if model_name is pointfusion.ModelName.DenseFusion:
            self.model = DenseFusion()
        elif model_name is pointfusion.ModelName.GlobalFusion:
            self.model = GlobalFusion()
        
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, color_image, depth_image):
        tensor_image = torch.from_numpy(np.transpose(color_image.copy() / 255.0, (2, 0, 1))).float().unsqueeze(0)
        tensor_image = tensor_image.to(self.device)
        outputs = self.frcn(tensor_image)

        # get the predicted boxes, labels, and scores from the output
        pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].detach().cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()

        scores = []
        corners = []
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > 0.9:
                box = box.astype(np.int32)
                # crop depth image and back project
                depth = np.zeros(depth_image.shape, dtype=depth_image.dtype)
                depth[box[1]:box[3], box[0]:box[2]] = depth_image[box[1]:box[3], box[0]:box[2]]
                point_cloud, _ = self.camera.back_project(depth, color_image)
                if point_cloud.shape[0] >= 400:
                    point_cloud = torch.from_numpy(np.transpose(point_cloud)).float().unsqueeze(0)
                    point_cloud = point_cloud.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    curr_scores, curr_corners = self._model(tensor_image, point_cloud)
                    scores.append(curr_scores)
                    curr_corners.append(curr_corners)
        return scores, corners

if __name__ == '__main__':
    camera = D455()
    inference = Inference('../weights/pointfusion_0.pt')

    for (color, depth, point_cloud) in camera:
        scores, corners = inference.predict(color, depth)

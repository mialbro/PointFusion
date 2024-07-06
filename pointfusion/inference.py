import argparse
from typing import Tuple, Optional

import open3d as o3d
import cv2
import torch
import torchvision
import numpy as np
import numpy.typing as npt

from pointfusion.camera import Camera
from pointfusion.enums import FusionMethod
from pointfusion.models import GlobalFusion, DenseFusion

class Inference:
    """Inference wrapper for pointfusion
    Args:
        fusion_method (FusionMethod): DENSE or GLOBAL
        filepath (str): Path to weights
    """
    def __init__(self, filepath: str, model_name: FusionMethod) -> None:
        if model_name is FusionMethod.DENSE:
            self.model = DenseFusion()
        elif model_name is FusionMethod.GLOBAL:
            self.model = GlobalFusion()
        # Load the pointfusion model
        #self.model.load_state_dict(torch.load(filepath, self.device))
        #self.model.to(self.device)
        #self.model.eval()
        self.frcn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        # Set faster rcnn to evaluation mode
        self.frcn.eval()
        # Move model to device (cuda or cpu)
        self.frcn.to(self.device)

    @property
    def device(self) -> torch.device:
        """CUDA / CPU device property

        Returns:
            torch.device: Current activated device
        """
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(
            self,
            image: Optional[npt.NDArray[np.uint8]] = None,
            depth: Optional[npt.NDArray[np.float32]] = None
        ) -> Tuple[float, float]:
        # Convert image to tensor
        tensor_image = torch.from_numpy(
            np.transpose(image.copy() / 255.0, (2, 0, 1))
        ).float().unsqueeze(0)
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
                depth = np.zeros(depth.shape, dtype=depth.dtype)
                depth[box[1]:box[3], box[0]:box[2]] = depth[box[1]:box[3], box[0]:box[2]]
                point_cloud, _ = self.camera.back_project(depth, image)
                if point_cloud.shape[0] >= 400:
                    point_cloud = torch.from_numpy(np.transpose(point_cloud)).float().unsqueeze(0)
                    point_cloud = point_cloud.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    curr_scores, curr_corners = self._model(tensor_image, point_cloud)
                    scores.append(curr_scores)
                    curr_corners.append(curr_corners)
        return scores, corners
    
def main() -> None:
    """Run training loop"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default='../weights/weights.pt'
    )
    parser.add_argument(
        '--fusion_method',
        type=FusionMethod,
        choices=list(FusionMethod),
        default=FusionMethod.DENSE
    )
    args = parser.parse_args()
    # Load model
    inference = Inference(args.weights, args.fusion_method)
    # Run camera
    camera = Camera()
    for image, depth, _ in camera:
        corners = inference(image, depth)

if __name__ == '__main__':
    main()

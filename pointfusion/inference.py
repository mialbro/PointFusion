import argparse
from typing import Tuple, Optional
import sys

import open3d as o3d
import cv2
import torch
import torchvision
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

from pointfusion.camera import Camera
from pointfusion.d455 import D455
from pointfusion.enums import FusionMethod
from pointfusion.models import GlobalFusion, DenseFusion

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class Inference:
    """Inference wrapper for pointfusion
    Args:
        fusion_method (FusionMethod): DENSE or GLOBAL
        filepath (str): Path to weights
    """
    def __init__(self) -> None:
        self.camera = None
        self.model = None
        self.frcn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
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
            color_image: Optional[npt.NDArray[np.uint8]] = None,
            depth_image: Optional[npt.NDArray[np.float32]] = None
        ) -> Tuple[float, float]:
        # Convert image to tensor
        color_image = np.transpose(color_image.copy() / 255.0, (2, 0, 1))
        tensor_color_image = torch.from_numpy(color_image).float().unsqueeze(0).to(self.device)
        # Convert depth image to tensor
        tensor_depth_image = torch.from_numpy(depth_image).float()
        frcn_output = self.frcn(tensor_color_image)
        # get the predicted boxes, labels, and scores from the output
        bboxes = []
        scores = []
        labels = []
        for i, score in enumerate(frcn_output[0]['scores']):
            if score >= 0.9:
                bboxes.append(frcn_output[0]['boxes'][i])
                labels.append(frcn_output[0]['labels'][i])
                scores.append(frcn_output[0]['scores'][i])
        objects = {'images': [], 'point_clouds': []}
        if len(scores) > 0:
            for box in bboxes:
                x1, y1, x2, y2 = box
                curr_image = tensor_color_image[0][:, int(y1):int(y2), int(x1):int(x2)].cpu().numpy()
                curr_image = curr_image.transpose(1, 2, 0)
                curr_depth = tensor_depth_image[int(y1):int(y2), int(x1):int(x2)].numpy()
                curr_point_cloud = self.camera.back_project(curr_depth, curr_image)
                # Convert image and point cloud to tensors
                tensor_curr_image = torch.from_numpy(curr_image.transpose(2, 0, 1)).to(self.device)
                tensor_curr_point_cloud = torch.from_numpy(curr_point_cloud[0].transpose(1, 0)).to(self.device)
                objects['images'].append(curr_image)
                objects['point_clouds'].append(curr_point_cloud)
                import pdb; pdb.set_trace()
            result = self.model(image=objects['image'][0], point_cloud=objects['point_cloud'][0])
            import pdb; pdb.set_trace()
        return None, None

        result = result.permute(1, 2, 0).cpu().numpy()
        result = (result * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            sys.exit()
        return None, None

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
    model = Inference(args.weights, args.fusion_method)
    # Run camera
    camera = D455()
    for image, depth, _ in camera:
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()

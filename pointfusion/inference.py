import cv2
import torch
import torchvision
import numpy as np
import pointfusion

class Inference:
    def __init__(self):
        self._dataset = None
        self._trainer = None
        self._camera = None

        self._frcn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        self._frcn.eval()
        self._frcn.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self._model = pointfusion.PointFusion()
        self._model.load_state_dict(torch.load('../weights/pointfusion.pth'))
        self._model.eval()

    @property
    def camera(self):
        return self._camera
    
    @camera.setter
    def camera(self, camera):
        self._camera = camera

    def predict(self, color_image, depth_image):
        tensor_image = torch.from_numpy(np.transpose(color_image.copy() / 255.0, (2, 0, 1))).float().unsqueeze(0)
        tensor_image = tensor_image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        outputs = self._frcn(tensor_image)

        # get the predicted boxes, labels, and scores from the output
        pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].detach().cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > 0.9:
                box = box.astype(np.int32)
                # crop depth image and back project
                depth = np.zeros(depth_image.shape, dtype=depth_image.dtype)
                depth[box[1]:box[3], box[0]:box[2]] = depth_image[box[1]:box[3], box[0]:box[2]]
                point_cloud, _ = self.camera.back_project(depth, color_image)
                if point_cloud.shape[0] > 100:
                    point_cloud = torch.from_numpy(np.transpose(point_cloud)).float().unsqueeze(0)
                    point_cloud = point_cloud.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    self._model(tensor_image, point_cloud)

if __name__ == '__main__':
    pf = pointfusion.Inference()
    pf.camera = pointfusion.D455()

    for (color_image, depth_image, point_cloud) in pf.camera:
        pf.predict(color_image, depth_image)
import torch
import torchvision
import pointfusion
import numpy as np
import cv2

class PointFusion:
    def __init__(self):
        self._dataset = None
        self._trainer = None
        self._frcn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        self._frcn.eval()
        self._frcn.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def train(self, dataset=None, epochs=10, batch_size=1, lr=1e-3, weight_decay=1e-5):
        self._dataset = dataset
        self._trainer = trainer.Trainer()

    def predict(self, color_image, point_cloud):
        tensor_image = torch.from_numpy(np.transpose(color_image.copy() / 255.0, (2, 0, 1))).float()
        tensor_image = tensor_image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        outputs = self._frcn([tensor_image])

        # get the predicted boxes, labels, and scores from the output
        pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].detach().cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()

        # load the original image
        #image = np.transpose(color_image.copy(), (2, 0, 1))
        image = color_image.copy()
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > 0.9:
                # convert the box coordinates to integers
                box = box.astype(np.int32)
                # draw the box and label on the image
                #cv2.rectangle(image, box[0], box[1], color=(0, 255, 0), thickness=2)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                #cv2.putText(image, f'{label_map[label]} {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       

        # show the image
        cv2.imshow('image', image)
        cv2.waitKey(1)

    def inference(self):
        for depth_image, color_image in self._camera.run():
            self.predict(depth_image, color_image)

pf = PointFusion()
camera = pointfusion.D455()

for (color_image, depth_image, point_cloud) in camera:
    pf.predict(color_image, point_cloud)

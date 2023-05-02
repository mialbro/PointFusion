import torch
import torchvision
import trainer

class PointFusion:
    def __init__(self):
        self._camera = None
        self._dataset = None
        self._trainer = None
        self._frcn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    def train(self, dataset=None, epochs=10, batch_size=1, lr=1e-3, weight_decay=1e-5):
        self._dataset = dataset
        self._trainer = trainer.Trainer()

    def predict(self, image=None, depth_image=None):
        out = self._frcn(image)
        cropped_images = []
        cropped_depths = []
        cropped_clouds = []
        out = self._model(cropped_images, cropped_clouds)


    def image_detection()
import torch
import torchvision
import trainer
import pointfusion

class PointFusion:
    def __init__(self):
        self._camera = pointfusion.D455()
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

    def inference(self):
        for depth_image, color_image in self._camera.run():
            self.predict(depth_image, color_image)

pf = PointFusion()
camera = pointfusion.D455()

for (color_image, point_cloud) in camera:
    pf.predict(color_image, point_cloud)

import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()
        model = models.resnet50(pretrained=True)

        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):
        batch = x.size(0)
        x = self.features(x)
        x = x.squeeze()
        x = x.view(batch, 1, 2048)
        return x

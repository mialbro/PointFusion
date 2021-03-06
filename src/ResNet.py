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
        x = self.features(x)
        return x 
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

import PointNet
import ResNet

class PointFusion(nn.Module):
    def __init__(self, global_fusion=True):
        super(PointFusion, self).__init__()
        self.global_fusion = global_fusion
        self.image_embedding = ResNet.ResNetFeatures()
        self.pcl_embedding = PointNet.PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 24)

        self.fusion_dropout = nn.Dropout2d(p=0.4)
        self.relu = torch.nn.ReLU()

    def forward(self, pnt, rgb):
        rgb_feats = self.image_embedding(rgb)
        pnt_feats = self.pcl_embedding(pnt)
        
        if self.global_fusion:
            pnt_feats = pnt_feats[0]

        pnt_feats = pnt_feats.squeeze()
        rgb_feats = rgb_feats.squeeze()

        global_feats = torch.cat((rgb_feats, pnt_feats), 0)
        x  = self.fusion_dropout(global_feats)

        x = self.fc1(global_feats)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = x.view(8, 3)
        return x
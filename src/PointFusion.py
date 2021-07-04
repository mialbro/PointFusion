import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

import PointNet
import ResNet

class PointFusion(nn.Module):
    def __init__(self, pnt_cnt=100):
        super(PointFusion, self).__init__()
        self.image_embedding = ResNet.ResNetFeatures()
        self.pcl_embedding = PointNet.PointNetEncoder(feature_transform=True, channel=3)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 24)
        self.fc5 = nn.Linear(128, 1)

        self.fusion_dropout = nn.Dropout2d(p=0.4)
        self.relu = torch.nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, img, cloud):
        B, N, D = cloud.shape
        # extract rgb (1 x 2048) features from image patch
        img_feats = self.image_embedding(img)
        # extract point-wise (n x 64) and global (1 x 1024) features from pointcloud
        point_feats, global_feats = self.pcl_embedding(cloud)
        # duplicate first row for each point in the pointcloud
        img_feats = img_feats.repeat(1, D, 1)
        global_feats = global_feats.repeat(1, D, 1)
        # concatenate features along columns
        dense_feats = torch.cat([img_feats, point_feats, global_feats], 2)
        dense_feats = self.fusion_dropout(dense_feats)
        # pass features through mlp
        x = self.fc1(dense_feats) # (n x 3136)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x) # (n x 128)
        # get corner offsets (n x 8 x 3) to 3d corners for each point
        corner_offsets = self.fc4(x)
        corner_offsets = corner_offsets.view(B, D, 8, 3)
        # obtain pdf ranking each offset to 3d bounding box
        scores = self.fc5(x)
        scores = self.softmax(scores)
        scores = scores.view(B, D)
        return corner_offsets, scores

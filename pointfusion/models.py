import open3d as o3d

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(2048, 3048, 1)
        self.conv2 = nn.Conv1d(3048, 2048, 1)
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):
        B = x.size(0)
        x = self.features(x)
        x = x.view(B, -1, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class PointNet(nn.Module):
    def __init__(self, feature_transform=True, channel=3):
        super(PointNet, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.input_transform = STN3d(channel)
        self.feature_transform = STNkd(k=64)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(64, 64, 1)
        self.conv5 = torch.nn.Conv1d(1088, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 256, 1)
        self.conv7 = torch.nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)

    def forward(self, x):
        B, D, N = x.size()
        transform = self.input_transform(x)
        x = torch.bmm(transform, x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv4(x))
        feature = self.feature_transform(x)
        x = torch.bmm(feature, x)
        point_features = x
        # Global features
        global_features = self.relu(self.conv2(x))
        global_features = self.relu(self.conv3(global_features))
        global_features = self.relu(torch.max(global_features, 2, keepdim=False)[0])
        # Concatenate global and local features
        fusion_features = global_features.view(-1, 1024, 1).repeat(1, 1, N)
        fusion_features = torch.cat([point_features, fusion_features], 1)
        fusion_features = self.relu(self.conv5(fusion_features))
        fusion_features = self.relu(self.conv6(fusion_features))
        fusion_features = self.relu(self.conv7(fusion_features))
        global_features = self.relu(global_features.unsqueeze(2))

        return point_features, global_features, fusion_features

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(3).flatten().astype(np.float32)).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointFusion(nn.Module):
    def __init__(self, pnt_cnt=100):
        super(PointFusion, self).__init__()
        self.pnt_cnt = pnt_cnt
        self.resnet = ResNet()
        self.pointnet = PointNet(feature_transform=True, channel=3)
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(3136, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 24, 1)
        self.conv5 = torch.nn.Conv1d(24, 24, 1)
        self.conv6 = torch.nn.Conv1d(24, 12, 1)
        self.conv7 = torch.nn.Conv1d(12, 1, 1)
        self.conv8 = torch.nn.Conv1d(6, 3, 1)
        self.conv9 = torch.nn.Conv1d(3, 1, 1)

        self.fusion_dropout = nn.Dropout2d(p=0.4)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, image, cloud):
        B, N, D = cloud.size()
        # extract rgb (1 x 2048) features from image patch
        image_features = self.resnet(image)
        # extract point-wise (n x 64) and global (1 x 1024) features from pointcloud
        point_features, global_features, fusion_features = self.pointnet(cloud)

        # duplicate first row for each point in the pointcloud
        image_features = image_features.repeat(1, 1, D)
        global_features = global_features.repeat(1, 1, D)
        # concatenate features along columns
        dense_features = torch.cat([image_features, point_features, global_features], 1)
        # pass features through mlp
        x = self.relu(self.conv1(dense_features))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # get corner offsets (n x 8 x 3) to 3d corners for each point
        x = self.conv4(x)
        corner_offsets = x.view(B, N, 8, D) #   (B, D, 8, 3)
        
        # obtain pdf ranking each offset to 3d bounding box
        x = self.relu(x)

        scores = self.relu(self.conv5(x))
        scores = self.relu(self.conv6(scores))
        scores = self.conv7(scores)
        #scores = self.relu(self.conv8(scores))
        #scores = self.relu(self.conv9(scores))

        import pdb; pdb.set_trace()
        print((scores.min(), scores.max()))

        scores = self.softmax(scores)

        scores = scores.squeeze(0)
        
        if self.training is False:
            indices = torch.argmax(scores, dim=1)
            # top scores
            scores = scores[torch.arange(B), indices]
            # top scoring points
            points = cloud[torch.arange(B), :, indices]
            # top scoring corner offsets
            offsets = corner_offsets[torch.arange(B), :, :, indices]

            # corners
            corners = torch.zeros(B, 3, 8).cuda()
            corners[:, :, 0] = points - offsets[:, :, 0]
            corners[:, :, 1] = points - offsets[:, :, 1]
            corners[:, :, 2] = points - offsets[:, :, 2]
            corners[:, :, 3] = points - offsets[:, :, 3]
            corners[:, :, 4] = points - offsets[:, :, 4]
            corners[:, :, 5] = points - offsets[:, :, 5]
            corners[:, :, 6] = points - offsets[:, :, 6]
            corners[:, :, 7] = points - offsets[:, :, 7]

            return corners, scores
        else:
            return corner_offsets, scores
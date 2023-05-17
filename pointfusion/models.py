import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pointfusion

class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):
        batch = x.size(0)
        x = self.features(x)
        x = x.squeeze()
        x = x.view(batch, 1, 2048)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3 :
            x, feature = x.split(3, dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)

        x = F.relu(self.conv1(x))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        pointfeat = pointfeat.permute(0, 2, 1)

        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, -1, 1024)
        globalfeat = x
        return pointfeat, globalfeat

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

        iden = Variable(
            torch.from_numpy(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

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

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointFusion(nn.Module):
    def __init__(self, pnt_cnt=100):
        super(PointFusion, self).__init__()
        self.pnt_cnt = pnt_cnt
        self.image_embedding = ResNetFeatures()
        self.cloud_embedding = PointNetEncoder(feature_transform=True, channel=3)

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

    def forward(self, image, cloud):
        B, N, D = cloud.size()
        # extract rgb (1 x 2048) features from image patch
        image_feats = self.image_embedding(image)
        # extract point-wise (n x 64) and global (1 x 1024) features from pointcloud
        point_feats, global_feats = self.cloud_embedding(cloud)
        # duplicate first row for each point in the pointcloud
        image_feats = image_feats.repeat(1, D, 1)
        global_feats = global_feats.repeat(1, D, 1)
        # concatenate features along columns
        dense_feats = torch.cat([image_feats, point_feats, global_feats], 2)
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
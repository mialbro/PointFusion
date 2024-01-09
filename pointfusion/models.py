import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Optional, List, Tuple

import pointfusion

class ResNet(nn.Module):
    """ResNet wrapper model
    Attributes:
        relu (nn.Relu): Relu activation function
        conv1 (nn.Conv1d): 1D convolution
        conv2 (nn.Conv1d): 1D convolution
        model (nn.Module): ResNet model
        features (nn.Sequential): ResNet with last layer removed
    Args:
        outupt_features (Optional[int]): Size of extracted features
    """
    def __init__(self, output_features: Optional[int] = 2048):
        super(ResNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(2048, 3048, 1)
        self.conv2 = nn.Conv1d(3048, output_features, 1)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass image to ResNet model
        Args:
            x (torch.Tensor): Input image
        Returns:
            Extracted image features"""
        B = x.size(0)
        x = self.features(x)
        x = x.view(B, -1, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.squeeze(2)
        return x

    def channels(self) -> tuple:
        """Extracts number of channels in the output data
        Returns:
            Number of channels in output
        """
        training = self.training
        self.eval()
        
        channels = None
        with torch.no_grad():
            device = None
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            channels = self.forward(torch.randn((1, 3, 244, 244), device=device)).size()[1]
      
        if training:
            self.train()

        return channels

# ============================================================================
# Point Net Backbone (main Architecture)
class PointNetBackbone(nn.Module):
    """This is the main portion of Point Net before the classification and segmentation heads.
    The main function of this network is to obtain the local and global point features, 
    which can then be passed to each of the heads to perform either classification or
    segmentation. The forward pass through the backbone includes both T-nets and their 
    transformations, the shared MLPs, and the max pool layer to obtain the global features.

    The forward function either returns the global or combined (local and global features)
    along with the critical point index locations and the feature transformation matrix. The
    feature transformation matrix is used for a regularization term that will help it become
    orthogonal. (i.e. a rigid body transformation is an orthogonal transform and we would like
    to maintain orthogonality in high dimensional space). "An orthogonal transformations preserves
    the lengths of vectors and angles between them

    https://github.com/romaintha/pytorch_pointnet.git
    
    Args:
        num_points (Optional[int]): Number of inputted points
        num_global_features (Optional[int]): Number of extracted global features
    Attributes:
        num_points (Optional[int]): Number of inputted points
        num_global_features (Optional[int]): Number of extracted global features
    """
    def __init__(self, num_points: Optional[int] = 500, num_global_feats: Optional[int] = 1024):
        super(PointNetBackbone, self).__init__()

        # if true concat local and global features
        self.num_points = num_points
        self.num_global_feats = num_global_feats

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

        # shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        self.conv6 = nn.Conv1d(1088, 512, kernel_size=1)
        self.conv7 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=1)
        
        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=False)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # get batch size
        bs = x.shape[0]
        
        # pass through first Tnet to get transform matrix
        A_input = self.tnet1(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        # pass through first shared MLP
        #x = self.bn1(F.relu(self.conv1(x)))
        #x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # get feature transform
        A_feat = self.tnet2(x)

        # perform second transformation across each (64 dim) feature in the batch
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        # store local point features for segmentation head
        local_features = x.clone()

        # pass through second MLP
        #x = self.bn3(F.relu(self.conv3(x)))
        #x = self.bn4(F.relu(self.conv4(x)))
        #x = self.bn5(F.relu(self.conv5(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        global_feature = self.max_pool(x)
        global_feature = global_feature.view(bs, -1)

        point_features = torch.cat((local_features, global_feature.unsqueeze(-1).repeat(1, 1, self.num_points)), dim=1)
        point_features = F.relu(self.conv6(point_features))
        point_features = F.relu(self.conv7(point_features))
        point_features = F.relu(self.conv8(point_features))
        
        return global_feature, point_features
    
    def channels(self) -> tuple:
        """Extracts number of channels in the output data
        Returns:
            Number of channels in output
        """
        training = self.training
        self.eval()

        channels = None
        with torch.no_grad():
            device = None
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            out = self.forward(torch.randn((1, 3, self.num_points), device=device))
            channels = [ x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in out ]
        
        if training:
            self.train()

        return channels

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, k: Optional[int] = 64):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
   
class Tnet(nn.Module):
    """https://github.com/romaintha/pytorch_pointnet.git
    T-Net learns a Transformation matrix with a specified dimension 
    Args:
        dim (int): Input dimension size
        num_points (Optional[int]): Number of input points
    Attributes:
        dim (int): Input dimension size
        conv1 (nn.Conv1d):
        conv2 (nn.Conv1d):
        conv3 (nn.Conv1d):
        linear1 (nn.Linear):
        linear2 (nn.Linear):
        linear3 (nn.Linear):
        bn1 (nn.BatchNorm1d):
        bn2 (nn.BatchNorm1d):
        bn3 (nn.BatchNorm1d):
        bn4 (nn.BatchNorm1d):
        bn5 (nn.BatchNorm1d):
        max_pool (nn.MaxPool1d): 
    """
    def __init__(self, dim: int, num_points: Optional[int] = 2500):
        super(Tnet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim 

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass batch of tensors through transformation network
        Args:
            x (torch.Tensor): Input point embedding
        Returns:
            Tensor
        """
        bs = x.shape[0]

        # pass through shared MLP layers (conv1d)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # max pool over num points
        x = self.max_pool(x).view(bs, -1)

        # pass through MLP
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x
    
class GlobalFusion(nn.Module):
    """Global Fusion model which ingests images and/or point clouds and directly regresses the 8 corners of the 3D bounding box
    Args:
        point_count (Optional[int]): Number of points in point cloud
        modalities (Optional[pointfusion.Modality]): Input data modalities
    Attributes:
        modalities (Optional[pointfusion.Modality]): Input data modalities
        image_encoder (nn.Module): Network that extracts image features
        point_encoder (nn.Module): Network that extracts point cloud features
        model (torch.nn.Sequential): Global fusion model
        """
    def __init__(self, point_count: Optional[int] = 100, modalities: Optional[List[pointfusion.Modality]] = [pointfusion.Modality.RGB]):
        super(GlobalFusion, self).__init__()
        self.modalities = modalities
        self.image_encoder = ResNet(output_features=2048)
        self.point_encoder = PointNetBackbone(num_points=point_count)

        input_fusion_size = 0
        if pointfusion.Modality.RGB in modalities:
            input_fusion_size += self.image_encoder.channels()
        if pointfusion.Modality.POINT_CLOUD in modalities:
            input_fusion_size += self.point_encoder.channels()[0]

        layers = []
        channels = np.linspace(input_fusion_size, 24, num=10, dtype=int)
        for i, channel in enumerate(channels):
            if (i + 1) < len(channels):
                layers.append(torch.nn.ReLU())
                layers.append(nn.Conv1d(channel, channels[i+1], 1))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, image: Optional[torch.Tensor] = None, point_cloud: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process images and / or point clouds
        Args:
            image (torch.Tensor): Input image
            point_cloud (torch.Tensor): Input point cloud
        Returns:
            8 corners of 3D bounding box
        """
        B, D, N = point_cloud.size() if point_cloud is not None else image.size()

        features = None
        point_features = None
        image_features = None

        # Extract point-wise (n x 64) and global (1 x 1024) features from point cloud
        if pointfusion.Modality.POINT_CLOUD in self.modalities:
            if point_cloud is None:
                raise Exception("Must supply Point Cloud...")
            point_features, _ = self.point_encoder(point_cloud)
        
        # Extract image features
        if pointfusion.Modality.RGB in self.modalities:
            if image is None:
                raise Exception("Must supply image")
            image_features = self.image_encoder(image)

        # Fuse features
        if len(self.modalities) == 2:
            features = torch.concatenate([image_features, point_features], axis=1).unsqueeze(2)
        elif pointfusion.Modality.RGB in self.modalities:
            features = image_features.unsqueeze(2)
        elif pointfusion.Modality.POINT_CLOUD in self.modalities:
            features = point_features.unsqueeze(2)
        
        features = self.model(features)
        features = features.view(B, 3, 8)
        
        return features
    

class DenseFusion(nn.Module):
    """Dense Fusion model which ingests images and/or point clouds and directly regresses the 8 corners of the 3D bounding box
    Args:
        point_count (Optional[int]): Number of points in point cloud
        modalities (Optional[pointfusion.Modality]): Input data modalities
    Attributes:
        modalities (Optional[pointfusion.Modality]): Input data modalities
        image_encoder (nn.Module): Network that extracts image features
        point_encoder (nn.Module): Network that extracts point cloud features
        model (torch.nn.Sequential): Global fusion model
        """
    def __init__(self, point_count: Optional[int] = 100, modalities: Optional[List[pointfusion.Modality]] = [pointfusion.Modality.RGB]):
        super(DenseFusion, self).__init__()
        self.modalities = modalities
        self.image_encoder = ResNet(output_features=2048)
        self.point_encoder = PointNetBackbone(num_points=point_count)

        input_fusion_size = 0
        if pointfusion.Modality.RGB in modalities:
            input_fusion_size += self.image_encoder.channels()
        if pointfusion.Modality.POINT_CLOUD in modalities:
            input_fusion_size += sum([ channel_count for channel_count in self.point_encoder.channels() ])
        
        layers = []
        channels = np.linspace(input_fusion_size, 128, num=10, dtype=int)
        for i, channel in enumerate(channels):
            if (i + 1) < len(channels):
                layers.append(nn.Conv1d(channel, channels[i+1], 1))
                layers.append(torch.nn.ReLU())

        self.backbone = torch.nn.Sequential(*layers)

        self.localization_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 24)
        )

        self.scoring_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

        self.soft_max = torch.nn.Softmax(dim=1)

    def forward(self, image: Optional[torch.Tensor] = None, point_cloud: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process images and / or point clouds
        Args:
            image (torch.Tensor): Input image
            point_cloud (torch.Tensor): Input point cloud
        Returns:
            N x 1 corner offset scores
            N x 8 corner offsets of 3D bounding box
        """
        B, D, N = point_cloud.size() if point_cloud is not None else image.size()

        features = None
        point_features = None
        image_features = None

        # Extract point-wise (n x 64) and global (1 x 1024) features from point cloud
        if pointfusion.Modality.POINT_CLOUD in self.modalities:
            if point_cloud is None:
                raise Exception("Must supply Point Cloud...")
            global_features, point_wise_features = self.point_encoder(point_cloud)
            global_features = global_features.unsqueeze(2).repeat(1, 1, 400)
            point_features = torch.concatenate((global_features, point_wise_features), axis=1)

        # Extract image features
        if pointfusion.Modality.RGB in self.modalities:
            if image is None:
                raise Exception("Must supply image")
            image_features = self.image_encoder(image)

        # Fuse features
        if len(self.modalities) == 2:
            image_features = image_features.unsqueeze(2).repeat((1, 1, point_features.size()[-1]))
            features = torch.concatenate([image_features, point_features], axis=1)
        elif pointfusion.Modality.RGB in self.modalities:
            features = image_features.unsqueeze(2) 
        elif pointfusion.Modality.POINT_CLOUD in self.modalities:
            features = point_features.squeeze(2)

        features = self.backbone(features)

        corner_offsets = self.localization_head(features.swapaxes(1, 2))
        scores = self.scoring_head(features.swapaxes(1, 2))

        corner_offsets = corner_offsets.view(B, 3, 8, -1)
        scores = scores.squeeze(2)

        print((scores[0].min().item(), scores[1].max().item()))
        scores = self.soft_max(scores)
        print((scores[0].min().item(), scores[1].max().item()))
        print()

        return scores, corner_offsets
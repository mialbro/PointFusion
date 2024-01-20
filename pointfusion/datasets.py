import os
import glob
import yaml
import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from typing import Optional

import pointfusion.utils as utils
from pointfusion.camera import Camera
from pointfusion.enums import ModelName, Modality

def normalize(mean, distance, points):
    points = points - mean
    points = ((points / distance) + 1) / 2
    return points

def normalize_point_cloud(points: np.ndarray, mean: float, distance: float) -> np.ndarray:
    
    points = points - mean.unsqueeze(2)
    distance = distance.max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    points = (points / distance)
    points = (points + 1) / 2
    return points

class NormalizePointCloud:
    def __call__(self, points):
        '''
        B, C, N = points.size()
        mean = points.mean(dim=2)
        points = points - mean.unsqueeze(2)
        distance = torch.sqrt(torch.sum(torch.abs(points) ** 2, dim=1))
        distance = distance.max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        points = (points / distance)
        points = (points + 1) / 2
        return points.view(C, N)
        '''
        B, C, N = points.size()
        mean = points.mean(dim=2)
        distance = torch.sqrt(torch.sum(torch.abs(points) ** 2, dim=1))
        distance = distance.max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        points = normalize(mean, distance, points)
        return points.view(C, N)
        '''
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= 
        '''

class LINEMOD(Dataset):
    """
    LINEMOD dataset reader
    Attributes:
        model_name (pointfusion.ModalityName): Name of pointfusion model used
        depths (list): List of depth images
        masks (list): 
        images (list):
        rvecs (list)
        tvecs (list):
        intrinsics (list): 
        models (list):
        ids (list):
        point_count (int):
        image_transform (torchvision.transforms):
        cloud_transform (torchvision.transforms):
    Args:
        root_dir (Optional[str]): Data directory
        point_count (Optional[int]):
        model_name (Optional[pointfusion.ModelName]):
    """
    def __init__(
            self, 
            root_dir: Optional[str] = '../datasets/Linemod_preprocessed', 
            point_count: Optional[int] = 400, 
            model_name: Optional[ModelName] = ModelName.DenseFusion,
            modalities: Optional[Modality] = [ Modality.RGB ]
    ) -> None:
        self.modalities = modalities
        self.model_name = model_name
        self.depths = []
        self.masks = []
        self.images = []
        self.rvecs = []
        self.tvecs = []
        self.intrinsics = []
        self.models = []
        self.ids = []
        self.point_count = point_count # np.random.randint(100, 1000)

        self.image_transform = transforms.Compose([
            transforms.Resize((300, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cloud_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizePointCloud()
        ])

        self.corner_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizePointCloud()
        ])

        for i, path in enumerate(sorted(glob.glob(os.path.join(root_dir, 'data', '*')))):
            ids = [int(path.split('/')[-1])]
            depths = sorted(glob.glob(os.path.join(path, 'depth', '*.png')))
            masks = sorted(glob.glob(os.path.join(path, 'mask', '*.png')))
            images = sorted(glob.glob(os.path.join(path, 'rgb', '*.png')))

            n = min([len(depths), len(masks), len(images)])
            self.depths += depths[:n]
            self.masks += masks[:n]
            self.images += images[:n]
            self.models += [f'{os.path.join(root_dir, "models", f"obj_{ids[0]:02d}.ply")}'] * n
            self.model_info = yaml.load(open(os.path.join(root_dir, 'models', 'models_info.yml')), Loader=yaml.FullLoader)
            self.ids += ids * n

            with open(os.path.join(path, 'info.yml')) as f:
                info = yaml.load(f, Loader=yaml.FullLoader)
                self.intrinsics += [ info[k]['cam_K'] for k in sorted(info.keys()) ][:n]

            with open(os.path.join(path, 'gt.yml')) as f:
                gt = yaml.load(f, Loader=yaml.FullLoader)
                for k in sorted(gt.keys()):
                    for j in range(len(gt[k])):
                        if gt[k][j]['obj_id'] == ids[0]:
                            self.rvecs += [ gt[k][j]['cam_R_m2c'] ]
                            self.tvecs += [ gt[k][j]['cam_t_m2c'] ]
            break
            
    def __getitem__(self, index: int) -> tuple:
        """
        Extracts input and target data
        Args:
            index (int): Index of data in dataset
        Returns:
            Input and target data
        """
        id = self.ids[index]
        depth = np.array(Image.open(self.depths[index]))
        mask = np.array(Image.open(self.masks[index]))
        image = np.array(Image.open(self.images[index])) # load the mask
        model = np.asarray(o3d.io.read_point_cloud(self.models[index]).points)
        
        camera = Camera(camera_matrix=self.intrinsics[index], rotation=self.rvecs[index], translation=self.tvecs[index])
        model = camera.transform(model)

        image = np.where(mask, image, np.nan).astype(image.dtype)
        depth = np.where(mask[:, :, 0], depth, np.nan).astype(depth.dtype)

        depth_cloud, colors = camera.back_project(depth, image)
        corners = utils.get_corners(model)
        xx = ([self.model_info[id]['min_x'], self.model_info[id]['min_x'] + self.model_info[id]['size_x']])
        yy = ([self.model_info[id]['min_y'], self.model_info[id]['min_y'] + self.model_info[id]['size_y']])
        zz = ([self.model_info[id]['min_z'], self.model_info[id]['min_z'] + self.model_info[id]['size_z']])

        cc = []
        for x in xx:
            for y in yy:
                for z in zz:
                    cc.append([x, y, z])
        cc = np.asarray(cc)

        corners = camera.transform(cc)
        
        corner_offsets = utils.get_corner_offsets(depth_cloud, corners)
        if depth_cloud.shape[0] > self.point_count:
            sample = np.random.choice(depth_cloud.shape[0], self.point_count, replace=False)
        else:
            sample = np.random.choice(depth_cloud.shape[0], self.point_count, replace=True)
        
        corner_offsets = corner_offsets[sample]
        # sample depth point cloud
        depth_cloud = np.transpose(depth_cloud[sample])
        # crop image
        rmin, rmax, cmin, cmax = utils.bbox_from_mask(mask)

        #import pdb; pdb.set_trace()
        #image_ = np.zeros(image.shape, dtype=image.dtype)
        #image_[rmin:rmax, cmin:cmax] = image[rmin:rmax, cmin:cmax]
        #image_ = Image.fromarray(image)
        image_ = Image.fromarray(image[rmin:rmax, cmin:cmax])
        #import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()
        
        # rearrange corner offset dimensions
        corner_offsets = np.swapaxes(corner_offsets, 0, 2)
        corners = np.swapaxes(corners, 0, 1)

        '''
        mean = points.mean(axis=1)
        distance = torch.sqrt(np.sum(np.abs(corners) ** 2))
        distance = distance.max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        points = normalize(mean, distance, points)
        '''

        if self.model_name is ModelName.GlobalFusion:
            return id, self.image_transform(image_), self.cloud_transform(depth_cloud).to(torch.float), self.corner_transform(corners)
        elif self.model_name is ModelName.DenseFusion:
            return id, self.image_transform(image_), self.cloud_transform(depth_cloud).to(torch.float), self.corner_transform(corner_offsets)

    def __len__(self) -> int:
        return len(self.ids)
    
    def split(self, ratio: Optional[float] = 0.8) -> torch.utils.data.Dataset:
        """
        Splits dataset using ratio
        Returns:
            Split dataset
        """
        train_size = int(ratio * len(self))
        test_size = int(len(self) - train_size)
        return torch.utils.data.random_split(self, [train_size, test_size])
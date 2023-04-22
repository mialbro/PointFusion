import os
import glob
import yaml

import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

import utils
from camera import Camera

class LINEMOD(Dataset):
    def __init__(self, root_dir='../datasets/Linemod_preprocessed', corner_offset_sample=50, point_sample=1000):
        self.depths = []
        self.masks = []
        self.images = []
        self.rvecs = []
        self.tvecs = []
        self.intrinsics = []
        self.models = []
        self.ids = []
        self.point_sample = point_sample
        self.corner_offset_sample = corner_offset_sample

        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cloud_transform = None

        for i, path in enumerate(sorted(glob.glob(os.path.join(root_dir, 'data', '*')))):
            depths = sorted(glob.glob(os.path.join(path, 'depth', '*.png')))
            masks = sorted(glob.glob(os.path.join(path, 'mask', '*.png')))
            images = sorted(glob.glob(os.path.join(path, 'rgb', '*.png')))
            ids = [i+1]

            n = min([len(depths), len(masks), len(images)])
            self.depths += depths[:n]
            self.masks += masks[:n]
            self.images += images[:n]
            self.models += [f'{os.path.join(root_dir, "models", f"obj_{i+1:02d}.ply")}'] * n
            self.ids += ids * n

            with open(os.path.join(path, 'info.yml')) as f:
                info = yaml.load(f, Loader=yaml.FullLoader)
                self.intrinsics += [ info[k]['cam_K'] for k in sorted(info.keys()) ][:n]
            with open(os.path.join(path, 'gt.yml')) as f:
                gt = yaml.load(f, Loader=yaml.FullLoader)
                self.rvecs += [ gt[k][0]['cam_R_m2c'] for k in sorted(gt.keys()) ][:n]
                self.tvecs += [ gt[k][0]['cam_t_m2c'] for k in sorted(gt.keys()) ][:n]
            break

    def __getitem__(self, index):
        id = self.ids[index]
        depth = np.array(Image.open(self.depths[index]))
        mask = np.array(Image.open(self.masks[index]))
        image = np.array(Image.open(self.images[index])) # load the mask
        model = np.asarray(o3d.io.read_point_cloud(self.models[index]).points)
        
        camera = Camera(camera_matrix=self.intrinsics[index], rotation=self.rvecs[index], translation=self.tvecs[index])
        model = camera.transform(model)
        image = np.where(mask, image, np.nan).astype(image.dtype)
        depth = np.where(mask[:, :, 0], depth, np.nan).astype(depth.dtype)
        
        depth_cloud = camera.depth_to_cloud(depth)
        corners = utils.get_corners(model)
        # per-point corner offsets
        corner_offsets = utils.get_corner_offsets(depth_cloud, corners)

        sample = np.random.choice(depth_cloud.shape[0], self.point_sample, replace=True)
        corner_offsets = corner_offsets[sample]
        # sample depth point cloud
        depth_cloud = np.transpose(depth_cloud[sample])
        # crop image
        rmin, rmax, cmin, cmax = utils.bbox_from_mask(mask)
        cropped_image = Image.fromarray(image[rmin:rmax, cmin:cmax])
        import pdb; pdb.set_trace()

        return id, self.image_transform(cropped_image), torch.from_numpy(depth_cloud).to(torch.float), torch.from_numpy(corners), torch.from_numpy(corner_offsets)

    def __len__(self):
        return len(self.ids)
    
    def split(self, ratio=0.8):
        train_size = int(ratio * len(self))
        test_size = int(len(self) - train_size)
        return torch.utils.data.random_split(self, [train_size, test_size])
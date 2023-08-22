import open3d as o3d
import utils
import os
import glob
import yaml
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

from camera import Camera
import utils

class LINEMOD(Dataset):
    def __init__(self, root_dir='../datasets/Linemod_preprocessed', point_count=400):
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cloud_transform = transforms.Compose([
            transforms.ToTensor(),
            utils.NormalizePointCloud()
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

        #image_ = np.zeros(image.shape, dtype=image.dtype)
        #image_[rmin:rmax, cmin:cmax] = image[rmin:rmax, cmin:cmax]
        image_ = Image.fromarray(image[rmin:rmax, cmin:cmax])
        
        # rearrange corner offset dimensions
        corner_offsets = np.swapaxes(corner_offsets, 0, 2)

        return id, self.image_transform(image_), self.cloud_transform(depth_cloud).to(torch.float), torch.from_numpy(corners), torch.from_numpy(corner_offsets) # , torch.from_numpy(np.asarray(self.rvecs[index]).reshape((3, 3))).to(torch.float)

    def __len__(self):
        return len(self.ids)
    
    def split(self, ratio=0.8):
        train_size = int(ratio * len(self))
        test_size = int(len(self) - train_size)
        return torch.utils.data.random_split(self, [train_size, test_size])
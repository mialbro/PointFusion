import os
import glob
import yaml

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import utils
from camera import Camera

class LINEMOD(Dataset):
    def __init__(self, root_dir='../datasets/Linemod_preprocessed'):
        self.depths = []
        self.masks = []
        self.images = []
        self.rvecs = []
        self.tvecs = []
        self.intrinsics = []
        self.models = []
        self.ids = []

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

    def __getitem__(self, index): # segmented image, segmented depth, corners, back projected points, pose
        id = self.ids[index]
        depth = np.array(Image.open(self.depths[index]))
        mask = np.array(Image.open(self.masks[index]))
        image = np.array(Image.open(self.images[index])) # load the mask
        model = self.models[index]
        camera = Camera(camera_matrix=self.intrinsics[index], rotation=self.rvecs[index], translation=self.tvecs[index])

        image = np.where(mask, image, np.nan).astype(image.dtype)
        depth = np.where(mask[:, :, 0], depth, np.nan).astype(depth.dtype)
        
        depth_cloud = camera.depth_to_cloud(depth)

        object_id = self.object_ids[index] # id of model corresponding to this index
        frame_id = self.ground_truth_ids[index] # the ground_truth id for this index
        # get the index of our model in the current groundtruth frame
        model_index = utils.getObjId(object_id, self.ground_truth[object_id][frame_id])
        # get the groundtruth data
        model_ground_truth = self.ground_truth[object_id][frame_id][model_index]
        # get the model corresponding to this index and normalize it
        model_points = self.model_pcd[object_id]
        # get the bounding box
        rmin, rmax, cmin, cmax = utils.getBoundingBox(mask)
        # segment the images
        segmented_img = img[rmin:rmax, cmin:cmax, :]
        segmented_img = Image.fromarray(segmented_img)
        depth_mask = np.zeros(depth.shape)
        depth_mask[rmin:rmax, cmin:cmax] = True
        # get the ground_truth pose
        mTc = utils.getPose(model_ground_truth)
        # apply transformation to the model pointcloud
        model_points = utils.transformCloud(model_points, (mTc))
        # get the model corners (ground truth)
        cloud = utils.depthToCloud(depth, depth_mask, self.camera_intrinsics)
        corners = utils.getCorners(model_points)
        corner_offsets = utils.getCornerOffsets(corners, cloud)
        # sample points
        corner_offsets, cloud = utils.sampleCloud(corner_offsets, cloud, self.pnt_cnt)
        #utils.draw3dCorners(cloud, utils.getCornersFromOffsets(corner_offsets, cloud)[0])
        #utils.draw3dCorners(cloud, corners)
        return (self.transform((segmented_img)),torch.from_numpy(cloud.astype(np.float32)),torch.from_numpy(corner_offsets.astype(np.float32)))

    def __len__(self):
        return self.length

dataset = LINEMOD()
dataset[0]
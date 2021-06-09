import os
import torch
from torch.utils.data import Dataset
#from skimage import io
import utils
import numpy as np
import open3d

import torch.nn as nn
from torchvision import transforms, datasets
import yaml
from PIL import Image
import utils
import numpy.ma as ma
from scipy.spatial import distance
import cv2
import random

class PointFusionDataset(Dataset):
    def __init__(self, root_dir='', mode='train', pnt_cnt=100, transform=None):
        self.object_list = [1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]
        #self.object_list = [4]
        self.model_pcd = {}
        self.image_paths = []
        self.mask_paths = []
        self.depth_paths = []
        self.ground_truth = {}
        self.ground_truth_ids = []
        self.object_ids = []
        self.root_dir = root_dir
        self.transform = transform
        self.pnt_cnt = pnt_cnt
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        item_cnt = 0
        for item in self.object_list:
            input_file = open('{}/data/{}/train.txt'.format(self.root_dir, '%02d' % int(item)))
            line_cnt = 0
            while True:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.image_paths.append('{}/data/{}/rgb/{}.png'.format(self.root_dir, '%02d' % item, input_line))
                self.depth_paths.append('{}/data/{}/depth/{}.png'.format(self.root_dir, '%02d' % item, input_line))
                self.mask_paths.append('{}/data/{}/mask/{}.png'.format(self.root_dir, '%02d' % item, input_line))
                self.object_ids.append(item)
                self.ground_truth_ids.append(int(input_line))
                line_cnt += 1
            item_cnt += 1
            # load ground truth file for this given model:
            # conatains the camera poses and bounding boxes for the model
            ground_truth_file = open('{}/data/{}/gt.yml'.format(self.root_dir, '%02d' % item), 'r')
            self.ground_truth[item] = yaml.load(ground_truth_file, Loader=yaml.FullLoader)
            self.model_pcd[item] = utils.openModel('{0}/models/obj_{1}.ply'.format(self.root_dir, '%02d' % item))
            print('Object {} buffer loaded'.format(item))

        fx = 572.41140
        fy = 573.57043
        cx = 325.26110
        cy = 242.04899
        self.camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        self.K = np.array([[fx, 0, cx, 0],
                            [0, fy, cy, 0],
                            [0, 0, 1, 0]])
        self.length = len(self.image_paths)

    def __getitem__(self, index):
        img = np.array(Image.open(self.image_paths[index])) # load the image
        depth = np.array(Image.open(self.depth_paths[index])) # load the depth image
        mask = np.array(Image.open(self.mask_paths[index])) # load the mask
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

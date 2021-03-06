import os
import pandas as pd
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



device = torch.device('cpu')

preprocessing = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PointFusionDataset(Dataset):
    def __init__(self, mode='train', pnt_cnt=0, csv_file='', root_dir='', transform=None):
        #self.object_list = [1, 2, 4, 5, 6, 9, 9, 10, 11, 12, 13, 14, 15]
        self.object_list = [2]

        self.model_pcd = {}

        self.image_paths = []
        self.mask_paths = []
        self.depth_paths = []

        self.ground_truth = {}
        self.ground_truth_ids = []

        self.object_ids = []

        self.root_dir = root_dir
        self.transform = transform

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

        self.length = len(self.image_paths)

        self.camera_intrinsics = {'fx': 572.41140, 'fy': 573.57043, 'cx': 325.26110, 'cy': 242.04899}

        self.pnt_cnt = pnt_cnt
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        print(len(self.image_paths))
        # pillow -> [col, row] == [480, 640]
        img = np.array(Image.open(self.image_paths[index])) # load the image
        depth = np.array(Image.open(self.depth_paths[index])) # load the depth image
        mask = np.array(Image.open(self.mask_paths[index])) # load the mask

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        #if self.mode == 'eval':
        #mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        #else:
        mask_label = ma.getmaskarray(ma.masked_equal(mask, np.array([255, 255, 255])))[:, :, 0]

        mask = mask_label * mask_depth

        object_id = self.object_ids[index] # id of model corresponding to this index
        frame_id = self.ground_truth_ids[index] # the ground_truth id for this index

        # get the index of our model in the current groundtruth frame
        model_index = utils.getObjId(object_id, self.ground_truth[object_id][frame_id])
        # get the groundtruth data
        model_ground_truth = self.ground_truth[object_id][frame_id][model_index]
        # get the model corresponding to this index and normalize it
        model_points = self.model_pcd[object_id] / 1000.0
        # get the bounding box
        rmin, rmax, cmin, cmax = utils.getBoundingBox(mask)
        # segment the images
        segmented_img = img[rmin:rmax, cmin:cmax, :]
        segmented_depth = depth[rmin:rmax, cmin:cmax]

        # project depth map
        cloud = utils.depthToCloud(segmented_depth, self.camera_intrinsics)

        # get the ground_truth pose
        trans = np.reshape(np.array(model_ground_truth['cam_t_m2c']), (3, 1))
        rot = np.resize(np.array(model_ground_truth['cam_R_m2c']), (3, 3))
        mTc = np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))

        # apply transformation to the model pointcloud
        model_points = utils.transformCloud(model_points, np.linalg.inv(mTc))

        # get the model corners
        model_corners = utils.getCorners(model_points)

        # get the n x 8 x 3 offsets from each point in the pointcloud...
        # to the 3D corners
        corner_offsets = utils.getCornerOffsets(model_corners, model_points)
        #utils.viewCloud(cloud)
        dist = utils.computeMinDistance(model_points, cloud)
        print(dist)
        #utils.draw3dCorners(cloud, model_corners)
        return (segmented_img, cloud, corner_offsets)

if __name__ == '__main__':
    pfd = PointFusionDataset(csv_file='pointfusion.csv', root_dir='../datasets/Linemod_preprocessed', transform=preprocessing)
    pfd.__getitem__(2)

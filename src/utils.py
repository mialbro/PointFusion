import numpy as np
import open3d
import torch
import torch.nn as nn
import torchvision
import numpy.ma as ma

def projectPoints(pnts3d, K):
    K = np.array([[K["fx"],  0.0,    K["cx"]],
                    [0.0,    K["fy"], -K["cy"]],
                    [0.0,     0.0,      1.0]])
    pnts2d = K @ pnts3d
    pnts2d = pnts2d / pnts2d[2, :]
    pnts2d = pnts2d[:2, :]
    return pnts2d

def draw2dCorners(color_image, corners):
    for i in range(0, corners.shape[1]):
        corner = corners[:2, i]
        cv2.circle(color_image, corner)
    return color_image


# get the corner offsets with the highest scores
def getPredictedCorner(offsets, scores):
    B, N, D = offsets.shape
    _, indices = torch.max(scores, 1)
    corners = np.zeros((B, 8, 3))
    for i in range(0, B):
        for j in range(0, 8):
            corners[i][j] = (cloud[i, indices[i]] - offsets[i, indices[i], j]).numpy()
    return corners
'''
def getCornersFromOffsets(offsets, cloud):
    B, N, D = cloud.shape
    corners = np.zeros((B, 8, 3))
    for i in range(0, 8):
        corners[B, i, :] = cloud[:, i] - offsets[:, :]
    for i in range(0, cnt):
        for j in range(0, 8):
            corners[i][j] = cloud[i] - offsets[i][j]
    return corners
'''
# (corner_offsets, model_points)
def sampleCloud(offsets, cloud, pnt_cnt):
    cnt = cloud.shape[0]
    if (pnt_cnt > cnt):
        return np.zeros((1, 3)), np.zeros((1, 3))
    idx = np.random.choice(cnt, pnt_cnt, replace=False)
    sampled_offsets = offsets[idx, :]
    sampled_cloud = cloud[idx, :]
    return sampled_offsets, sampled_cloud

def getPose(gt):
    trans = np.reshape(np.array(gt['cam_t_m2c']), (3, 1))
    rot = np.resize(np.array(gt['cam_R_m2c']), (3, 3))
    T = np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))
    return T

'''
def projectPoints(K, T, pnts_3d):
    zeros = np.zeros((pnts_3d.shape[0], 1))
    pnts_3d = np.hstack((pnts_3d, zeros))
    pnts_2d = K @ T @ pnts_3d.T
    pnts_2d = pnts_2d / pnts_2d[2]
    pnts_2d = pnts_2d[0:2, :].T.astype(int)
    return pnts_2d
'''

def normalizeCloud(v):
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    norm_cloud = (v - v_min)/(v_max - v_min)
    return norm_cloud

def normalize2Cloud(cloud1, cloud2):
    cloud1_min = cloud1.min(axis=(0, 1), keepdims=True)
    cloud1_max = cloud1.max(axis=(0, 1), keepdims=True)

    cloud2_min = cloud2.min(axis=(0, 1), keepdims=True)
    cloud2_max = cloud2.max(axis=(0, 1), keepdims=True)

    min = np.minimum(cloud1_min, cloud2_min)
    max = np.maximum(cloud1_max, cloud2_max)

    cloud1_norm = (cloud1 - min)/(max - min)
    cloud2_norm = (cloud2 - min)/(max - min)

def computeMinDistance(pnts1, pnts2):
    pnts1 = pnts1.reshape((-1, 1, 3))                 # [200x1x3]
    pnts2 = np.expand_dims(pnts2, axis=0)             # [1x100x3]
    pnts2 = pnts2.repeat(pnts1.shape[0], axis=0)          # [200x100x3]

    distance = np.linalg.norm(pnts2 - pnts1, axis=2)

def getObjId(object_id, ground_truth):
    model_index = 0
    if object_id == 2:
        # get the number of models observed in this frame
        model_cnt = len(ground_truth)
        for i in range(0, model_cnt):
            # get the id for this model
            model_id = ground_truth[i]['obj_id']
            if model_id == 2:
                # this is the model that we want...
                model_index = i
    return model_index

def depthToCloud(depth, mask, K):
    rows, cols = depth.shape
    cnt = rows * cols
    u, v = np.meshgrid(np.arange(cols), np.arange(rows), sparse=False)
    valid = (depth > 0) & (mask != False)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, (z * (u - K['cx'])) / K['fx'], 0)
    y = np.where(valid, (z * (v - K['cy'])) / K['fy'], 0)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    points = np.dstack((x, y, z))[0]
    points = points[~np.isnan(points).any(axis=1)]
    return points

def getBoundingBox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def getCornerOffsets(corners, cloud):
    cnt = cloud.shape[0]
    corner_offsets = np.zeros((cnt, 8, 3))
    for i in range(0, cnt):
        for j in range(0, 8):
            corner_offsets[i][j] = cloud[i] - corners[j]
    return corner_offsets

def getCornersFromOffsets(corner_offsets, cloud):
    cnt = cloud.shape[0]
    corners = np.zeros((cnt, 8, 3))
    for i in range(0, cnt):
        for j in range(0, 8):
            corners[i][j] = cloud[i] - corner_offsets[i][j]
    return corners

def getCorners(points):
    # get the corners from the transformed pointcloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    obb = open3d.geometry.OrientedBoundingBox()
    obb = obb.create_from_points(pcd.points)
    corners = np.asarray(obb.get_box_points())
    return corners

def transformCloud(cloud, T):
    # get the corners from the transformed pointcloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud)
    pcd = pcd.transform(T)
    points = np.asarray(pcd.points)
    return points

def openModel(path):
    pcd = open3d.io.read_point_cloud(path)
    obb = open3d.geometry.OrientedBoundingBox()
    obb = obb.create_from_points(pcd.points)
    corner_points = np.asarray(obb.get_box_points())
    object_points = np.asarray(pcd.points)
    return object_points

def viewCloud(cloud):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud)
    open3d.visualization.draw_geometries([pcd])

def view2Cloud(cloud1, cloud2):
    pcd1 = open3d.geometry.PointCloud()
    pcd2 = open3d.geometry.PointCloud()
    pcd1.points = open3d.utility.Vector3dVector(cloud1)
    pcd2.points = open3d.utility.Vector3dVector(cloud2)
    open3d.visualization.draw_geometries([pcd1, pcd2])

def draw3dCorners(cloud, corners):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud)
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 6],
        [1, 7],
        [2, 5],
        [2, 7],
        [3, 5],
        [3, 6],
        [4, 5],
        [4, 6],
        [4, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(corners),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([pcd, line_set])

# Get the tightest bounding box surrounding keypoints
def get_2d_bb(box, size=1):
    x = box[0]
    y = box[1]
    min_x = np.min(np.reshape(box, [-1,2])[:,0])
    max_x = np.max(np.reshape(box, [-1,2])[:,0])
    min_y = np.min(np.reshape(box, [-1,2])[:,1])
    max_y = np.max(np.reshape(box, [-1,2])[:,1])
    w = max_x - min_x
    h = max_y - min_y
    new_box = [x*size, y*size, w*size, h*size]
    rmin = min_y
    rmax = max_y
    cmin = min_x
    cmax = max_x
    return (rmin, rmax, cmin, cmax)

def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

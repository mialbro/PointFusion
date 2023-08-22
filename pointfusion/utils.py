import open3d as o3d

import cv2
import numpy as np
import distinctipy

import torch
from torch import nn
from torchvision import transforms

class NormalizePointCloud:
    def __call__(self, points):
        B, C, N = points.size()
        mean = points.mean(dim=2)
        points = points - mean.unsqueeze(2)
        distance = torch.sqrt(torch.sum(torch.abs(points) ** 2, dim=1))
        distance = distance.max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        points = (points / distance)
        points = (points + 1) / 2
        return points.view(C, N)

def draw_corners(image, points):
    # https://github.com/sk-aravind/3D-Bounding-Boxes-From-Monocular-Images/blob/98e9e7caf98edc6a6841d3eac7bd6f62b6866e10/lib/Utils.py#L315
    for point in points:
        cv2.circle(image, (int((point[0])), int((point[1]))), 5, (0, 0, 255), -1)

    points = points.astype(np.int64)
    cv2.line(image, (points[0][0], points[0][1]), (points[2][0], points[2][1]), (0, 0, 255), 3)
    cv2.line(image, (points[4][0], points[4][1]), (points[6][0], points[6][1]), (0, 0, 255), 3)
    cv2.line(image, (points[0][0], points[0][1]), (points[4][0], points[4][1]), (0, 0, 255), 3)
    cv2.line(image, (points[2][0], points[2][1]), (points[6][0], points[6][1]), (0, 0, 255), 3)

    cv2.line(image, (points[1][0], points[1][1]), (points[3][0], points[3][1]), (0, 0, 255), 3)
    cv2.line(image, (points[1][0], points[1][1]), (points[5][0], points[5][1]), (0, 0, 255), 3)
    cv2.line(image, (points[7][0], points[7][1]), (points[3][0], points[3][1]), (0, 0, 255), 3)
    cv2.line(image, (points[7][0], points[7][1]), (points[5][0], points[5][1]), (0, 0, 255), 3)

    for i in range(0, 7, 2):
        cv2.line(image, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), (0, 0, 255), 3)
    
    return image

def bbox_from_mask(mask):
    """get the bounding box corners from a mask"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def get_corners(points=None):
    """get the corners from the transformed pointcloud"""
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif isinstance(points, o3d.geometry.PointCloud):
        pcd = points

    obb = o3d.geometry.OrientedBoundingBox()
    obb = obb.create_from_points(pcd.points)
    corners = np.asarray(obb.get_box_points())
    return corners

def draw(points):
    """draw a pointcloud or a list of pointclouds"""
    clouds = []
    colors = distinctipy.get_colors(len(points))
    for i in range(0, len(points)):
        if isinstance(points[i], np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[i])
            pcd.colors = o3d.utility.Vector3dVector(np.tile(colors[i], (points[i].shape[0], 1)))
            clouds.append(pcd)
        else:
            clouds.append(points[i])
    o3d.visualization.draw_geometries(clouds)

def draw_(geometries):
    """draw a list of geometries"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

def to_pcd(points):
    """convert ndarray to open3d pointcloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def to_lines(corners):
    """convert corners to lines"""
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
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def get_corner_offsets(points, corners):
    corner_offsets = np.zeros((points.shape[0], 8, 3))
    for i in range(0, 8):
        corner_offsets[:, i, :] = points - corners[i]
    return corner_offsets

def corners_from_offsets(points, corner_offsets):
    corners = np.zeros((8, 3))
    for i in range(0, 8):
        corners[i, :] = points[0, :] - corner_offsets[0, i, :]
    return corners
 
import numpy as np
import open3d as o3d
import distinctipy

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
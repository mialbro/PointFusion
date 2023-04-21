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

def get_corners(points):
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
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

def to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def to_obb(corners):
    # compute the center of the box
    center = np.mean(corners, axis=0)

    # compute the lengths of the edges of the box
    deltas = corners - center
    length = np.max(np.abs(deltas[:, 0]))
    width = np.max(np.abs(deltas[:, 1]))
    height = np.max(np.abs(deltas[:, 2]))

    # compute the orientation of the box
    cov = deltas.T @ deltas
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    max_eigenvalue_idx = np.argmax(eigenvalues)
    orientation = eigenvectors[:, max_eigenvalue_idx]


    if orientation.shape == (3,):
        # orientation is a 1D array, reshape it to a 3x3 matrix
        orientation = np.diag([1, 1, 1])
        orientation[:, max_eigenvalue_idx] = eigenvectors[:, max_eigenvalue_idx]

    obb = o3d.geometry.OrientedBoundingBox(center=center, lengths=[length, width, height], R=orientation.reshape(3, 3))

def to_lines(corners):
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
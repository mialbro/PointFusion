import numpy as np

def bbox_from_mask(mask):
    """get the bounding box corners from a mask"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def polygon_from_mask(mask):
    """get the polygon corners from a mask"""
    pass
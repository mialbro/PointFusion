import numpy as np

class Camera:
    def __init__(self):
        pass

    @property
    def intrinsics(self):
        return self._intrinsics
    
    @property
    def extrinsics(self):
        return self._extrinsics
    
    @property
    def projection_matrix(self):
        return np.multiply(self.intrinsics, self.extrinsics)
    
    def project(self, points):
        return np.matmul(self.projection_matrix, points)
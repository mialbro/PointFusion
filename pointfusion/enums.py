from enum import Enum

class Mode(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3

    def __str__(self):
        return self.name

class Modality(Enum):
    RGB = 1
    POINTCLOUD = 2

    def __str__(self):
        return self.name

class ModelName(Enum):
    GlobalFusion = 1
    DenseFusion = 2

    def __str__(self):
        return self.name
    
class LossFcn(Enum):
    GlobalFusion = 1
    DenseFusion = 2

    def __str__(self):
        return self.name
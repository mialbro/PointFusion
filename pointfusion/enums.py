from enum import Enum

class Mode(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3

class Modality(Enum):
    RGB = 1
    POINT_CLOUD = 2

class ModelName(Enum):
    GlobalFusion = 1
    DenseFusion = 2
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
    RGB_POINTCLOUD = 3

    def __str__(self):
        return self.name

class FusionMethod(Enum):
    """Enumerator for fusion strategy"""
    GLOBAL = 1
    DENSE = 2

    def __str__(self) -> str:
        """Get fusion method name
        Returns:
            Fusion name
        """
        return self.name

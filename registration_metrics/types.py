from typing import Union
from nptyping import NDArray, Shape, Float

import numpy as np
import open3d as o3d


__all__ = [
    "Vector3dVector",
    "PointCloud",
    "Tensor",
    "NearestNeighborSearch",

    "NumpyPointCloud3D",
    "NumpyPointCloud4D",
    "PointCloudType",
    "TransformationType",

    "PointCloud_",
    "_to_pc",
    "_to_np",
]


Vector3dVector = o3d.utility.Vector3dVector
PointCloud = o3d.geometry.PointCloud
Tensor = o3d.core.Tensor
NearestNeighborSearch = o3d.core.nns.NearestNeighborSearch

NumpyPointCloud3D = NDArray[Shape["*, 3"], Float]
NumpyPointCloud4D = NDArray[Shape["*, 4"], Float]
PointCloudType = Union[PointCloud, NumpyPointCloud3D, NumpyPointCloud4D]
TransformationType = NDArray[Shape["4, 4"], Float]


def PointCloud_(points):
    """Nice constructor for open3d PointCloud from numpy.ndarray"""
    res = PointCloud()
    res.points = o3d.utility.Vector3dVector(np.copy(points[:, :3]))
    return res


def _to_pc(pc: PointCloudType) -> PointCloud:
    if isinstance(pc, np.ndarray):
        pc = PointCloud_(pc)
    return pc


def _to_np(pc: PointCloudType) -> NumpyPointCloud3D:
    if isinstance(pc, PointCloud):
        pc = np.array(pc.points)
    return pc

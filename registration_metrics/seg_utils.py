import sys
import functools
from typing import Tuple
from nptyping import NDArray, Shape, Bool, Int

import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate

# Local code
from .types import PointCloudType, NumpyPointCloud3D, _to_np
from .models.spvcnn import SPVCNN

from paths import spvnas_dir


__all__ = [
    "SPVCNN_prepare_input",
    "get_model",
    "get_planes",
]


def SPVCNN_prepare_input(
    lidar: PointCloudType,
    prepare_voxel_size: float
) -> Tuple[SparseTensor, NDArray[Shape["*"], Int]]:
    lidar = _to_np(lidar)

    if lidar.shape[1] == 3:
        lidar_4d = np.zeros((len(lidar), 4))
        lidar_4d[:, :3] = lidar
        lidar = lidar_4d

    # Quantize coordinates
    coords = np.round(lidar[:, :3] / prepare_voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar

    # Filter out duplicate points
    coords, indices, inverse = sparse_quantize(
        coords, return_index=True, return_inverse=True
    )

    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(feats[indices], dtype=torch.float)

    inputs = SparseTensor(coords=coords, feats=feats)
    inputs = sparse_collate([inputs])

    return inputs, inverse


def get_model(model_name: str = "spvcnn") -> torch.nn.Module:
    model_name = model_name.lower()

    if model_name in ("semantickitti", "spvcnn"):
        model = SPVCNN(num_classes=3, cr=0.5, pres=0.05, vres=0.05)

        params = torch.load(
            f"{spvnas_dir}/spvcnn-planes-and-ground--cr=0.5-"
            "pretrained=True/max-iou-test.pt",
            map_location=torch.device('cpu')
        )['model']

        model.load_state_dict(params)
    elif model_name == "carla":
        model = SPVCNN(num_classes=3, cr=0.25, pres=0.05, vres=0.05)

        params = torch.load(
            f"{spvnas_dir}/"
            "CARLA-spvcnn-planes-and-ground--cr=0.25-pretrained=False-v2/"
            "max-iou-test.pt",
            map_location=torch.device('cpu')
        )['model']

        model.load_state_dict(params)
    else:
        raise ValueError("Only 'spvcnn' is supported.")

    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        print("Model loaded to CPU.", file=sys.stderr)

    return model


def set_model_decorator(func):
    func.model = None
    func.model_name = None

    @functools.wraps(func)
    def wrapper(cld, prepare_voxel_size, model):
        if isinstance(model, str):
            if func.model_name != model:
                func.model_name = model
                func.model = get_model(func.model_name)
            model = func.model
        elif model is None:
            if func.model is None:
                func.model_name = "spvcnn"
                func.model = get_model(func.model_name)
            model = func.model

        return func(cld, prepare_voxel_size, model)

    wrapper.func = func  # For debug
    return wrapper


@torch.inference_mode()
@set_model_decorator
def get_planes(
    cld: PointCloudType,
    prepare_voxel_size: float,
    model: torch.nn.Module = None
) -> Tuple[NumpyPointCloud3D, NDArray[Shape["*"], Bool]]:
    cld = _to_np(cld)

    model.eval()

    device = next(iter(model.parameters())).device

    inputs, inverse = SPVCNN_prepare_input(cld, prepare_voxel_size)
    inputs = inputs.to(device)
    outputs = model(inputs)

    pred_labels = outputs.argmax(dim=1).cpu().numpy()
    planes_ground_pred_mask = pred_labels[inverse] > 0
    planes_pts_indices = np.where(planes_ground_pred_mask)[0]
    planes_cld = cld[planes_ground_pred_mask]

    return planes_cld, planes_pts_indices

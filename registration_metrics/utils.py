import os
import copy
import random

import numpy as np
import pandas as pd
from nptyping import NDArray, Shape, Float
from easydict import EasyDict
from tqdm.auto import tqdm
import matplotlib
import plotly
import plotly.express as px
import plotly.graph_objs as go

import open3d as o3d

# Import local code
from .types import PointCloud, PointCloudType, _to_np


__all__ = [
    "seed_everything",
    "sparse_quantize",
    "transform_pcd",
    "T_add_translation",
    "prepare_src_tgt_poses",
    "get_erroneous_Ts",
    "cloud_loop",
    "plot_lidar",
    "visualize_clds",
    "sample_planes",

    "trajectory_perturbation",
    "aggregate_pcs",
]


def seed_everything(seed: int = 123, seed_torch=True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)

    if seed_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Random seed set as {seed}")
    return


def sparse_quantize(cld, voxel_size, return_index: bool = False, return_inverse: bool = False):
    coords = np.floor(cld / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    outputs = [cld[indices]]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs


def sample_planes(cld, plane_mask, voxel_size=1, return_idx=False):
    idx = np.arange(len(cld))
    idx_plane = idx[plane_mask.astype(bool)]

    planes = cld[plane_mask.astype(bool)]
    output = sparse_quantize(planes, voxel_size, return_index=return_idx)
    
    if return_idx:
        samples = output[0]
        idx_unique = output[1]
        idx_sample = idx_plane[idx_unique]
        return samples, idx_sample
    
    return output


def transform_pcd(pcd: PointCloudType, transformation: NDArray[Shape["4, 4"], Float]):
    """
    Transforms point cloud `pcd` with provided transformation matrix.
    pcd (point cloud) — n x d (3 first dimensions are given, the rest are ignored)
    transformation — 4 x 4
    """
    if isinstance(pcd, PointCloud):
        return pcd.transform(transformation)

    assert pcd.shape[1] >= 3

    points = np.empty((len(pcd), 4))
    points[:, :3] = pcd[:, :3]
    points[:, 3] = 1

    # Transform
    points = points @ transformation.T

    # Leave only 3 coordinates
    return points[:, :3]


def T_add_translation(T, coord, translation_delta):
    """Adds provided translation to the transformation matrix T"""
    if isinstance(coord, str):
        coord = coord.lower()
        coor2ind = {
            "x": 0,
            "y": 1,
            "z": 2
        }
        assert coord in coor2ind
        coord = coor2ind[coord]
    assert isinstance(coord, int)

    T_err = np.copy(T)
    T_err[coord, 3] += translation_delta
    return T_err


def prepare_src_tgt_poses(*, source, target, spose, tpose, R=None, to_transform=None):
    """
    Prepare source, target and poses before passing them to algorithms:
    1. Leave only 3 coordinates for source and target;
    2. Calculate ground truth transformation between source and target;
    3. Transform target, GT transformation and other provided point clouds with some random matrix R.
    """
    # Leave only 3 coordinates for each point
    source, target = map(
        lambda x: x[:, :3],
        [source, target]
    )

    # Calculate ground truth transformation
    T_gt = np.linalg.inv(tpose) @ spose

    # Add rotational noize
    if R is not None:
        T_gt = R @ T_gt
        target = transform_pcd(target, R)

        if isinstance(to_transform, np.ndarray):
            to_transform = transform_pcd(to_transform, R)
        elif isinstance(to_transform, list):
            to_transform = [transform_pcd(plane, R) for plane in to_transform]
        elif to_transform is not None:
            raise ValueError("to_transform should be np.ndarray or list of them")

    to_return = source, target, T_gt
    if to_transform is not None:
        to_return = *to_return, to_transform

    return to_return


def get_erroneous_Ts(T_gt, deltas, coords):
    """Generate erroneous transformation metrices (translation only supported)"""
    if deltas is None:
        deltas = np.linspace(0.1, 2.0, 20)
    if coords is None:
        coords = "xy"

    for translation_delta in deltas:
        for coord in coords:
            T_err = T_add_translation(T_gt, coord, translation_delta)
            yield translation_delta, coord, T_err


def cloud_loop(*args, data, step):
    """Loop over source, target, source pose, target pose and other provided lists"""
    return tqdm(
        zip(data.clds, data.clds[step:], data.poses,
            data.poses[step:], *args),
        desc="Point clouds", leave=True, total=len(data.poses[step:])
    )


def plot_lidar(
    lidar, labels,
    label_names=None, color=None, color_map=None, sizes=None,
    draw=False, save_as=None, custom_data=None
):
    num_points_for_debug = None

    if color is None:
        if label_names is None:
            d = {l: i for i, l in enumerate(np.unique(labels))}
            color_names = [str(d[l]) for l in labels]
        else:
            color_names = [str(x) for x in np.array(label_names)[labels[:num_points_for_debug]]]
        if color_map is not None:
            color_map = {str(name): color_map[i] for i, name in enumerate(np.unique(color_names))}

    print(color_map)
    if sizes is None:
        size = np.ones_like(labels)
    else:
        size = sizes[labels[:num_points_for_debug]]

    df = pd.DataFrame({
        "x": lidar[:num_points_for_debug, 0],
        "y": lidar[:num_points_for_debug, 1],
        "z": lidar[:num_points_for_debug, 2],
        "color": color_names,
        "size": size,
        "custom_data": custom_data
    })
    custom_data = None if custom_data is None else ["custom_data"]

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', size='size',
                        custom_data=custom_data,
                        color_discrete_map=color_map, size_max=20, opacity=1.)
    fig.update_traces(
        marker=dict(
            sizeref=0.1,
            opacity=0.8,
            line=dict(
                width=0
            )
        )
    )

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.2)),
        legend=dict(
            itemsizing="constant",
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="left",
            x=0
        ),
    )

    fig.update_layout(layout)
    if draw:
        plotly.offline.iplot(fig)
    if save_as is not None:
        fig.write_html(f"{save_as}.html")


def visualize_clds(
    clds, save_as, draw=False, label_names=None
):
    clds = list(map(_to_np, clds))
    for i, cld in enumerate(clds):
        if cld.ndim != 2 or (cld.shape[0] <= 3 and cld.shape[1] <= 3) \
                or (cld.shape[0] != 3 and cld.shape[1] != 3):
            raise ValueError(f"Point cloud with index {i} has not 2 dims, "
                             "or it has less than 4 points, "
                             "or it has not 3 spatial dimenstion for points")
        if cld.shape[0] == 3:
            clds[i] = cld.T

    color_tab = matplotlib.cm.tab10
    colors = ['#a8b3b5'] + [matplotlib.colors.to_hex(c) for c in color_tab.colors]
    color_map = np.array(colors)

    labels = sum(([i] * len(cld) for i, cld in enumerate(clds)), [])

    sizes = np.ones(len(clds)) * 1.5

    return plot_lidar(
        lidar=np.vstack(clds),
        labels=labels,
        label_names=label_names,
        color_map=color_map,
        sizes=sizes,
        draw=draw,
        save_as=save_as,
    )


# *** MoM testing Pipeline *** #
def trajectory_perturbation(Ts, cov=0.1):
    Ts_noise = []
    for T in Ts:
        T_noised = copy.deepcopy(T)
        T_noised[:3, 3] += [np.random.normal(0, cov), np.random.normal(0, cov), 
                            np.random.normal(0, cov)]
        Ts_noise.append(T_noised)
    return Ts_noise


def aggregate_pcs(pcs, Ts):
    pc_map = o3d.geometry.PointCloud()
    for i, pc in enumerate(pcs):
        pc_map += copy.deepcopy(pc).transform(Ts[i])
        
    return pc_map


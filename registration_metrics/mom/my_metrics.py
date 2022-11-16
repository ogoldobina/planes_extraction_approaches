import copy

from numba import jit
from easydict import EasyDict
import numpy as np
import open3d as o3d

from typing import Optional, List, Callable

from ..types import (
    Tensor,
    NearestNeighborSearch,

    TransformationType,
    NumpyPointCloud3D,
    PointCloud,
    _to_np
)

__all__ = ["aggregate_map", "mme", "mpv", "mom", "rpe", "mom_min_nn"]


def aggregate_map(
    pcs: List[PointCloud], ts: List[TransformationType]
) -> PointCloud:
    """
    Build a map from point clouds with their poses
    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    Returns
    -------
    pc_map: o3d.geometry.PointCloud
        Map aggregated from point clouds
    Raises
    ------
    ValueError
        If number of point clouds does not match number of poses
    """
    if len(pcs) != len(ts):
        raise ValueError("Number of point clouds does not match number of poses")

    pc_map = PointCloud()
    for i, pc in enumerate(pcs):
        pc_map += copy.deepcopy(pc).transform(ts[i])

    return pc_map


def _plane_variance_slow(points: NumpyPointCloud3D) -> float:
    """
    Compute plane variance of given points
    Parameters
    ----------
    points: NDArray[(Any, 3), np.float64]
        Point Cloud points
    Returns
    -------
    plane_variance: float
        Points plane variance
    """
    cov = np.cov(points.T)
    eigenvalues = np.linalg.eig(cov)[0]
    return min(eigenvalues)


@jit(nopython=True, cache=True)
def _plane_variance(points: NumpyPointCloud3D) -> float:
    """
    Compute plane variance of given points
    Parameters
    ----------
    points: NDArray[(Any, 3), np.float64]
        Point Cloud points
    Returns
    -------
    plane_variance: float
        Points plane variance
    """
    cov = np.cov(points.T)
    eigenvalues = np.linalg.eig(cov)[0]
    return min(eigenvalues)


@jit(nopython=True, cache=True)
def _plane_variance_broadcasted(pc_map, indices, ranges, ranges_mask):
    """Compute plane variance for several point clouds"""
    metric = [
        _plane_variance(pc_map[indices[begin:end]])
        for begin, end in
        zip(ranges[:-1][ranges_mask], ranges[1:][ranges_mask])
    ]
    return metric


def _entropy_slow(points: NumpyPointCloud3D) -> Optional[float]:
    """
    Compute entropy of given points
    Parameters
    ----------
    points: NDArray[(Any, 3), np.float64]
        Point Cloud points
    Returns
    -------
    entropy: Optional[float]
        Points entropy
    """
    cov = np.cov(points.T)
    det = np.linalg.det(2 * np.pi * np.e * cov)
    if det > 0:
        return 0.5 * np.log(det)

    return None


@jit(nopython=True, cache=True)
def _entropy(points: NumpyPointCloud3D) -> Optional[float]:
    cov = np.cov(points.T)
    det = np.linalg.det(2 * np.pi * np.e * cov)
    if det > 0:
        return 0.5 * np.log(det)
    return None


def _mean_map_metric_slow(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict = None,
    alg: Callable = _plane_variance_slow,
) -> float:
    """
    No-reference metric algorithms helper
    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: EasyDict
        Scene hyperparameters
    alg: Callable
        Metric algorithm basis (e.g., plane variance, entropy)
    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    pc_map = aggregate_map(pcs, ts)

    map_tree = o3d.geometry.KDTreeFlann(pc_map)
    points = np.asarray(pc_map.points)
    metric = []
    for i in range(points.shape[0]):
        point = points[i]
        _, idx, _ = map_tree.search_radius_vector_3d(point, config.normals_estimation.max_radius)
        if len(idx) >= config.metrics.mean_map_metric.min_nn:
            metric_value = alg(points[idx])
            if metric_value is not None:
                metric.append(metric_value)

    return 0.0 if len(metric) == 0 else np.mean(metric)


def _mean_map_metric(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict = None,
    alg: Callable = _plane_variance,  # This is fast!
) -> float:
    return _mean_map_metric_slow(pcs, ts, config, alg)


def _orth_mpv_slow(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict = None,
    orth_list: List[PointCloud] = None,
    return_intermediate: bool = False
):
    """
    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: BaseConfig
        Scene hyperparameters
    orth_list: List[o3d.geometry.PointCloud], default=None
        List of orthogonal planes of the map
    Returns
    -------
    val: float
        The value of MPV computed on orthogonal planes of the map
    """
    pc_map = aggregate_map(pcs, ts)
    map_tree = o3d.geometry.KDTreeFlann(pc_map)
    points = np.asarray(pc_map.points)

    orth_axes_stats = []
    intermediate = []

    for chosen_points in orth_list:
        metric = []
        for i in range(np.asarray(chosen_points).shape[0]):
            point = chosen_points[i]
            _, idx, _ = map_tree.search_radius_vector_3d(point, config.normals_estimation.max_radius)
            if len(idx) >= config.metrics.mom.min_nn:
                metric.append(_plane_variance_slow(points[idx]))

        avg_metric = np.median(metric)
        orth_axes_stats.append(avg_metric)
        intermediate.append(metric)

    orth_axes_stat = np.sum(orth_axes_stats)
    if return_intermediate:
        return orth_axes_stat, intermediate

    return orth_axes_stat


def _orth_mpv(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict,
    orth_list: List[PointCloud],
    return_intermediate: bool = False
):
    pc_map = _to_np(aggregate_map(pcs, ts))

    map_tree = NearestNeighborSearch(Tensor(pc_map))
    map_tree.fixed_radius_index()

    orth_axes_stats = []
    intermediate = []

    for plane in orth_list:
        indices, dists2, ranges = map(
            lambda x: x.numpy(),
            map_tree.fixed_radius_search(
                Tensor(plane), config.normals_estimation.max_radius
            )
        )

        num_neighbours = ranges[1:] - ranges[:-1]
        ranges_mask = num_neighbours >= config.metrics.mom.min_nn

        metric = _plane_variance_broadcasted(pc_map, indices, ranges, ranges_mask)

        avg_metric = np.median(metric)
        orth_axes_stats.append(avg_metric)
        intermediate.append(metric)

    orth_axes_stat = np.sum(orth_axes_stats)
    if return_intermediate:
        return orth_axes_stat, intermediate

    return orth_axes_stat


def mme_slow(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict = None,
) -> float:
    """
    Mean Map Entropy
    A no-reference metric algorithm based on entropy
    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: BaseConfig
        Scene hyperparameters
    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _mean_map_metric(pcs, ts, config, alg=_entropy_slow)


def mme(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict = None,
) -> float:
    return _mean_map_metric(pcs, ts, config, alg=_entropy)


def mpv_slow(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict = None,
) -> float:
    """
    Mean Plane Variance
    A no-reference metric algorithm based on plane variance
    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: BaseConfig
        Scene hyperparameters
    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _mean_map_metric(pcs, ts, config, alg=_plane_variance_slow)


def mpv(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    config: EasyDict = None,
) -> float:
    return _mean_map_metric(pcs, ts, config, alg=_plane_variance)


def mom_slow(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    orth_list: List[PointCloud] = None,
    config: EasyDict = None,
    return_intermediate: bool = False
):
    """
    Mutually Orthogonal Metric
    A no-reference metric algorithm based on MPV on orthogonal planes subset
    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    orth_list: List[o3d.geometry.PointCloud], default=None
        List of orthogonal planes of the map
    config: BaseConfig
        Scene hyperparameters
    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _orth_mpv_slow(pcs, ts, config, orth_list=orth_list,
                          return_intermediate=return_intermediate)


def mom(
    pcs: List[PointCloud],
    ts: List[TransformationType],
    orth_list: List[PointCloud] = None,
    config: EasyDict = None,
    return_intermediate: bool = False
):
    return _orth_mpv(pcs, ts, config, orth_list=orth_list,
                     return_intermediate=return_intermediate)


def rpe(T_gt, T_est):
    seq_len = len(T_gt)
    err = 0
    for i in range(seq_len):
        for j in range(seq_len):
            d_gt = T_gt[i] @ np.linalg.inv(T_gt[j])
            d_est = T_est[i] @ np.linalg.inv(T_est[j])
            # dt = (d_gt @ np.linalg.inv(d_est))[:3, 3]  # The same things
            dt = d_est[:3, 3] - d_gt[:3, 3]
            err += np.linalg.norm(dt) ** 2

    return err

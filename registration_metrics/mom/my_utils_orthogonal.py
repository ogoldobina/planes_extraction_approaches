import numpy as np
import open3d as o3d
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from easydict import EasyDict
from typing import List
from nptyping import NDArray, Shape

from ..types import PointCloud, PointCloudType, _to_pc, Vector3dVector, NumpyPointCloud3D
from ..seg_utils import get_planes


__all__ = [
    "pc_select_by_indices",
    "filter_planes",
    "extract_planes",
    "estimate_normals",
    "extract_orthogonal_subsets",
    "extract_orthogonal_subsets_indices",
    "extract_orthogonal_subsets_indices_from_planes",
]


def pc_select_by_indices(pc, selected_indices):
    if isinstance(pc, np.ndarray):
        return pc[selected_indices]

    res_pc = PointCloud()
    res_pc.points = Vector3dVector(np.array(pc.points)[selected_indices])
    if pc.has_normals():
        res_pc.normals = Vector3dVector(np.array(pc.normals)[selected_indices])
    return res_pc


def filter_planes(pc, knn_rad):
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    if pc.has_normals():
        main_normals = np.asarray(pc.normals)
    normals = []
    lambdas = []
    new_points = []
    indices = []
    # TODO: Add mutable tqdm bar
    for i in range(points.shape[0]):
        point = points[i]
        _, idx, _ = pc_tree.search_radius_vector_3d(point, knn_rad)
        if len(idx) > 3:
            cov = np.cov(points[idx].T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            if 100 * eigenvalues[0] < eigenvalues[1]:
                if pc.has_normals():
                    normals.append(main_normals[i])
                lambdas.append(eigenvalues[0])
                new_points.append(point)
                indices.append(i)

    plane_pc = o3d.geometry.PointCloud()
    plane_pc.points = o3d.utility.Vector3dVector(new_points)
    if pc.has_normals():
        plane_pc.normals = o3d.utility.Vector3dVector(normals)

    return plane_pc, np.array(indices)


def extract_planes(pc: PointCloudType, config: EasyDict) -> NumpyPointCloud3D:
    algo = config.planes_segmentation.algo
    assert algo in ("svd_filter", "spvcnn")

    pc = _to_pc(pc)

    if algo == "svd_filter":
        knn_rad = config.normals_estimation.max_radius
        return filter_planes(pc=pc, knn_rad=knn_rad)
    if algo == "spvcnn":
        voxel_size = config.planes_segmentation.spvcnn.voxel_size
        _, planes_pts_indices = get_planes(
            cld=pc, prepare_voxel_size=voxel_size, model=config.dataset.name
        )
        return pc_select_by_indices(pc, planes_pts_indices), planes_pts_indices


def estimate_normals(
    pc: PointCloudType,
    *,
    selected_indices: List[int] = None,
    config: EasyDict
) -> PointCloud:
    pc = _to_pc(pc)

    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.normals_estimation.max_radius,
            max_nn=config.normals_estimation.max_nn
        )
    )

    if selected_indices is None:
        return pc
    return pc_select_by_indices(pc, selected_indices)


def _calc_mean_normal(normals: NDArray[Shape["*, 3"], np.float64]):
    scalar_prod = normals @ normals[0]
    normals[scalar_prod < 0] *= -1

    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)

    return mean_normal


def _filter_clusters(clustering, normals, min_clust_size):
    n_clusters = np.unique(clustering.labels_).shape[0]
    labels = clustering.labels_
    huge_clusters = []
    cluster_means, cluster_means_ind = [], []

    for i in range(n_clusters):
        ind = np.where(labels == i)
        if ind[0].shape[0] > min_clust_size:
            huge_clusters.append(i)
            cluster_means.append(_calc_mean_normal(np.vstack(normals)[ind]))

            cluster_means_ind.append(i)

    cluster_means = np.vstack(cluster_means)

    return cluster_means, cluster_means_ind


def _find_max_clique(labels, cluster_means, cluster_means_ind, eps=1e-1):
    N = cluster_means.shape[0]
    adj_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            x = np.abs(np.dot(cluster_means[i], cluster_means[j]))
            if x < eps:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    D = nx.Graph(adj_matrix)
    x = nx.algorithms.clique.find_cliques(D)

    full_cliques_size = []
    full_cliques = []
    for clique in x:
        if len(clique) > 2:
            amount = 0
            for j in clique:
                amount += np.sum(labels == cluster_means_ind[j])
            full_cliques_size.append(amount)
            full_cliques.append(clique)

    if len(full_cliques) == 0:
        raise ValueError("Length of full_cliques == 0")

    max_ind = full_cliques_size.index(max(full_cliques_size))
    return full_cliques[max_ind]


def extract_orthogonal_subsets_indices_from_planes(planes: PointCloud, config: EasyDict, details=None):
    assert planes.has_normals(), "Point cloud has no normals."

    normals = np.asarray(planes.normals)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=config.orth_planes_extraction.dist_threshold,
        compute_full_tree=True, linkage='complete', affinity=config.orth_planes_extraction.affinity
    ).fit(normals)

    if details is not None:
        details["AC_num_clusters"] = clustering.n_clusters_

    cluster_means, cluster_means_ind = _filter_clusters(
        clustering, normals, min_clust_size=config.orth_planes_extraction.min_clust_size
    )

    labels = clustering.labels_
    max_clique = _find_max_clique(labels, cluster_means, cluster_means_ind,
                                  eps=config.orth_planes_extraction.dist_threshold)

    orth_subset = [
        np.where(labels == cluster_means_ind[i])[0] for i in max_clique
    ]

    return orth_subset


def extract_orthogonal_subsets_indices(pc: PointCloud, config: EasyDict, details=None):
    assert pc.has_normals(), "Point cloud has no normals."

    planes, indices = filter_planes(pc, config.normals_estimation.max_radius)
    orth_indices = extract_orthogonal_subsets_indices_from_planes(planes, config, details)

    return [indices[orth_idx] for orth_idx in orth_indices]


def extract_orthogonal_subsets(pc: PointCloud, config: EasyDict, details=None):
    assert pc.has_normals(), "Point cloud has no normals."

    orth_indices = extract_orthogonal_subsets_indices(pc, config, details)
    points = np.array(pc.points)
    return [points[orth_idx] for orth_idx in orth_indices], orth_indices

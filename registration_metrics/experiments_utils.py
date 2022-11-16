import copy
from pprint import pprint
from time import time
from collections import defaultdict
from argparse import Namespace
from functools import partial
from typing import List
import yaml

import tqdm
import numpy as np
import open3d as o3d
from easydict import EasyDict

from .mom.my_utils_orthogonal import (
    pc_select_by_indices,
    estimate_normals,
    extract_planes,
    extract_orthogonal_subsets_indices_from_planes,
)
from .mom.my_metrics import rpe, mom  # mom_min_nn as mom
from .utils import sparse_quantize, trajectory_perturbation
from .types import PointCloudType, TransformationType, _to_pc, _to_np

from paths import PROJECT_DIR


__all__ = [
    "load_config",
    "new_value_if_none",
    "get_filename_fmt",
    "prepare_ts",
    "prepare_data",
    "sampling_pipeline",
]


def load_config(filename, verbose=True,
                config_dir=f"{PROJECT_DIR}/configs/mom_pipeline"):
    with open(f'{config_dir}/{filename}.yaml', 'r') as file:
        config = EasyDict(yaml.safe_load(file))

    if verbose:
        # print(json.dumps(config, indent=2))
        pprint(config)

    return config


def new_value_if_none(old_val, new_val):
    return new_val if old_val is None else old_val


def recursive_dict_all_keys(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            for x in recursive_dict_all_keys(v):
                yield [k] + x
        else:
            yield [k]


def get_filename_fmt(config: EasyDict):
    filename_fmt = {}
    for long_key in recursive_dict_all_keys(config):
        key_name = '__'.join(long_key)
        param = config
        for key in long_key:
            param = param[key]
        filename_fmt[key_name] = param
    return filename_fmt


def prepare_ts(Ts, orth_planes_cld_ind):
    """Transformations from frame2world to frame2frame_orth_planes"""
    T_orth_planes_inv = np.linalg.inv(Ts[orth_planes_cld_ind])
    Ts = [T_orth_planes_inv @ T for T in Ts]
    return Ts


def construct_small2big(inv):
    small2big = defaultdict(list)
    for big_pc_ind, small_pc_ind in enumerate(inv):
        small2big[small_pc_ind].append(big_pc_ind)
    return small2big


def upsample_indices(indices, small2big):
    if not isinstance(indices, list) or len(indices) > 0 and isinstance(indices[0], int):
        return np.array(sum(map(small2big.__getitem__, indices), []))
    return [upsample_indices(idx, small2big) for idx in indices]


def downsample_pc(pc, voxel_size):
    np_pc_d, ind, inv = sparse_quantize(
        _to_np(pc), voxel_size, return_index=True, return_inverse=True
    )
    pc_d = pc_select_by_indices(pc, ind)
    s2b = construct_small2big(inv)

    return Namespace(
        pc=pc_d,
        s2b=s2b,
        upsample=partial(upsample_indices, small2big=s2b)
    )


def prepare_data(pcs, T_gt, config: EasyDict):
    start_time = time()

    # 0. Data preparation
    t0s = time()
    details = {}

    orth_planes_cld_from_pc = config.orth_planes_extraction.from_pc
    assert orth_planes_cld_from_pc in ("first", "middle")
    if orth_planes_cld_from_pc == "first":
        orth_planes_cld_ind = 0
    elif orth_planes_cld_from_pc == "middle":
        orth_planes_cld_ind = len(pcs) // 2

    # Cast pcs to PointCloud
    pcs = list(map(_to_pc, copy.deepcopy(pcs)))

    # Transformations from frame2world to frame2frame_orth_planes
    T_gt = prepare_ts(T_gt, orth_planes_cld_ind)

    # Get pc to extract orthogonal planes
    pc = pcs[orth_planes_cld_ind]  # _to_pc()
    t0e = time()

    # 0.1. Refine transformations
    t01s = time()
    if "dataset" in config and config.dataset.refine_ts:
        target = pc
        T_gt_refined = []
        for src_i, (source, T_init) in enumerate(zip(pcs, T_gt)):
            if src_i == orth_planes_cld_ind:
                T_gt_refined.append(T_init)
                continue
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, 0.2, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            T_gt_refined.append(reg_p2p.transformation)
        T_gt = T_gt_refined
    t01e = time()

    #  1. Estimate normals
    t1s = time()
    pc = estimate_normals(pc, config=config)
    t1e = time()

    #  2. Extract planes (and indices of points)
    t2s = time()
    pc_planes, pc_planes_indices = extract_planes(pc, config)
    t2e = time()

    #   2.1. Downsampling
    t21s = time()
    conf_down = config.orth_planes_extraction.downsample
    if conf_down.use:
        d1 = downsample_pc(pc_planes, conf_down.voxel_size)
        # d2 = downsample_pc(d1.pc, 0.6)
        orth_extractor_input = d1.pc
    else:
        orth_extractor_input = pc_planes
    t21e = time()

    #  3. Extract orthogonal planes (indices of points)
    t3s = time()
    orth_planes_indices_d = extract_orthogonal_subsets_indices_from_planes(
        orth_extractor_input, config, details
    )
    t3e = time()

    #   3.1. Upsampling
    t31s = time()
    if conf_down.use and conf_down.upsample:
        # orth_planes_indices_d1 = d2.upsample(orth_planes_indices_d2)
        orth_planes_indices = d1.upsample(orth_planes_indices_d)

        orth_planes_indices_chosen_level = orth_planes_indices
        pc_planes_chosen_level = pc_planes
    else:
        orth_planes_indices_chosen_level = orth_planes_indices_d
        pc_planes_chosen_level = orth_extractor_input
    t31e = time()

    #  4. Reconstruct points from indices
    t4s = time()
    orth_planes = [
        pc_select_by_indices(_to_np(pc_planes_chosen_level), orth_plane_idx)
        for orth_plane_idx in orth_planes_indices_chosen_level
    ]
    t4e = time()

    # 5. Extract planes from all point clouds if needed
    conf_mom = config.metrics.mom
    assert conf_mom.input in ("whole", "planes")
    t5s = time()
    if conf_mom.input == "planes":
        pcs = [extract_planes(pc, config)[0] for pc in pcs]
    elif conf_mom.input == "whole":
        pass
    t5e = time()

    end_time = time()

    details["time_dict"] = {
        "Data preparation": t0e - t0s,
        "Refine transformations": t01e - t01s,
        "Normals estimation": t1e - t1s,
        "Planes extraction": t2e - t2s,
        "Downsampling": t21e - t21s,
        "Orthogonal planes extraction": t3e - t3s,
        "Upsampling": t31e - t31s,
        "Getting points by indices": t4e - t4s,
        "Point clouds planes extraction": t5e - t5s,
        "Total time": end_time - start_time
    }

    return pcs, T_gt, orth_planes, orth_planes_cld_ind, details


def sampling_pipeline(
    *,
    pcs: List[PointCloudType],
    T_gt: List[TransformationType],
    map_tips: EasyDict,
    N_samples: int = 20,
    cov_scaler: int = 10,
    orth_planes_cld_ind: int,
    verbose="auto"
) -> EasyDict:  # Tuple[NDArray[Shape["*"], Float], NDArray[Shape["*"], Float]]:
    nr = []
    fr = []

    samples_iter = range(N_samples)
    cov_iter = range(cov_scaler)
    if verbose is not False:
        if verbose == "auto":
            tqdm_wrapper = tqdm.auto.tqdm
        else:
            tqdm_wrapper = tqdm.tqdm
        samples_iter = tqdm_wrapper(samples_iter, desc="Samples", leave=False)
        cov_iter = tqdm_wrapper(cov_iter, desc="Covs", leave=False)

    mom_calc_time_avg = 0
    for _ in samples_iter:
        for cov_i in cov_iter:
            cov = 0.1 * (cov_i + 1) / len(pcs)
            T_pert = trajectory_perturbation(T_gt, cov=cov)

            T_pert = prepare_ts(T_pert, orth_planes_cld_ind)

            rpe_val = rpe(T_gt, T_pert)
            mom_time_start = time()
            mom_val = mom(pcs, T_pert, map_tips.orth_list, map_tips.config)
            mom_time_end = time()
            mom_calc_time_avg += mom_time_end - mom_time_start

            fr.append(rpe_val)
            nr.append(mom_val)
    mom_calc_time_avg /= N_samples * cov_scaler

    return EasyDict(nr=nr, fr=fr, mom_calc_time_avg=mom_calc_time_avg)

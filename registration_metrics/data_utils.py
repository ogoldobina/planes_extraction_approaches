import os
import json
import pickle
from easydict import EasyDict

import numpy as np
import open3d as o3d

from .utils import transform_pcd
from paths import ROOT_DIR


__all__ = [
    "parse_calibration",
    "parse_poses",
    "CARLA_dataset",
    "SemanticKITTI_dataset"
]


def parse_calibration(filename):
    """ read calibration file with given filename

    Returns
    -------
    dict
      Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

    Returns
    -------
    list
        list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


class SemanticKITTI_dataset:
    WORLD = "world"
    FRAME = "frame"

    def __init__(
        self,
        *,
        dir_path=f'{ROOT_DIR}/PlaneKITTI_large_planes_and_ground/00_train/',
        start_id=0,
        end_id=100,
    ):

        self.dir_path = dir_path
        self.start_id = start_id
        self.end_id = end_id

        self.clds = []
        self.labels = []
        self.poses = []

        frames_gen = self.dir_frames_gen
        for frame in frames_gen():
            self.poses.append(frame.T)
            self.clds.append(frame.cld)
            self.labels.append(frame.label)

    def dir_frames_gen(self):
        files = sorted(
            os.listdir(os.path.join(self.dir_path, 'velodyne')))
        files = [
            os.path.join(self.dir_path, 'velodyne', x) for x in files
        ]
        
        poses_file = os.path.join(self.dir_path, 'poses.txt')
        calib_file =  os.path.join(self.dir_path, 'calib.txt')
        calibration = parse_calibration(calib_file)
        poses = parse_poses(poses_file, calibration)
        
        for ind in range(self.start_id, self.end_id):
            cld_file = files[ind]
            label_file = files[ind].replace('velodyne', 'labels').replace('.bin', '.npy')
            
            cld = np.fromfile(cld_file, dtype=np.float32).reshape(-1, 4)
            label = np.load(label_file).astype(np.int32)
            pose = poses[ind]
            
            frame = EasyDict({
                'cld': cld,
                'label': label,
                'T': pose
            })
            yield frame

    def __getitem__(self, i):
        return EasyDict({
            'cld'    : self.clds[i],
            'pose'   : self.poses[i],
            'label'  : self.labels[i]
        })

    def __len__(self):
        return len(self.clds)


class CARLA_dataset:
    WORLD = "world"
    FRAME = "frame"

    def __init__(
        self,
        *,
        dir_path=f'{ROOT_DIR}/carla_semantic/seq5_withlabels/point_clouds',
        filename=None,  # f'{ROOT_DIR}/carla_semantic/seq5_withlabels/orig.csv',
        start_id=0,
        end_id=100,
        dataset_coords_type,
        dataset_transformations_type="frame2world",
        wanted_coords_type="frame"
    ):
        assert (dir_path is None) != (filename is None)
        assert dataset_coords_type in (self.FRAME, self.WORLD)
        assert dataset_transformations_type in ("frame2world",)
        assert wanted_coords_type in (self.FRAME, self.WORLD)

        self.dir_path = dir_path
        self.filename = filename
        self.start_id = start_id
        self.end_id = end_id
        self.dataset_coords_type = dataset_coords_type
        self.dataset_transformations_type = dataset_transformations_type
        self.wanted_coords_type = wanted_coords_type

        self.poses = []
        self.clds = []
        self.labels = []
        self.obj_idx = []

        frames_gen = self.dir_frames_gen if filename is None else self.csv_file_frames_gen
        for frame in frames_gen():
            frame.cld = self.prepare_cld(frame.cld, frame.T)

            self.poses.append(frame.T)
            self.clds.append(frame.cld)
            self.labels.append(frame.label)
            self.obj_idx.append(frame.obj_idx)

    @staticmethod
    def parse_line(elem):
        transformation, lidar = elem.split(",[[[")

        location, rotation = transformation.split("Location(")[1].split("), Rotation(")
        rotation = rotation[:-2]
        location, rotation = map(lambda x: eval(f"dict({x})"), [location, rotation])
        T = CARLA_dataset.get_T(location, rotation)

        lidar = json.loads("[[[" + lidar)
        cld, label, obj_idx = map(np.array, zip(*lidar))

        return EasyDict(cld=cld, T=T, label=label, obj_idx=obj_idx)

    def csv_file_frames_gen(self):
        with open(self.filename, mode='r') as csv_file:
            for line_number, elem in enumerate(csv_file):
                if line_number >= self.end_id:
                    break
                if line_number < self.start_id:
                    continue

                yield CARLA_dataset.parse_line(elem)

    def dir_frames_gen(self):
        for ind in range(self.start_id, self.end_id):
            file = os.path.join(self.dir_path, f"{ind:06}.bin")
            with open(file, "rb") as fin:
                frame = EasyDict(pickle.load(fin))
            yield frame

    def prepare_cld(self, cld, T):
        if (
            self.dataset_coords_type == self.WORLD and
            self.wanted_coords_type == self.FRAME
        ):
            cld = transform_pcd(cld, np.linalg.inv(T))
        elif (
            self.dataset_coords_type == self.FRAME and
            self.wanted_coords_type == self.WORLD
        ):
            cld = transform_pcd(cld, T)

        return cld

    def __getitem__(self, i):
        return EasyDict({
            'cld'    : self.clds[i],
            'pose'   : self.poses[i],
            'label'  : self.labels[i],
            'obj_id' : self.obj_idx[i]
        })

    def __len__(self):
        return len(self.clds)

    @staticmethod
    def get_T(location, rotation):
        t = np.array([location['x'], location['y'], location['z']])
        euler_rad = np.array([0, 0, rotation['yaw']]) / 180 * np.pi
        R = o3d.geometry.get_rotation_matrix_from_xyz(euler_rad)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

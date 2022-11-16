import os
from pathlib import Path


PROJECT_DIR = Path(os.path.abspath(__file__)).parents[0]
ROOT_DIR = PROJECT_DIR.parents[0]  # Project root directory

plane_kitti_dir = os.path.join(ROOT_DIR, "PlaneKITTI_large_planes_and_ground/00_train/")
plane_carla_dir = os.path.join(ROOT_DIR, "carla_semantic/seq5_withlabels/point_clouds")

# SPVNAS dir
spvnas_dir = os.path.join(PROJECT_DIR, "checkpoints/spvnas")
# spvnas_dir = os.path.join(ROOT_DIR, "checkpoints/spvnas")

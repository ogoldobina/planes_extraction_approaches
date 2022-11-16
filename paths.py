import os
from pathlib import Path


PROJECT_DIR = Path(os.path.abspath(__file__)).parents[0]
ROOT_DIR = PROJECT_DIR.parents[0]  # Project root directory

plane_kitti_dir = os.path.join(ROOT_DIR, "PlaneKITTI_large/00_test")

# SPVNAS dir
spvnas_dir = os.path.join(ROOT_DIR, "spvnas")

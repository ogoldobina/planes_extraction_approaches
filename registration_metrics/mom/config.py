from easydict import EasyDict
from map_metrics.map_metrics.config import CustomConfig


class BaseConfig(CustomConfig):
    def __init__(self, *, min_knn=3, knn_rad=2.0, max_nn=30, min_clust_size=10, voxel_size=0.25, ac_affinity='cosine'):
        super().__init__(min_knn, knn_rad, max_nn, min_clust_size)

        self.min_knn = min_knn
        self.knn_rad = knn_rad
        self.max_nn = max_nn
        self.min_clust_size = min_clust_size
        self.voxel_size = voxel_size
        self.ac_affinity = ac_affinity

        self.MIN_KNN = min_knn
        self.KNN_RAD = knn_rad
        self.MAX_NN = max_nn
        self.MIN_CLUST_SIZE = min_clust_size
        self.VOXEL_SIZE = voxel_size
        self.AC_AFFINITY = ac_affinity

    def __repr__(self):
        return (f"{type(self).__name__}("
                f"min_knn={self.min_knn}, "
                f"knn_rad={self.knn_rad}, "
                f"max_nn={self.max_nn}, "
                f"min_clust_size={self.min_clust_size}, "
                f"voxel_size={self.voxel_size}, "
                f"ac_affinity={self.ac_affinity})")

    def state(self):
        return EasyDict(
            min_knn=self.min_knn,
            knn_rad=self.knn_rad,
            max_nn=self.max_nn,
            min_clust_size=self.min_clust_size,
            voxel_size=self.voxel_size,
            ac_affinity=self.ac_affinity
        )


LidarConfig = BaseConfig(
    knn_rad=0.1,
    min_knn=3,
    max_nn=5,
    min_clust_size=10000,
)

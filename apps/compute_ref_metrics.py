"""Compute reference metrics of data in dataset_path w.r.t themselves."""

import numpy as np

from utils.configs import GlobalConfig
from utils.metrics import print_metrics
from utils.metrics.metric import wasserstein_distances

if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='configs/exp/base.yaml')
    data = np.load(global_config.dataset_path)
    data1 = data[:len(data) // 2][:5000]
    data2 = data[len(data) // 2:][:5000]
    unit_component_size = global_config.metrics.unit_component_size
    connectivity = global_config.metrics.connectivity
    metrics = wasserstein_distances(data1, data2,
                                    unit_component_size=unit_component_size,
                                    connectivity=connectivity,
                                    save_boxes_path='perfect_metrics.png')[0]
    print_metrics(metrics)

"""Compute reference metrics of data in dataset_path w.r.t themselves."""

import numpy as np

from gan_facies.metrics import print_metrics
from gan_facies.metrics.metric import wasserstein_distances
from gan_facies.utils.configs import GlobalConfig

if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='gan_facies/configs/exp/base.yaml')
    data = np.load(global_config.dataset_path)
    data1 = data[:len(data) // 2][:5000]
    data2 = data[len(data) // 2:][:5000]
    unit_component_size = global_config.metrics.unit_component_size
    connectivity = global_config.metrics.connectivity
    metrics = wasserstein_distances(data1, data2,
                                    unit_component_size=unit_component_size,
                                    connectivity=connectivity,
                                    save_boxes_path='perfect_metrics.png')[0]
    print_metrics(metrics=(metrics, {}))

"""Wasserstein metric functions."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance as w_dist

from utils.metrics.indicators import compute_indicators
from utils.metrics.visualize import plot_boxes

IndicatorsList = List[Dict[str, List[float]]]


def wasserstein_distances(data1: Union[np.ndarray, IndicatorsList],
                          data2: Union[np.ndarray, IndicatorsList],
                          connectivity: Optional[int] = None,
                          unit_component_size: int = 1,
                          save_boxes_path: Optional[str] = None
                          ) -> Tuple[Dict[str, float],
                                     Tuple[IndicatorsList, IndicatorsList]]:
    """Compute Wasserstein metrics of data1 w.r.t reference data2.

    Parameters
    ----------
    data1 : np.ndarray, dtype=np.uint8 or List[Dict[str, List[float]]]
        Batch of images of dims 3 (2D images) or 4 (3D images).
        Shape (n_images, [depth,] height, with).
        OR
        Output of utils.metrics.compute_indicators if already computed.
    data2 : np.ndarray, dtype=np.uint8 or List[Dict[str, List[float]]]
        Batch of images of dims 3 (2D images) or 4 (3D images).
        Shape (n_images, [depth,] height, with).
        OR
        Output of utils.metrics.compute_indicators if already computed.
    connectivity : int
        Either 1 for 4-neighborhood (6-neighborhood in 3D)
        or 2 for 8-neighborhood (18-neighborhood in 3D)
        or 3 for 26-neighborhood in 3D (connectivity must be < 3 in 2D).
        By default it is set to 2 for 2D images, 3 for 3D images.
    unit_component_size : int, optional
        Maximum size to consider a component as a unit component.
        By default, 1.
    save_boxes_path : str or None, optional
        If not None or empty, save the boxes of the metrics to this path.
        By default, None.

    Raises
    ------
    TypeError
        If data1 or data2 are not of type np.ndarray or list.

    Returns
    -------
    metrics : Dict[str, float]
        List of Wasserstein metrics from data1 w.r.t data2. The keys
        are indicators name (with corresponding class) and the value is
        the Wasserstein distance between values of the two classes.
        Lower values means better similarity.
    indicators_list_1 : List[Dict[str, List[float]]]
        List of indicators from data1.
    indicators_list_2 : List[Dict[str, List[float]]]
        List of indicators from data2.
    """
    # indictors: list of size n_classes (- 1) of dicts :
    # {indicator_name (str): indicator_values (1D np.ndarray)}
    if isinstance(data1, np.ndarray):
        indicators_list_1: list = compute_indicators(
            data1, connectivity=connectivity,
            unit_component_size=unit_component_size)
    elif isinstance(data1, list):
        indicators_list_1 = data1
    else:
        raise TypeError('data1 must be either a np.ndarray (data) or a list '
                        '(output of "compute_indicators").')

    if isinstance(data2, np.ndarray):
        indicators_list_2: list = compute_indicators(
            data2, connectivity=connectivity,
            unit_component_size=unit_component_size)
    elif isinstance(data2, list):
        indicators_list_2 = data2
    else:
        raise TypeError('data2 must be either a np.ndarray (data) or a list '
                        '(output of "compute_indicators").')

    if len(indicators_list_1) != len(indicators_list_2):
        print('Warning: data1 and data2 have different number of classes. '
              f'Found {len(indicators_list_1)} and {len(indicators_list_2)}. '
              'This behavior is only expected for bad models '
              '(e.g. during unit tests). The indicators of missing classes '
              'will be copy from the last one.')
        len_1, len_2 = len(indicators_list_1), len(indicators_list_2)
        if len_1 > len_2:
            indicators_list_2 = (indicators_list_2
                                 + [indicators_list_2[-1]] * (len_1-len_2))
        else:
            indicators_list_1 = (indicators_list_1
                                 + [indicators_list_1[-1]] * (len_2-len_1))
    metrics = {}
    for i, (indicators_1, indicators_2) in enumerate(zip(indicators_list_1,
                                                         indicators_list_2)):
        for ind_name in indicators_1:
            metric_name = ind_name + f'_cls_{i + 1}'
            dist = w_dist(indicators_1[ind_name],
                          indicators_2[ind_name])
            # Normalize wasserstein distance by the expected value
            mean = np.mean(indicators_1[ind_name] + indicators_2[ind_name])
            metrics[metric_name] = dist / (mean + 1e-5)

    if save_boxes_path:
        n_classes = len(indicators_list_1)
        indicators_list: List[Dict[str, List[float]]] = [{}] * (2 * n_classes)
        indicators_list[::2] = indicators_list_1
        indicators_list[1::2] = indicators_list_2
        indicator_names = [f'class_{i // 2 + 1}' for i in range(2*n_classes)]
        indicator_colors = ['lightblue', 'lightgreen'] * n_classes
        plt.figure(figsize=(30, 30))
        plot_boxes(indicators_list, indicator_names=indicator_names,
                   indicator_colors=indicator_colors)
        plt.savefig(save_boxes_path)
    return metrics, (indicators_list_1, indicators_list_2)

# Example of use:
# if __name__ == '__main__':
#     data = np.load('datasets/gansim_small.npy')
#     data1 = data[:len(data) // 2]
#     data2 = data[len(data) // 2:]
#     print(wasserstein_distances(data1, data2,
#                                 unit_component_size=4,
#                                 save_boxes_path='perfect_metrics.png')[0])

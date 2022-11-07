"""Wasserstein metric functions."""

import json
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table
from scipy.stats import wasserstein_distance as w_dist
from torch import nn

from gan_facies.data.data_loader import DistributedDataLoader
from gan_facies.metrics.indicators import compute_indicators
from gan_facies.metrics.tools import (MetricsType, get_reference_indicators,
                                      save_metrics, split_wass_dists)
from gan_facies.metrics.visualize import plot_boxes
from gan_facies.utils.conditioning import generate_pixel_maps
from gan_facies.utils.configs import ConfigType

IndicatorsList = List[Dict[str, List[float]]]


def wasserstein_distances(data1: Union[np.ndarray, IndicatorsList],
                          data2: Union[np.ndarray, IndicatorsList],
                          connectivity: Optional[int],
                          unit_component_size: int,
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
    connectivity : int or None
        Either 1 for 4-neighborhood (6-neighborhood in 3D)
        or 2 for 8-neighborhood (18-neighborhood in 3D)
        or 3 for 26-neighborhood in 3D (connectivity must be < 3 in 2D).
        If None, it is set to 2 for 2D images, 3 for 3D images.
    unit_component_size : int
        Maximum size to consider a component as a unit component.
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
              'will be arbitrarily copy from the first values.')
        len_1, len_2 = len(indicators_list_1), len(indicators_list_2)
        if len_1 > len_2:
            indicators_list_2 = (indicators_list_2 + [indicators_list_1[0]] *
                                 (len_1-len_2))
        else:
            indicators_list_1 = (indicators_list_1 + [indicators_list_2[0]] *
                                 (len_2-len_1))
    metrics = {}
    for i, (indicators_1, indicators_2
            ) in enumerate(zip(indicators_list_1, indicators_list_2)):
        for ind_name in indicators_1:
            metric_name = ind_name + f'_cls_{i + 1}'
            dist = w_dist(indicators_1[ind_name], indicators_2[ind_name])
            # Normalize wasserstein distance by the expected value
            mean = np.mean(indicators_1[ind_name] + indicators_2[ind_name])
            metrics[metric_name] = dist / (mean+1e-5)
    metrics_list = list(metrics.values())
    metrics['global'] = np.mean(metrics_list) if metrics_list else 0.0
    if save_boxes_path:
        n_classes = len(indicators_list_1)
        indicators_list: List[Dict[str, List[float]]] = [{}] * (2*n_classes)
        indicators_list[::2] = indicators_list_1
        indicators_list[1::2] = indicators_list_2
        indicator_names = [f'class_{i // 2 + 1}' for i in range(2 * n_classes)]
        indicator_colors = ['lightblue', 'lightgreen'] * n_classes
        plt.figure(figsize=(30, 30))
        plot_boxes(indicators_list, indicator_names=indicator_names,
                   indicator_colors=indicator_colors)
        plt.savefig(save_boxes_path)
    return metrics, (indicators_list_1, indicators_list_2)


def compute_save_indicators(data_loader: DistributedDataLoader,
                            config: ConfigType) -> str:
    """Compute and save indicators from data loader and configuration.

    The indicators are saved in a json file and the path is inferred
    from the configuration and data if connectivity is None. The output
    path is returned.

    Parameters
    ----------
    data_loader: DistributedDataLoader
        Data loader to compute indicators.
    config: ConfigType
        Configuration of the experiment.

    Returns
    -------
    indicators_path: str
        Path to the indicators file.
    """
    rprint = Console().print
    data_iter = iter(data_loader.loader())
    for batch in data_iter:
        data = batch[0]
        default_connectivity = data.ndim - 2
        train_data = data.detach().cpu().numpy()
        break
    torch.cuda.empty_cache()  # Free GPU memory
    data_size = config.model.data_size
    connectivity = config.metrics.connectivity or default_connectivity
    unit_component_size = config.metrics.unit_component_size
    dataset_body, _ = os.path.splitext(config.dataset_path)

    indicators_path = (f'{dataset_body}_ds{data_size}_co'
                       f'{connectivity}_us{unit_component_size}_'
                       'indicators.json')

    overwrite = config.metrics.overwrite_indicators
    if overwrite or not osp.exists(indicators_path):
        rprint(
            'Compute indicators from training dataset (used for '
            'metrics)...', style='bold cyan', highlight=False)
        for batch in data_iter:
            if len(train_data) >= 2000:
                # Enough data to compute indicators
                break
            data = batch[0]
            data = data.detach().cpu().numpy()
            torch.cuda.empty_cache()  # Free GPU memory
            train_data = np.concatenate([train_data, data], axis=0)

        # Binarize data (channel first)
        train_data = np.argmax(train_data, axis=1)
        train_data = train_data.astype(np.uint8)
        # Compute dataset indicators
        connectivity = config.metrics.connectivity
        unit_component_size = config.metrics.unit_component_size
        indicators_list = compute_indicators(
            train_data, connectivity=connectivity,
            unit_component_size=unit_component_size)
        # Save indicators in the same folder as the dataset
        indicators_dir, _ = os.path.split(indicators_path)
        os.makedirs(indicators_dir, exist_ok=True)
        with open(indicators_path, 'w', encoding='utf-8') as file_out:
            json.dump(indicators_list, file_out, separators=(',', ':'),
                      sort_keys=False, indent=4)
    else:
        rprint(
            'Indicators from training set already found at '
            f'{indicators_path}, they are re-used to compute metrics. '
            'To recompute indicators, switch '
            'config.metrics.overwrite_indicators to True.', style='cyan',
            highlight=False)
    return indicators_path


def evaluate(gen: nn.Module, config: ConfigType, training: bool, step: int,
             indicators_path: Optional[str] = None, save_json: bool = False,
             save_csv: bool = False, n_images: int = 1024) -> MetricsType:
    """Compute metrics from generator output.

    Compute Wasserstein distances between generated images indicators
    and reference indicators then eventually save boxes.

    Parameters
    ----------
    gen : torch.nn.Module
        Generator module to evaluate.
    config : ConfigType
        Configuration of the experiment.
    training : bool
        True if the generator is training and False otherwise.
    step : int
        Current step of the experiment. It is used for title of plot
        boxes when saved.
    indicators_path : str or None, optional
        Path of reference indicators. If None, the path is inferred
        from the configuration and the generated images.
    n_images : int, optional
        Minimum number of images to generate to compute metrics.
        The actual number of images generated is the minimum multiple
        of batch_size greater or equal to n_images.

    Returns
    -------
    w_dists : Dict[str, float]
        List of Wasserstein metrics from data1 w.r.t data2. The keys
        are indicators name (with corresponding class) and the value is
        the Wasserstein distance between values of the two classes.
        Lower values means better similarity.
    other_metrics : Dict[str, float]
        Other metrics.
    """
    gen.eval()
    device = next(gen.parameters()).device
    batch_size = (config.data.train_batch_size
                  if training else config.data.test_batch_size)
    print(" -> Generating images for metrics calculation:", end='\r')
    # Generate more than n_images images to compute metrics
    data_gen = []
    with torch.no_grad():
        n_wrongs, total_pixels = 0, 0
        for k in range(max(1, int(np.ceil(n_images // batch_size)))):
            if config.trunc_ampl > 0:
                # Truncation trick
                z_input = torch.fmod(
                    torch.randn(batch_size, config.model.z_dim, device=device),
                    config.trunc_ampl)
            else:
                z_input = torch.randn(batch_size, config.model.z_dim,
                                      device=device)
            if 'cond' in config.model.architecture:
                n_pixels = config.data.n_pixels_cond
                pixel_size = config.data.pixel_size_cond
                data_size = config.model.data_size
                pixel_maps = generate_pixel_maps(batch_size=batch_size,
                                                 n_classes=gen.n_classes,
                                                 n_pixels=n_pixels,
                                                 pixel_size=pixel_size,
                                                 data_size=data_size,
                                                 device=device)
                out = gen(z_input, pixel_maps=pixel_maps)
                out = torch.argmax(out, dim=1).detach().cpu().numpy()
                pixel_maps_np = pixel_maps.detach().cpu().numpy()
                pixel_classes = np.concatenate([
                    1 - np.sum(pixel_maps_np[:, 1:], axis=1, keepdims=True),
                    pixel_maps_np[:, 1:]
                    ], axis=1)
                pixel_classes = np.argmax(pixel_classes, axis=1)
                pixel_classes = np.where(pixel_maps_np[:, 0], pixel_classes, 0)
                out_pixels = np.where(pixel_maps_np[:, 0], out, 0)
                n_wrongs += np.count_nonzero(out_pixels - pixel_classes)
                total_pixels += np.count_nonzero(pixel_maps_np[:, 0])

            else:
                out = gen(z_input)
                out = torch.argmax(out, dim=1).detach().cpu().numpy()
            torch.cuda.empty_cache()  # Free GPU memory
            data_gen.append(out)
            print(
                " -> Generating images for metrics calculation: "
                f"{(k + 1)*batch_size} images", end='\r')
    if total_pixels > 0:  # conditional GAN
        other_metrics = {'cond_acc': 1.0 - n_wrongs / total_pixels}
    else:
        other_metrics = {}
    print()
    data_gen_arr = np.vstack(data_gen)
    data_gen_arr = data_gen_arr.astype(np.uint8)
    print(" -> Computing indicators...")

    indicators_list_ref = get_reference_indicators(config, indicators_path,
                                                   data_gen_arr)

    metrics_save_dir = osp.join(config.output_dir, config.run_name, 'metrics')
    os.makedirs(metrics_save_dir, exist_ok=True)
    save_boxes_path = None

    if config.training.save_boxes:
        if training:
            save_boxes_path = osp.join(metrics_save_dir,
                                       f'boxes_step_{step}.png')
            metrics_save_path = osp.join(metrics_save_dir,
                                         f'metrics_step_{step}')
        else:
            save_boxes_path = osp.join(metrics_save_dir,
                                       f'test_boxes_step_{step}.png')
            metrics_save_path = osp.join(metrics_save_dir,
                                         f'test_metrics_step_{step}')

    # Compute metrics and save boxes locally if needed
    w_dists = wasserstein_distances(data_gen_arr, indicators_list_ref,
                                    save_boxes_path=save_boxes_path,
                                    **config.metrics)[0]
    save_metrics((w_dists, other_metrics), metrics_save_path,
                 save_json=save_json, save_csv=save_csv)
    return w_dists, other_metrics


def print_metrics(metrics: MetricsType, step: Optional[int] = None) -> None:
    """Print metrics witch colored table."""
    console = Console()
    title = 'Wasserstein dists' if step is None else f'Metrics (step {step})'
    table = Table(title=title, show_header=True, header_style='yellow',
                  title_style='bold yellow')
    wdists = metrics[0]
    split_wdists = split_wass_dists(wdists)

    table.add_column('Indicators', style='green', justify='center')
    n_classes = len(split_wdists) - 1
    for class_id in range(1, n_classes + 1):
        table.add_column(f'Class {class_id}', style='#0835ff',
                         justify='center')
    table.add_column('Mean', style='#b108ff', justify='center')
    rows, ind_names = {}, []
    for i, metrics_cls in enumerate(split_wdists[:-1]):
        for ind_name, value in metrics_cls.items():
            if ind_name not in rows:
                ind_names.append(ind_name)
                rows[ind_name] = ['-'] * (n_classes+1)
            rows[ind_name][i] = f'{value:.4f}'

    # Compute mean of classes for each indicator
    for ind_name, values in rows.items():
        row = [float(val) for val in values if val != '-']
        values[-1] = f'{np.mean(row):.4f}'
    rows['global'] = ['-'] * n_classes + [f'{wdists["global"]:.4f}']

    # Compute mean of indicators for each class
    for i in range(0, len(rows[list(rows.keys())[0]]) - 1):
        mean = np.mean([float(row[i]) for row in rows.values()
                        if row[i] != '-'])
        rows['global'][i] = f'{mean:.4f}'

    for ind_name, values in rows.items():
        if ind_name == ind_names[-1]:  # Last indicator
            table.add_row(ind_name, *values, end_section=True)
        elif ind_name == 'global':  # Global
            table.add_row(ind_name, *values, style='#ff8519')
        else:
            table.add_row(ind_name, *values)
    console.print(table)

    # Print other metrics if any
    if 'cond_acc' in metrics[1]:
        cond_acc = metrics[1]['cond_acc']
        console.print("Conditional accuracy: ", style='bold yellow', end='')
        console.print(f"{cond_acc:.4f}", style='bold #ff8519', highlight=False)

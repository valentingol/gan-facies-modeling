"""Auxiliaries for metrics."""

import json
import os
from typing import Dict, List, Tuple

import pandas as pd

MetricsType = Tuple[Dict[str, float], Dict[str, float]]


def save_metrics(metrics: MetricsType, metrics_save_path: str,
                 save_json: bool, save_csv: bool) -> None:
    """Save metrics in json and/or csv file."""
    metrics_dir, _ = os.path.split(metrics_save_path)
    if metrics_dir == '':
        metrics_dir = '.'
    os.makedirs(metrics_dir, exist_ok=True)
    if save_json:
        save_metrics_path_json = metrics_save_path + '.json'
        with open(save_metrics_path_json, 'w', encoding='utf-8') as file_out:
            json.dump({**metrics[0], **metrics[1]}, file_out,
                      separators=(',', ':'),
                      sort_keys=False, indent=4)
        print(f'Metrics saved at {save_metrics_path_json}')
    if save_csv:
        save_metrics_path_csv = metrics_save_path + '.csv'
        n_classes = get_n_classes(metrics[0])
        split_wdists = split_wass_dists(metrics[0])
        header = [f'Class {i}' for i in range(1, n_classes + 1)]
        header += ['Mean']
        if 'cond_acc' in metrics[1]:
            header += ['Conditional Acc']
            split_metrics = split_wdists + [metrics[1]]
        else:
            split_metrics = split_wdists
        pd.DataFrame(split_metrics).T.to_csv(save_metrics_path_csv,
                                             index=True,
                                             header=header,
                                             float_format='%.4f')
        print(f'Metrics saved at {save_metrics_path_csv}')


def split_wass_dists(metrics: Dict[str, float]) -> List[Dict[str, float]]:
    """Split metrics by classes."""
    n_classes = get_n_classes(metrics)
    split_metrics = []
    for class_id in range(1, n_classes + 1):
        metrics_cls = {}
        cls_str = f'_cls_{class_id}'
        for ind_name, value in metrics.items():
            if ind_name.endswith(cls_str):
                base_name = ind_name[:-len(cls_str)]
                base_name.replace('_', ' ')
                metrics_cls[base_name] = value
        split_metrics.append(metrics_cls)
    split_metrics += [{'global': metrics['global']}]
    return split_metrics


def get_n_classes(metrics: Dict[str, float]) -> int:
    """Get number of classes from metrics."""
    classes = []
    for ind_name in metrics:
        if ind_name.split('_')[-1].isdigit():
            classes.append(int(ind_name.split('_')[-1]))
    return max(classes)

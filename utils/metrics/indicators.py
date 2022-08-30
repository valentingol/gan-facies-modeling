"""Indicators implementation on numpy.

Indicators inspired by those presented in:
Guillaume Rongier and al,
"Comparing connected structures in ensemble of random fields"
(https://www.sciencedirect.com/science/article/abs/pii/S0309170816302573)
"""

from math import prod
from typing import Dict, List, Optional

import numpy as np

from utils.metrics.components import get_components_properties


def compute_indicators(data: np.ndarray, connectivity: Optional[int] = None,
                       unit_component_size: int = 1
                       ) -> List[Dict[str, List[float]]]:
    """Compute class indicators from data.

    Parameters
    ----------
    data : np.ndarray, dtype=np.uint8
        Batch of images of dims 3 (2D images) or 4 (3D images).
        Shape (n_images, [depth,] height, with).
    connectivity : int or None, optional
        Either 1 for 4-neighborhood (6-neighborhood in 3D)
        or 2 for 8-neighborhood (18-neighborhood in 3D)
        or 3 for 26-neighborhood in 3D (connectivity must be < 3 in 2D).
        If None, it is set to 2 for 2D images, 3 for 3D images.
        By default, None.
    unit_component_size : int, optional
        Maximum size to consider a component as a unit component.
        By default, 1.

    Raises
    ------
    ValueError
        If data dtype is not uint8.
        If data.ndim is not 2 or 3.
        If connectivity is not 1, 2, 3 or None
        If connectivity is 3 and data.ndim is 3 (2D images).

    Returns
    -------
    indicators_list : List[Dict[str, List[float]]]
        List of indicators from data, keys are indicators name and values are
        indicators values for each image. One indicators dict per classes
        (but class 0).
    """
    if data.dtype != np.uint8:
        raise ValueError(f'Data must be uint8, found {data.dtype}.')
    if data.ndim not in {3, 4}:
        raise ValueError(f'Data dim must be 3 (2D images) or 4 '
                         f'(3D images), found {data.ndim}.')
    if connectivity not in {1, 2, 3, None}:
        raise ValueError('Connectivity must be 1, 2, 3 or None.'
                         f'found {connectivity}.')
    if connectivity == 3 and data.ndim == 3:
        raise ValueError('Connectivity 3 is not supported for 2D images.')

    connectivity = data.ndim - 1 if connectivity is None else connectivity

    n_classes = np.max(data) + 1
    properties_list, neighbors = get_components_properties(
        data,
        connectivity=connectivity,
        unit_component_size=unit_component_size)
    indicators_list = []
    for properties in properties_list:
        indicators = compute_indicators_from_props(properties, neighbors,
                                                   n_classes=n_classes)
        indicators_list.append(indicators)
    return indicators_list


def class_prop(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute class proportion."""
    mask = properties['mask']
    axis = tuple(range(1, mask.ndim))
    prop = np.sum(mask, axis=axis) / (prod(mask.shape[1:]) + 1e-3)
    return prop


def class_adj_prop(properties: Dict[str, np.ndarray],
                   neighbors: np.ndarray,
                   adj_class_id: int) -> np.ndarray:
    """Compute class adjacency proportion."""
    class_id, mask = properties['class'], properties['mask']
    exteriors = ~ (neighbors == class_id).all(axis=-1) * mask
    neighbors = (neighbors == adj_class_id).any(axis=-1) * mask
    axis = tuple(range(1, mask.ndim))
    adj_prop = (np.sum(neighbors, axis=axis)
                / (np.sum(exteriors, axis=axis) + 1e-3))
    return adj_prop


def connectivity_proba(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute connection probability."""
    mask, areas = properties['mask'], properties['areas']
    numerator = np.sum(areas ** 2, axis=-1)
    axis = tuple(range(1, mask.ndim))
    proba = numerator / (np.sum(mask, axis=axis)**2 + 1e-3)
    return proba


def connected_density(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute connected component density."""
    components = properties['components']
    axis = tuple(range(1, components.ndim))
    n_components = np.max(components, axis=axis)
    density = n_components / prod(components.shape[1:])
    return density


def unit_connected_prop(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute unit connected component proportion."""
    components, n_units = properties['components'], properties['n_units']
    axis = tuple(range(1, components.ndim))
    n_components = np.max(components, axis=axis)
    unit_prop = n_units / (n_components + 1e-3)
    return unit_prop


def traversing_prop(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute traversing connected component proportion."""
    components, n_units = properties['components'], properties['n_units']
    axis = tuple(range(1, components.ndim))
    n_components = np.max(components, axis=axis)
    axis_any = -1 if components.ndim == 3 else (-2, -1)
    # For each object, check if it is adjacent to opposite side in
    # axis x or y. Then, sum to get the number of objects satisfying
    # this condition.
    traversing = np.sum(
        [np.logical_or(
            np.logical_and(np.any(components[..., 0, :] == i, axis=axis_any),
                           np.any(components[..., -1, :] == i, axis=axis_any)),
            np.logical_and(np.any(components[..., :, 0] == i, axis=axis_any),
                           np.any(components[..., :, -1] == i, axis=axis_any))
            )
         for i in range(1, components.max() + 1)],
        axis=0)
    travers_prop = traversing / (n_components - n_units + 1e-3)
    return travers_prop


def number_connected(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute number of connected component cells."""
    areas, mask_unit = properties['areas'], properties['mask_unit']
    num_connected = (np.sum(areas * mask_unit, axis=-1)
                     / (np.sum(mask_unit, axis=-1) + 1e-3))
    return num_connected


def box_ratio(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute box ratio."""
    extents, mask_unit = properties['extents'], properties['mask_unit']
    b_ratio = (np.sum(extents * mask_unit, axis=-1)
               / (np.sum(mask_unit, axis=-1) + 1e-3))
    return b_ratio


def face_cell_ratio(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute faces/cells ratio."""
    areas, perimeters = properties['areas'], properties['perimeters']
    mask_unit = properties['mask_unit']
    cc_ratio = (np.sum(perimeters * mask_unit / (areas + 1e-3), axis=-1)
                / (np.sum(mask_unit, axis=-1) + 1e-3))
    return cc_ratio


def sphericity(properties: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute sphericity."""
    areas, perimeters = properties['areas'], properties['perimeters']
    mask_unit = properties['mask_unit']
    # Convert to float to avoid overflow in division
    perimeters = perimeters.astype(np.float32)
    areas = areas.astype(np.float32)
    spher = (np.sum(areas**2 / (perimeters**3 + 1e-3), axis=-1)
             / (np.sum(mask_unit, axis=-1) + 1e-3))
    spher *= 36 * np.pi
    return spher


def compute_indicators_from_props(properties: Dict[str, np.ndarray],
                                  neighbors: np.ndarray, n_classes: int
                                  ) -> Dict[str, List[float]]:
    """Compute all the indicators from properties and neighbors."""
    # Basic indicators
    indicators = {
        'prop': class_prop(properties),
        'proba': connectivity_proba(properties),
        'density': connected_density(properties),
        'unit_prop': unit_connected_prop(properties),
        'traversing_prop': traversing_prop(properties),
        'num_connected': number_connected(properties),
        'box_ratio': box_ratio(properties),
        'face_cell_ratio': face_cell_ratio(properties),
        'sphericity': sphericity(properties)
    }

    # Adjacency indicators
    adj_class_ids = list(range(n_classes))
    class_id = properties['class']
    for adj_class_id in adj_class_ids:
        if adj_class_id != class_id:  # Do not compare with self
            indicators[f'adj_to_{adj_class_id}_prop'] = class_adj_prop(
                properties, neighbors, adj_class_id)

    # Normalization
    indicators_list = {}
    for ind_name, values in indicators.items():
        indicators_list[ind_name] = values.tolist()
    return indicators_list

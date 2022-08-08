"""Compute properties and architecture of connected components and classes.

These properties are used in utils/metrics/indicators.py.
"""

from typing import Dict, List, Tuple

import numpy as np
from skimage.measure import label, regionprops

PropertiesType = Dict[str, np.ndarray]


def get_components_properties(data: np.ndarray, connectivity: int,
                              unit_component_size: int = 1
                              ) -> Tuple[List[PropertiesType], np.ndarray]:
    """Get all components and class properties from data.

    Parameters
    ----------
    data : np.ndarray, dtype=np.uint8
        Batch of images of dims 3 (2D images) or 4 (3D images).
        Shape (n_images, [depth,] height, with).
    connectivity : int
        Either 1 for 4-neighborhood (6-neighborhood in 3D)
        or 2 for 8-neighborhood (18-neighborhood in 3D)
        or 3 for 26-neighborhood in 3D (connectivity must be < 3 in 2D).
    unit_component_size : int, optional
        Maximum size to consider a component as a unit component.
        By default, 1.

    Returns
    -------
    properties_list : List[PropertiesType]
        List of dictionaries of properties of each class.
    neighbors : np.ndarray, dtype = np.uint8
        Neighbors of each pixel of shape
        (batch, [depth,] height, width, n_neighbors).
        n_neighbors is either 4 (2D) or 6 (3D) for connectivity=1
        or 8 (2D) or 18 (3D) for connectivity=2.
        or 26 (3D) for connectivity=3.
        By convention, the neighbors outside the images are set to 255
        (equivalent to -1 in np.uint8 dtype).
    """
    n_classes: int = np.max(data)  # 0 is not considered as a class here
    properties_list = []
    neighbors = get_neighbors(data, connectivity)
    for class_id in range(1, n_classes + 1):
        components = get_connected_components(
            data, class_id=class_id, connectivity=connectivity)
        areas, extents, perimeters = get_props(components, neighbors, class_id)
        n_units, mask_unit = get_units_props(
            areas, unit_component_size=unit_component_size)

        properties = {
            'class': class_id,
            'mask': data == class_id,
            'components': components,
            'areas': areas,
            'extents': extents,
            'perimeters': perimeters,
            'n_units': n_units,
            'mask_unit': mask_unit
        }
        properties_list.append(properties)
    return properties_list, neighbors


def get_connected_components(data: np.ndarray, class_id: int,
                             connectivity: int) -> np.ndarray:
    """Get connected components of a class.

    Parameters
    ----------
    data : np.ndarray, dtype=np.uint8
        Batch of images of dims 3 (2D images) or 4 (3D images).
        Shape (n_images, [depth,] height, with).
    class_id : int
        Class id to get connected components.
    connectivity : int
        Either 1 for 4-neighborhood (6-neighborhood in 3D)
        or 2 for 8-neighborhood (18-neighborhood in 3D)
        or 3 for 26-neighborhood in 3D (connectivity must be < 3 in 2D).

    Returns
    -------
    components : np.ndarray
        Connected components labels of the same shape as data.
    """
    components = np.concatenate(
        [label(data[i] == class_id, background=0,
               connectivity=connectivity)[None, ...]
         for i in range(data.shape[0])],
        axis=0)
    return components


def get_props(components: np.ndarray,
              neighbors: np.ndarray,
              class_id: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get area/volume, perimeter/surface area and extents of components.

    Parameters
    ----------
    components : np.ndarray
        Connected components of shape (n_images, max_n_components).
    neighbors : np.ndarray
        Neighbors of each pixel of shape
        (batch, [depth,] height, width, n_neighbors).
    class_id : int
        Class number.

    Returns
    -------
    areas : np.ndarray
        Areas (2D) or volumes (3D) of each component of shape
        (n_images, max_n_components).
    extents : np.ndarray
        Extents of each component of shape (n_images, max_n_components).
        Extent is defined by (area or volume) / (bounding box area or volume).
    perimeters : np.ndarray
        Perimeters (2D) or surface areas (3D) of each component of shape
        (n_images, max_n_components).
    """
    def get_perimeter(components: np.ndarray, neighbors: np.ndarray,
                      class_id: int
                      ) -> np.ndarray:
        """Compute perimeter (2D objects) or surface area (3D objects)."""
        # connect_1: neighbors with 1-connectivity
        # 2D: 4-neighbors, 3D: 6-neighbors
        connect_1 = (neighbors[..., :4] if neighbors.ndim == 4
                     else neighbors[..., :6])
        # Get all neighbors different from class_id and border
        # (255 by convention, see 'get_neighbors_2d/_3d')
        mask_ext = np.where((connect_1 != class_id) & (connect_1 != 255), 1, 0)
        axis = tuple(range(1, connect_1.ndim))
        # perimeter: number of edges (2D) or faces (3D) in the surface of
        # the components
        perimeters = [np.sum(np.where((components == i)[..., None]
                                      & (mask_ext == 1), 1, 0), axis=axis)
                      for i in range(1, components.max() + 1)]
        perimeters_arr = np.array(perimeters, dtype=np.int32).T
        return perimeters_arr  # shape (n_imgs, max_n_components)

    areas = np.zeros((components.shape[0], np.max(components)),
                     dtype=np.int32)
    extents = np.zeros((components.shape[0], np.max(components)),
                       dtype=np.float32)

    for im_id in range(components.shape[0]):
        # Get components properties object
        props = regionprops(components[im_id])
        # Get areas / volumes
        area_im = list(map(lambda x: x.area, props))
        areas[im_id][:len(area_im)] = area_im
        # Get extents (= area / bounding box area)
        extents_im = list(map(lambda x: x.extent, props))
        extents[im_id][:len(extents_im)] = extents_im
    # Get perimeters / surface areas
    perimeters = get_perimeter(components, neighbors, class_id)

    return areas, extents, perimeters


def get_units_props(areas: np.ndarray, unit_component_size: int = 1
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Get unit cells (small cells) properties.

    Parameters
    ----------
    areas : np.ndarray
        Areas of connected components of shape (n_images, max_n_components).
    unit_component_size : int, optional
        Maximum size to consider a component as a unit component.
        By default, 1.

    Returns
    -------
    n_units : np.ndarray
        Number of unit components of shape (n_images, ).
    mask_unit : np.ndarray
        Binary mask with 0 for unit or empty components.
    """
    n_units = np.sum((0 < areas) & (areas <= unit_component_size),
                     axis=-1)
    mask_unit = areas > unit_component_size
    return n_units, mask_unit


def get_neighbors(data: np.ndarray, connectivity: int) -> np.ndarray:
    """Get neighbors of each pixel for 2D or 3D images.

    Parameters
    ----------
    data : np.ndarray, dtype=np.uint8
        Batch of images of dims 3 (2D images) or 4 (3D images).
        Shape (n_images, [depth,] height, with).
    connectivity : int
        Either 1 for 4-neighborhood (6-neighborhood in 3D)
        or 2 for 8-neighborhood (18-neighborhood in 3D)
        or 3 for 26-neighborhood in 3D (connectivity must be < 3 in 2D).

    Returns
    -------
    neighbors : np.ndarray, dtype = np.uint8
        Neighbors of each pixel of shape
        (batch, [depth,] height, width, n_neighbors).
        n_neighbors is either 4 (2D) or 6 (3D) for connectivity=1
        or 8 (2D) or 18 (3D) for connectivity=2.
        or 26 (3D) for connectivity=3.
        By convention, the neighbors outside the images are set to 255
        (equivalent to -1 in np.uint8 dtype).
    """
    if data.ndim == 3:  # 2D images
        return get_neighbors_2d(data, connectivity)
    # 3D images
    return get_neighbors_3d(data, connectivity)


def get_neighbors_2d(data: np.ndarray, connectivity: int) -> np.ndarray:
    """Get neighbors of each pixel for 2D images.

    See get_neighbors docstring for more details on inputs and outputs.
    """
    pad = [(0, 0)] * data.ndim
    h, w = data.shape[-3:-1]
    pad[-2] = pad[-3] = (1, 1)
    # Border neighbors are set to 255
    data_pad = np.pad(data, pad, 'constant', constant_values=255)
    neighbors = [
        data_pad[..., 0:h + 0, 1:w + 1, :, None],  # -1, 0
        data_pad[..., 1:h + 1, 0:w + 0, :, None],  # 0, -1
        data_pad[..., 1:h + 1, 2:w + 2, :, None],  # 0, 1
        data_pad[..., 2:h + 2, 1:w + 1, :, None],  # 1, 0
    ]
    if connectivity == 2:
        neighbors_2 = [
            data_pad[..., 0:h + 0, 0:w + 0, :, None],  # -1, -1
            data_pad[..., 0:h + 0, 2:w + 2, :, None],  # -1, 1
            data_pad[..., 2:h + 2, 0:w + 0, :, None],  # 1, -1
            data_pad[..., 2:h + 2, 2:w + 2, :, None],  # 1, 1
        ]
        neighbors = neighbors + neighbors_2

    neighbors_arr = np.concatenate(neighbors, axis=-1)
    return neighbors_arr


def get_neighbors_3d(data: np.ndarray, connectivity: int) -> np.ndarray:
    """Get neighbors of each pixel for 2D images.

    See get_neighbors docstring for more details on inputs and outputs.
    """
    pad = [(0, 0)] * data.ndim
    z, w, h = data.shape[-4:-1]
    pad[-2] = pad[-3] = pad[-4] = (1, 1)
    # Border neighbors are set to 255
    data_pad = np.pad(data, pad, 'constant', constant_values=255)
    neighbors = [
        data_pad[..., 0:z + 0, 1:w + 1, 1:h + 1, :, None],  # -1, 0, 0
        data_pad[..., 1:z + 1, 0:w + 0, 1:h + 1, :, None],  # 0, -1, 0
        data_pad[..., 1:z + 1, 1:w + 1, 0:h + 0, :, None],  # 0, 0, -1
        data_pad[..., 1:z + 1, 1:w + 1, 2:h + 2, :, None],  # 0, 0, 1
        data_pad[..., 1:z + 1, 2:w + 2, 1:h + 1, :, None],  # 0, 1, 0
        data_pad[..., 2:z + 2, 1:w + 1, 1:h + 1, :, None],  # 1, 0, 0
    ]
    if connectivity >= 2:  # 2 or 3
        neighbors_2 = [
            data_pad[..., 0:z + 0, 0:w + 0, 1:h + 1, :, None],  # -1, -1, 0
            data_pad[..., 0:z + 0, 1:w + 1, 0:h + 0, :, None],  # -1, 0, -1
            data_pad[..., 0:z + 0, 1:w + 1, 2:h + 2, :, None],  # -1, 0, 1
            data_pad[..., 0:z + 0, 2:w + 2, 1:h + 1, :, None],  # -1, 1, 0

            data_pad[..., 1:z + 1, 0:w + 0, 0:h + 0, :, None],  # 0, -1, -1
            data_pad[..., 1:z + 1, 0:w + 0, 2:h + 2, :, None],  # 0, -1, 1
            data_pad[..., 1:z + 1, 2:w + 2, 0:h + 0, :, None],  # 0, 1, -1
            data_pad[..., 1:z + 1, 2:w + 2, 2:h + 2, :, None],  # 0, 1, 1

            data_pad[..., 2:z + 2, 0:w + 0, 1:h + 1, :, None],  # 1, -1, 0
            data_pad[..., 2:z + 2, 1:w + 1, 0:h + 0, :, None],  # 1, 0, -1
            data_pad[..., 2:z + 2, 1:w + 1, 2:h + 2, :, None],  # 1, 0, 1
            data_pad[..., 2:z + 2, 2:w + 2, 1:h + 1, :, None],  # 1, 1, 0
        ]
        neighbors = neighbors + neighbors_2

    if connectivity == 3:
        neighbors_3 = [
            data_pad[..., 0:z + 0, 0:w + 0, 0:h + 0, :, None],  # -1, -1, -1
            data_pad[..., 0:z + 0, 0:w + 0, 2:h + 2, :, None],  # -1, -1, 1
            data_pad[..., 0:z + 0, 2:w + 2, 0:h + 0, :, None],  # -1, 1, -1
            data_pad[..., 0:z + 0, 2:w + 2, 2:h + 2, :, None],  # -1, 1, 1

            data_pad[..., 2:z + 2, 0:w + 0, 0:h + 0, :, None],  # 1, -1, -1
            data_pad[..., 2:z + 2, 0:w + 0, 2:h + 2, :, None],  # 1, -1, 1
            data_pad[..., 2:z + 2, 2:w + 2, 0:h + 0, :, None],  # 1, 1, -1
            data_pad[..., 2:z + 2, 2:w + 2, 2:h + 2, :, None],  # 1, 1, 1
        ]
        neighbors = neighbors + neighbors_3

    neighbors_arr = np.concatenate(neighbors, axis=-1)
    return neighbors_arr

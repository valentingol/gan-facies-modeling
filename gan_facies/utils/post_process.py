"""Post-processing functions."""

from typing import Dict, Tuple

import cv2
import numpy as np

import gan_facies.metrics.metric as met


def clip_by_indicator(data: np.ndarray, indicators: met.IndicatorsList,
                      indicator_name: str, class_id: int, alpha: float,
                      order: str) -> Tuple[np.ndarray, met.IndicatorsList]:
    """Clip dataset by indicator value.

    Parameters
    ----------
    data: np.ndarray
        Whole dataset.
    indicators: IndicatorsList
        List of indicators. Each element is a dictionary (for each class)
        where keys are indicator names and values is the list
        of indicator values.
    indicator_name: str
        Indicator name to clip by.
    class_id: int
        Class id to clip by (0 is the first class in indicators list).
    alpha: float
        Percentile to clip by.
    order: str
        Must be 'above' or 'below'. If 'above', then clip above
        alpha percentile. If 'below', then clip below alpha percentile.

    Returns
    -------
    new_data: np.ndarray
        Clipped dataset.
    new_indicators: IndicatorsList
        Indicators of the returned new dataset.
    """
    indicator = np.array(indicators[class_id][indicator_name])
    quant = np.quantile(indicator, alpha)
    if order == 'above':
        ind = np.where(indicator >= quant)[0]
    elif order == 'below':
        ind = np.where(indicator <= quant)[0]
    else:
        raise ValueError('Order argument should be "above" or "below", found'
                         f'"{order}" instead.')
    new_data = data[ind]
    new_indicators: met.IndicatorsList = []
    for indics in indicators:
        new_indicators.append({})
        for key in indics:
            new_indicators[-1][key] = np.array(indics[key])[ind].tolist()
    return new_data, new_indicators


def clip_indicator_eval(indicators_ref: met.IndicatorsList,
                        clip_by_indicator_params: Dict,
                        metric_params: Dict) -> Tuple[np.ndarray,
                                                      met.IndicatorsList,
                                                      int,
                                                      Dict[str, float]]:
    """Clip by indicator and evaluate.

    Parameters
    ----------
    indicators_ref: IndicatorsList
        List of indicators of the reference dataset. Each element is a
        dictionary (for each class) where keys are indicator names
        and values is the list of indicator values.
    clip_by_indicator_params: Dict
        Parameters for the clip_by_indicator function
        (see `clip_by_indicator` docstring).
    metric_params: Dict
        Metric parameters.

        connectivity : int or None
            Either 1 for 4-neighborhood (6-neighborhood in 3D)
            or 2 for 8-neighborhood (18-neighborhood in 3D)
            or 3 for 26-neighborhood in 3D (connectivity must be < 3
            in 2D). If None, it is set to 2 for 2D images, 3 for
            3D images.
        unit_component_size : int
            Maximum size to consider a component as a unit component.

    Returns
    -------
    new_data: np.ndarray
        Clipped dataset.
    new_indicators: IndicatorsList
        Indicators of the returned new dataset.
    length_new: int
        Length of the new dataset.
    dists_new: Dict[str, float]
        Wasserstein distances of the new dataset.
    """
    data = clip_by_indicator_params['data']
    indicators = clip_by_indicator_params['indicators']
    indicator_name = clip_by_indicator_params['indicator_name']
    class_id = clip_by_indicator_params['class_id']
    alpha = clip_by_indicator_params['alpha']
    order = clip_by_indicator_params['order']
    data_new, indicators_new = clip_by_indicator(data, indicators,
                                                 indicator_name=indicator_name,
                                                 class_id=class_id,
                                                 alpha=alpha,
                                                 order=order)
    connectivity = metric_params['connectivity']
    unit_component_size = metric_params['unit_component_size']
    dists_new = met.wasserstein_distances(indicators_new, indicators_ref,
                                          connectivity, unit_component_size)[0]
    length_new = len(data_new)
    return data_new, indicators_new, length_new, dists_new


def resize_by_factor(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize the input image by a factor."""
    shape = image.shape
    if factor == 1:
        return image
    if factor >= 1:
        factor_int = int(np.round(factor))
        new_shape = (shape[0] * factor_int, shape[1] * factor_int)
    else:
        factor_int = int(np.round(1.0 / factor))
        new_shape = (shape[0] // factor_int, shape[1] // factor_int)
    return cv2.resize(image, new_shape, interpolation=cv2.INTER_NEAREST)


def erode_dilate(mask: np.ndarray) -> np.ndarray:
    """Erode and dilate the mask to remove thin parts."""
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)
    mask1 = cv2.erode(mask, kernel1, iterations=1)
    mask2 = cv2.erode(1 - mask, kernel2, iterations=1)

    mask = mask - (mask1 * mask2)

    return mask


def fill_blank_neighborhood(img: np.ndarray) -> np.ndarray:
    """Fill blank pixels (-1) in the image (int32) using neighborhood."""
    blank_idx = np.where(img == -1)
    for id_y, id_x in zip(blank_idx[0], blank_idx[1]):
        if (0 < id_y < img.shape[0] - 1 and 0 < id_x < img.shape[1] - 1):
            neighbours = img[id_y - 1:id_y + 2, id_x - 1:id_x + 2]
            neighbours = neighbours.reshape((-1, ))
            uniques, counts = np.unique(neighbours, return_counts=True)
            uniques_list, counts_list = uniques.tolist(), counts.tolist()
            blank_idx = uniques_list.index(-1)
            del counts_list[blank_idx], uniques_list[blank_idx]
            max_i = np.argmax(counts_list)
            value = uniques_list[max_i]
            img[id_y, id_x] = value
        else:
            img[id_y, id_x] = 0
    return img


def postprocess_erode_dilate(img: np.ndarray) -> np.ndarray:
    """Post-process the image with erosion + dilatation."""
    masks = []
    for value in range(img.max() + 1):
        mask = np.where(img == value, 1, 0).astype(np.uint8)
        mask = erode_dilate(mask)
        masks.append(mask)
    new_img = np.zeros_like(img, dtype=np.int32) - 1
    for value, mask in enumerate(masks):
        new_img = np.where((mask == 1), value, new_img)
    new_img = fill_blank_neighborhood(new_img)
    return new_img

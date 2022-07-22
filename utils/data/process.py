"""Pre-process and post-process data utilities."""

import numpy as np
from skimage.transform import resize


def to_one_hot_np(data: np.ndarray) -> np.ndarray:
    """One-hot encode the input numpy data."""
    num_classes = np.max(data) + 1
    data_shape = data.shape
    data_flat = data.reshape(-1)
    data_one_hot = np.zeros((data_flat.shape[0], num_classes))
    data_one_hot[np.arange(data_flat.shape[0]), data_flat] = 1
    data_one_hot = data_one_hot.reshape(data_shape + (num_classes,))
    return data_one_hot


def resize_np(data: np.ndarray, data_size: int) -> np.ndarray:
    """Resize the numpy data to data_size.

    The function preserve the aspect ratio and the normalization.
    """
    data_y, data_x = data.shape[0:2]
    min_dim = min(data_y, data_x)
    if min_dim < data_size:
        if min_dim == data_y:
            dim_x = int(data_x * data_size / data_y)
            data = resize(data, (data_size, dim_x))
        else:
            dim_y = int(data_y * data_size / data_x)
            data = resize(data, (dim_y, data_size))
    data = data / np.sum(data, axis=-1, keepdims=True)
    return data


def random_crop_np(data: np.ndarray, data_size: int) -> np.ndarray:
    """Random crop the numpy data to fit data_size."""
    data_y, data_x = data.shape[0:2]
    if data_y > data_size:
        y_0 = np.random.randint(0, data_y - data_size)
        data = data[y_0:y_0 + data_size]
    if data_x > data_size:
        x_0 = np.random.randint(0, data_x - data_size)
        data = data[:, x_0:x_0 + data_size]
    return data


def color_data_np(data: np.ndarray) -> np.ndarray:
    """Color data (one color per class).

    Parameters
    ----------
    data : np.ndarray
        Data of type int.

    Returns
    -------
    rgb_data : np.ndarray
        RGB data of type int with shape data.shape + (3,).
    """
    rgb_data = np.zeros(data.shape + (3,), dtype=np.uint8)
    data = data[..., None]
    color_dict = {
        0: [0, 0, 130],
        1: [0, 110, 250],
        2: [94, 235, 0],
        3: [235, 223, 0],
        4: [255, 153, 0],
        5: [255, 64, 0],
        6: [255, 0, 0],
        7: [183, 0, 255],
    }
    n_classes = np.max(data) + 1
    for i in range(n_classes):
        if i <= 7:
            rgb_data[(data == i)[..., 0]] = color_dict[i]
        else:
            red = np.random.randint(0, 256)
            green = np.random.randint(0, 256)
            blue = np.random.randint(0, 256)
            rgb_data[(data == i)[..., 0]] = np.array([red, green, blue])
    return rgb_data


def to_img_grid(batched_images: np.ndarray) -> np.ndarray:
    """Transform batched RGB images as a grid of images.

    Parameters
    ----------
    batched_images : np.ndarray
        Batched RGB images of type np.uint8 with shape
        (batch_size, height, width, 3).

    Returns
    -------
    img_grid : np.ndarray
        Grid of images of type np.uint8 with shape
        (height*int(sqrt(batch_size)), width*int(sqrt(batch_size), 3).
        Note that remaining images are discarded!
    """
    y_dim, x_dim = batched_images.shape[1:3]
    side = min(8, int(np.sqrt(batched_images.shape[0])))
    image = batched_images[:side * side].reshape(side, side, y_dim, x_dim, 3)
    image = np.transpose(image, (0, 2, 1, 3, 4))
    img_grid = np.reshape(image, (side * y_dim, side * x_dim, 3))
    return img_grid

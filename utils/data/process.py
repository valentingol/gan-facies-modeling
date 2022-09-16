"""Pre-process and post-process data utilities."""

import random
from typing import List, Union

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
    # Pad images with white pixels
    batched_images = np.pad(batched_images, ((0, 0), (1, 1), (1, 1), (0, 0)),
                            constant_values=255)
    y_dim, x_dim = batched_images.shape[1:3]
    side = min(8, int(np.sqrt(batched_images.shape[0])))
    image = batched_images[:side * side].reshape(side, side, y_dim, x_dim, 3)
    image = np.transpose(image, (0, 2, 1, 3, 4))
    img_grid = np.reshape(image, (side * y_dim, side * x_dim, 3))
    return img_grid


def sample_pixels_2d_np(data: np.ndarray, n_pixels: Union[int, List[int]],
                        pixel_size: int) -> np.ndarray:
    """Sample pixel map similar to GANSim's 'well facies data' (2D case).

    Return a binary map containing n_classes values for each pixel.
    The first value is 1 if a log is sampled in the pixel,
    the n_classes-1 last values determine the class of the log.
    By convention, if the class is 0 (background facies), the last
    values are all 0. More details in the GANSim paper:
    https://link.springer.com/article/10.1007/s11004-021-09934-0

    Parameters
    ----------
    data : np.ndarray
        2D data matrix with probabilities of shape (h, w, n_classes)
        to sample from.
    n_pixels : int or List[int] (length 2)
        If int, the number of pixels to sample for each 2D matrix.
        If tuple, the number of pixels to sample for each 2D matrix
        will be sampled uniformly between the two values.
    pixel_size : int
        Size of the pixel to sample. The class is uniform in the pixel.

    Returns
    -------
    pixel_map : np.ndarray, np.float32
        Pixel map of shape (h, w, n_classes).
    """
    # First binarize the data to have binary pixel maps
    data = np.argmax(data, axis=-1)  # shape (h, w)
    data = to_one_hot_np(data)  # shape (h, w, n_classes)
    if isinstance(n_pixels, int):
        n_pixels_int = n_pixels
    elif isinstance(n_pixels, list) and len(n_pixels) == 2:
        n_pixels_int = np.random.randint(n_pixels[0], n_pixels[1])
    else:
        if isinstance(n_pixels, list):
            raise ValueError("n_pixels must be int or list of length 2, "
                             f"found list of lenght {len(n_pixels)}.")
        raise ValueError("n_pixels must be int or list of 2 ints, "
                         f"found type {type(n_pixels)}.")
    height, width = data.shape[0:2]
    # Get all possible big-pixel coordinates
    # in (data_size // pixel_size) scale
    tuples = [(i, j) for i in range(height // pixel_size)
              for j in range(width // pixel_size)]
    # Randomly sample big-pixels (without replacement)
    pixels_idx = random.sample(tuples, n_pixels_int)
    # Get mini-pixel coordinates of a big-pixel w.r.t the top-left corner
    grid = np.meshgrid(np.arange(pixel_size), np.arange(pixel_size))
    grid_h, grid_w = grid[0].flatten(), grid[1].flatten()
    # Get mini-pixel coordinates in data_size scale
    pixels_h = [i*pixel_size + k for i, _ in pixels_idx for k in grid_h]
    pixels_w = [j*pixel_size + k for _, j in pixels_idx for k in grid_w]
    pixel_mask = np.zeros((height, width), dtype=np.float32)
    pixel_mask[pixels_h, pixels_w] = 1.0
    pixel_mask = pixel_mask[..., None]
    # Remove first class and all information about not sampled pixels
    classes = data[..., 1:]
    pixel_classes = np.zeros_like(classes)
    for pixel_group, (i, j) in enumerate(pixels_idx):
        y_center = i * pixel_size + pixel_size // 2
        x_center = j * pixel_size + pixel_size // 2
        pixel_class = classes[y_center, x_center]
        start = pixel_group * pixel_size**2
        end = (pixel_group + 1) * pixel_size**2
        pixel_classes[pixels_h[start:end], pixels_w[start:end]] = pixel_class
    pixel_map = np.concatenate([pixel_mask, pixel_classes], axis=-1)
    pixel_map = pixel_map.astype(np.float32)
    return pixel_map  # shape (h, w, n_classes)

"""Generate ground_truth data."""

import numpy as np
import PIL.Image

from gan_facies.data.process import color_data_np, random_crop_np, to_img_grid
from gan_facies.utils.configs import GlobalConfig

if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='gan_facies/configs/exp/base.yaml')
    data = np.load(global_config.dataset_path)
    data_size = global_config.model.data_size
    # Random crop data
    data_crop_list = []
    for i in range(data.shape[0]):
        data_crop_list.append(random_crop_np(data[i], data_size))
    data_crop = np.array(data_crop_list)
    np.random.shuffle(data_crop)
    data = color_data_np(data_crop[:64])
    img_grid = to_img_grid(data)
    # Save the image
    PIL.Image.fromarray(img_grid).save('dataset_real.png')

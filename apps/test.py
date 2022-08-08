"""Test the generator."""

import os
import os.path as osp

import numpy as np
import torch
from PIL import Image

from utils.configs import ConfigType, GlobalConfig
from utils.data.process import to_img_grid
from utils.sagan.modules import SAGenerator
from utils.train.random_utils import set_global_seed


def test(config: ConfigType) -> None:
    """Test the generator."""
    # For reproducibility
    set_global_seed(seed=config.seed)

    architecture = config.model.architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config.test_batch_size
    dataset = np.load(config.dataset_path)
    n_classes = dataset.max() + 1

    model_dir = osp.join('res', config.run_name, 'models')
    step = config.recover_model_step
    if step <= 0:
        model_path = osp.join(model_dir, 'generator_last.pth')
    else:
        model_path = osp.join(model_dir, f'generator_step_{step}.pth')

    if architecture == 'sagan':
        generator = SAGenerator(n_classes=n_classes,
                                model_config=config.model).to(device)

    z_input = torch.fmod(torch.randn(batch_size, config.model.z_dim,
                                     device=device),
                         config.trunc_ampl)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    with torch.no_grad():
        images, attn_list = generator.generate(z_input, with_attn=True)
    # Save sample images in a grid
    img_out_path = osp.join('res', config.run_name, 'samples',
                            'test_samples.png')
    img_grid = to_img_grid(images)
    pil_images = Image.fromarray(img_grid)
    pil_images.show(title='Test Samples (run ' + config.run_name + ')')
    pil_images.save(img_out_path)

    if config.save_attn:
        # Save attention
        attn_out_path = osp.join('res', config.run_name, 'attention',
                                 'test_gen_attn')
        os.makedirs(attn_out_path, exist_ok=True)
        attn_list = [attn.detach().cpu().numpy() for attn in attn_list]
        for i, attn in enumerate(attn_list):
            np.save(osp.join(attn_out_path, f'attn_{i}.npy'), attn)


if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='configs/exp/base.yaml')
    # NOTE: The config is not saved when testing only
    test(global_config)

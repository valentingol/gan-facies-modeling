"""Test the generator."""

import os
import os.path as osp

import numpy as np
import torch
from PIL import Image

from utils.configs import ConfigType, GlobalConfig
from utils.data.data_loader import DataLoader2DFacies
from utils.data.process import to_img_grid
from utils.metrics import compute_save_indicators, evaluate, print_metrics
from utils.sagan.modules import SAGenerator
from utils.train.random_utils import set_global_seed


def test(config: ConfigType) -> None:
    """Test the generator."""
    # For reproducibility
    set_global_seed(seed=config.seed)

    architecture = config.model.architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config.data.test_batch_size
    dataset = np.load(config.dataset_path)
    n_classes = dataset.max() + 1

    model_dir = osp.join(config.output_dir, config.run_name, 'models')
    step = config.recover_model_step

    if step <= 0:
        model_path = osp.join(model_dir, 'generator_last.pth')
    else:
        model_path = osp.join(model_dir, f'generator_step_{step}.pth')

    if architecture == 'sagan':
        generator = SAGenerator(n_classes=n_classes,
                                model_config=config.model).to(device)
        data_loader = DataLoader2DFacies(config.dataset_path,
                                         data_size=config.model.data_size,
                                         training=False,
                                         data_config=config.data)
    else:
        raise ValueError(f'Unknown architecture: {architecture}.')
    if config.trunc_ampl > 0:
        z_input = torch.fmod(
            torch.randn(batch_size, config.model.z_dim, device=device),
            config.trunc_ampl)
    else:
        z_input = torch.randn(batch_size, config.model.z_dim, device=device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    with torch.no_grad():
        images, attn_list = generator.generate(z_input, with_attn=True)

    # Save sample images in a grid
    img_out_dir = osp.join(config.output_dir, config.run_name, 'samples')
    img_out_path = osp.join(img_out_dir, f'test_samples_step_{step}.png')
    img_grid = to_img_grid(images)
    pil_images = Image.fromarray(img_grid)
    pil_images.show(title=f'Test Samples (run {config.run_name}, step {step})')
    os.makedirs(img_out_dir, exist_ok=True)
    pil_images.save(img_out_path)

    if config.save_attn:
        # Save attention
        attn_out_path = osp.join(config.output_dir, config.run_name,
                                 'attention', 'test_gen_attn_step')
        os.makedirs(attn_out_path, exist_ok=True)
        attn_list = [attn.detach().cpu().numpy() for attn in attn_list]
        for i, attn in enumerate(attn_list):
            np.save(osp.join(attn_out_path, f'attn_{i}_step_{step}.npy'), attn)

    compute_save_indicators(data_loader, config)
    w_dists = evaluate(gen=generator, config=config, training=False, step=step,
                       save_json=False, save_csv=True)
    print("Metrics w.r.t training set:")
    print_metrics(w_dists)


if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='configs/exp/base.yaml')
    # NOTE: The config is not saved when testing only
    test(global_config)

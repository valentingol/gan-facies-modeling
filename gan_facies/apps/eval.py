"""Test the generator."""

import os
import os.path as osp
from typing import Optional

import numpy as np
import torch
from PIL import Image
from thop import profile

from gan_facies.data.data_loader import DatasetCond2D, DistributedDataLoader
from gan_facies.data.process import to_img_grid
from gan_facies.gan.cond_sagan.modules import CondSAGenerator
from gan_facies.gan.uncond_sagan.modules import UncondSAGenerator
from gan_facies.metrics import compute_save_indicators, evaluate, print_metrics
from gan_facies.utils.auxiliaries import set_global_seed
from gan_facies.utils.conditioning import (colorize_pixel_map,
                                           generate_pixel_maps)
from gan_facies.utils.configs import ConfigType, GlobalConfig


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
        generator = UncondSAGenerator(n_classes=n_classes,
                                      model_config=config.model).to(device)
        data_loader = DistributedDataLoader(config.dataset_path,
                                            data_size=config.model.data_size,
                                            training=False,
                                            data_config=config.data)
    elif architecture == 'cond_sagan':
        generator = CondSAGenerator(n_classes=n_classes,
                                    model_config=config.model).to(device)
        data_loader = DistributedDataLoader(config.dataset_path,
                                            data_size=config.model.data_size,
                                            training=False,
                                            data_config=config.data,
                                            dataset_class=DatasetCond2D)
    else:
        raise ValueError(f'Unknown architecture: {architecture}.')
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    if config.trunc_ampl > 0:
        z_input = torch.fmod(
            torch.randn(batch_size, config.model.z_dim, device=device),
            config.trunc_ampl)
    else:
        z_input = torch.randn(batch_size, config.model.z_dim, device=device)
    if 'cond' in architecture:
        # Create pixel_maps
        with torch.no_grad():
            n_pixels = config.data.n_pixels_cond
            pixel_size = config.data.pixel_size_cond
            data_size = config.model.data_size
            pixel_maps = generate_pixel_maps(
                batch_size=batch_size, n_classes=n_classes,
                n_pixels=n_pixels, pixel_size=pixel_size, data_size=data_size,
                device=device)
            colored_pixel_maps = colorize_pixel_map(pixel_maps)
            images, attn_list = generator.generate(z_input, pixel_maps,
                                                   with_attn=True)
            # pylint: disable=unbalanced-tuple-unpacking
            macs, _ = profile(generator, inputs=(z_input, pixel_maps))
            print(f'MACs: {macs / 1e9:.2f}G')
            _, _, proba_map = generator.proba_map(z_input, pixel_maps[0])
    else:
        colored_pixel_maps = None
        proba_map = None
        with torch.no_grad():
            images, attn_list = generator.generate(z_input, with_attn=True)
            # pylint: disable=unbalanced-tuple-unpacking
            macs, _ = profile(generator, inputs=(z_input,))

    # Save and show sample images in a grid
    img_out_dir = osp.join(config.output_dir, config.run_name, 'samples')
    img_out_path = osp.join(img_out_dir, f'test_samples_step_{step}.png')
    img_grid = to_img_grid(images)
    save_and_show(img_grid, img_out_path)

    # Save and show other images (if any)
    cond_save_path = img_out_path.replace('samples', 'cond_pixels')
    save_and_show(colored_pixel_maps, cond_save_path)
    proba_save_path = img_out_path.replace('samples', 'proba_map')
    save_and_show(proba_map, proba_save_path)

    # Save attention (if save_attn is True)
    if config.save_attn and attn_list != []:
        attn_out_path = osp.join(config.output_dir, config.run_name,
                                 'attention', 'test_gen_attn_step')
        os.makedirs(attn_out_path, exist_ok=True)
        attn_list = [attn.detach().cpu().numpy() for attn in attn_list]
        for i, attn in enumerate(attn_list):
            np.save(osp.join(attn_out_path, f'attn_{i}_step_{step}.npy'), attn)

    # Compute reference indicators if not already saved
    compute_save_indicators(data_loader, config)
    # Compute and print metrics
    metrics = evaluate(gen=generator, config=config, training=False, step=step,
                       save_json=False, save_csv=True)
    print(f'MACs: {macs / 1e9:.2f}G')
    print("Metrics w.r.t training set:")
    print_metrics(metrics)


def save_and_show(image: Optional[np.ndarray], path: str) -> None:
    """Save and show image using PIL."""
    if image is None:
        return
    image_pil = Image.fromarray(image)
    image_pil.show()
    dir_path, _ = osp.split(path)
    os.makedirs(dir_path, exist_ok=True)
    image_pil.save(path)


if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='gan_facies/configs/exp/base.yaml')
    # NOTE: The config is not saved when testing only
    test(global_config)

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from data import create_dataloader, create_dataset
from models import ISPModel
from utils.util import parse_config, save_image, setup_logger, tensor2img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--dataroot_RAW', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    opt = parse_config(args.config)
    opt['phase'] = 'test'
    if args.dataroot_RAW:
        opt['datasets']['test']['dataroot_RAW'] = args.dataroot_RAW
    if args.checkpoint:
        opt['resume']['path'] = args.checkpoint

    torch.backends.cudnn.benchmark = True
    exp_dir = args.output or os.path.join(opt['experiments_root'], opt['name'])
    logger = setup_logger(exp_dir)

    dataset = create_dataset(opt, 'test')
    loader = create_dataloader(dataset, opt, 'test')
    logger.info(f'{len(dataset.paths)} images -> {len(dataset)} tiles')

    model = ISPModel(opt, logger)
    model.net.set_deploy(True)
    logger.info('Re-parameterised branches merged for inference.')

    bit_depth = opt['datasets']['output_bit_depth']
    scale = opt['datasets']['scale']
    pad = opt['datasets']['border_padding'] * scale
    dtype = np.uint16 if bit_depth == 16 else np.uint8

    canvases = {}
    gsrm_cache = {}

    for batch in tqdm(loader, desc='Inference'):
        # The GSRM is evaluated once per image and reused for every tile of that
        # image, which is what keeps tone and white balance consistent across tiles.
        for i, path in enumerate(batch['image_path']):
            if path not in gsrm_cache:
                gsrm_cache[path] = model.compute_global_params(
                    batch['Global'][i:i + 1], batch['WB'][i:i + 1])
        global_params = tuple(
            torch.cat([gsrm_cache[p][k] for p in batch['image_path']], dim=0)
            for k in range(3))

        model.feed_data(batch)
        model.test(global_params=global_params)
        output = model.get_visuals()['output']

        for i in range(output.size(0)):
            name = batch['image_name'][i]
            height = batch['height'][i].item()
            width = batch['width'][i].item()
            if name not in canvases:
                canvases[name] = np.zeros((height, width, 3), dtype=dtype)

            tile = tensor2img(output[i], bit_depth)
            if pad > 0 and not bool(batch['is_full_image'][i]):
                if tile.shape[0] > 2 * pad and tile.shape[1] > 2 * pad:
                    tile = tile[pad:-pad, pad:-pad]

            x0 = batch['crop_X'][i].item()
            y0 = batch['crop_Y'][i].item()
            y1 = min(y0 + tile.shape[0], height)
            x1 = min(x0 + tile.shape[1], width)
            canvases[name][y0:y1, x0:x1] = tile[:y1 - y0, :x1 - x0]

            if bool(batch['last_tile'][i]):
                out_path = os.path.join(exp_dir, 'results',
                                        os.path.splitext(name)[0] + '.tif')
                save_image(canvases.pop(name), out_path)
                logger.info(f'Saved {out_path}')


if __name__ == '__main__':
    main()

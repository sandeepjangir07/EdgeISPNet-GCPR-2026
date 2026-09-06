import logging
import math
import os
import random
import sys

import numpy as np
import torch
import yaml


def parse_config(path, overrides=None):
    with open(path, 'r') as f:
        opt = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is not None:
            opt[key] = value
    if opt.get('gpu_ids'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in opt['gpu_ids'])
    return opt


def setup_logger(log_dir, name='edgeispnet'):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(log_dir, 'log.txt'))):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor2img(tensor, bit_depth=8, raw=False):
    """(C,H,W) float tensor in [0,1] -> (H,W,3) uint8/uint16 array.

    raw=True treats the input as packed RGGB and renders a gray-world,
    gamma-corrected preview for visual inspection only.
    """
    img = tensor.detach().float().cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))

    if raw and img.shape[2] == 4:
        r, g, b = img[..., 0], (img[..., 1] + img[..., 2]) / 2.0, img[..., 3]
        rgb = np.stack([r, g, b], axis=2)
        rgb[..., 0] *= (g.mean() + 1e-6) / (r.mean() + 1e-6)
        rgb[..., 2] *= (g.mean() + 1e-6) / (b.mean() + 1e-6)
        img = np.power(np.clip(rgb, 0, 1), 1 / 2.2)

    max_val = 65535.0 if bit_depth == 16 else 255.0
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    return np.clip(np.round(img * max_val), 0, max_val).astype(dtype)


def calculate_psnr(img1, img2, bit_depth=8):
    max_val = (2 ** bit_depth) - 1
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20 * math.log10(max_val / math.sqrt(mse))


def save_image(img, path):
    """Write an (H,W,3) uint8/uint16 array as a TIFF."""
    import tifffile
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tifffile.imwrite(path, img, photometric='rgb')


def save_comparison(images, path):
    """Write a horizontal strip of equally sized (H,W,3) arrays."""
    height = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != height:
            factor = height // im.shape[0]
            im = np.repeat(np.repeat(im, factor, axis=0), factor, axis=1)
        resized.append(im)
    save_image(np.concatenate(resized, axis=1), path)

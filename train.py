import argparse
import os

import torch
from tqdm import tqdm

from data import create_dataloader, create_dataset
from models import ISPModel
from utils.util import (calculate_psnr, parse_config, save_comparison,
                        set_random_seed, setup_logger, tensor2img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--dataroot_RAW', default=None)
    parser.add_argument('--dataroot_RGB', default=None)
    parser.add_argument('--resume', default=None)
    return parser.parse_args()


def validate(model, loader, opt, exp_dir, step, logger):
    bit_depth = opt['datasets']['output_bit_depth']
    out_dir = os.path.join(exp_dir, 'val', f'step_{step}')
    psnr_sum, count = 0.0, 0

    for batch in tqdm(loader, desc='Validating', leave=False):
        names = batch.pop('image_name', None)
        model.feed_data(batch)
        model.test()
        visuals = model.get_visuals()

        for i in range(visuals['output'].size(0)):
            pred = tensor2img(visuals['output'][i], bit_depth)
            gt = tensor2img(visuals['HQ'][i], bit_depth)
            psnr_sum += calculate_psnr(pred, gt, bit_depth)
            count += 1
            if count <= opt['train']['val_save_images']:
                raw = tensor2img(visuals['LQ'][i], bit_depth, raw=True)
                name = names[i] if names else f'{count}'
                save_comparison([raw, pred, gt],
                                os.path.join(out_dir, f'{name}_{count}.tif'))

    psnr = psnr_sum / max(count, 1)
    logger.info(f'[val] step {step} | PSNR {psnr:.4f} dB over {count} crops')
    return psnr


def main():
    args = parse_args()
    opt = parse_config(args.config)
    opt['phase'] = 'train'
    if args.resume:
        opt['resume']['path'] = args.resume
    for phase in ('train', 'val'):
        if args.dataroot_RAW:
            opt['datasets'][phase]['dataroot_RAW'] = args.dataroot_RAW
        if args.dataroot_RGB:
            opt['datasets'][phase]['dataroot_RGB'] = args.dataroot_RGB

    torch.backends.cudnn.benchmark = True
    set_random_seed(opt['seed'])

    exp_dir = os.path.join(opt['experiments_root'], opt['name'])
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    logger = setup_logger(exp_dir)
    logger.info(f'Experiment: {opt["name"]}')

    train_loader = create_dataloader(create_dataset(opt, 'train'), opt, 'train')
    val_loader = create_dataloader(create_dataset(opt, 'val'), opt, 'val')
    logger.info(f'Train samples: {len(train_loader.dataset)} | '
                f'Val samples: {len(val_loader.dataset)}')

    model = ISPModel(opt, logger)
    step, epoch = model.begin_step, model.begin_epoch
    total_iters = opt['train']['n_iter']
    best_psnr, best_step = 0.0, 0

    while step < total_iters:
        epoch += 1
        for batch in train_loader:
            step += 1
            if step > total_iters:
                break

            model.feed_data(batch)
            losses = model.optimize_parameters()
            model.update_learning_rate()

            if step % opt['train']['print_freq'] == 0:
                parts = ' | '.join(f'{k}: {v:.4f}' for k, v in losses.items())
                logger.info(f'[train] epoch {epoch} step {step} '
                            f'lr {model.get_lr():.2e} | {parts}')

            if step % opt['train']['save_freq'] == 0:
                model.save(epoch, step, ckpt_dir)

            if step % opt['train']['val_freq'] == 0:
                psnr = validate(model, val_loader, opt, exp_dir, step, logger)
                if psnr > best_psnr:
                    best_psnr, best_step = psnr, step
                    model.save(epoch, step, os.path.join(exp_dir, 'best'))
                logger.info(f'Best PSNR {best_psnr:.4f} dB at step {best_step}')

    model.save(epoch, step, ckpt_dir)
    logger.info('Training finished.')


if __name__ == '__main__':
    main()

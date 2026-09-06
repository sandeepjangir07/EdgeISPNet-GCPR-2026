from torch.utils.data import DataLoader

from .large_image_dataset import LargeRAWDataset
from .paired_dataset import PairedRAWRGBDataset


def create_dataset(opt, phase):
    cfg = opt['datasets'][phase]
    if phase in ('train', 'val'):
        return PairedRAWRGBDataset(
            dataroot_raw=cfg['dataroot_RAW'],
            dataroot_rgb=cfg['dataroot_RGB'],
            rgb_crop_size=opt['datasets']['rgb_crop_size'],
            crops_per_image=cfg['crops_per_image'],
            split=phase,
            seed=opt.get('seed', 0),
            preprocessed_dir=cfg.get('preprocessed_dir'),
        )
    return LargeRAWDataset(
        dataroot_raw=cfg['dataroot_RAW'],
        tile_size=opt['datasets']['tile_size'],
        border_padding=opt['datasets']['border_padding'],
        border_padding_type=opt['datasets']['border_padding_type'],
        scale=opt['datasets']['scale'],
        cache_images=cfg.get('cache_images', True),
        preprocessed_dir=cfg.get('preprocessed_dir'),
    )


def create_dataloader(dataset, opt, phase):
    cfg = opt['datasets'][phase]
    if phase == 'train':
        return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg.get('num_workers', 2), pin_memory=True,
                          drop_last=True)
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                      num_workers=cfg.get('num_workers', 1), pin_memory=True)


__all__ = ['create_dataset', 'create_dataloader',
           'PairedRAWRGBDataset', 'LargeRAWDataset']

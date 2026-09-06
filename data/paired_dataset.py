import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tifffile import imread
from tqdm import tqdm

from .raw_utils import load_packed, pair_by_stem

GLOBAL_SIZE = (256, 256)


def _load_rgb(path):
    """Load an RGB reference, keeping its native dtype to bound cache memory."""
    img = imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img


def _rgb_scale(dtype):
    if dtype == np.uint16:
        return 65535.0
    if dtype == np.uint8:
        return 255.0
    return 1.0


def _global_view(packed):
    tensor = torch.from_numpy(np.ascontiguousarray(packed)).float().unsqueeze(0)
    return F.interpolate(tensor, size=GLOBAL_SIZE, mode='area').squeeze(0)


class PairedRAWRGBDataset(Dataset):
    """Paired RAW / RGB crops.

    RAW and RGB files are matched by filename stem; no index file is needed.
    Each sample is an rgb_crop_size RGB crop and the co-located
    rgb_crop_size // 2 packed-RAW crop, plus the 256x256 global view.
    """

    def __init__(self, dataroot_raw, dataroot_rgb, rgb_crop_size=512,
                 crops_per_image=64, split='train', seed=0, preprocessed_dir=None):
        self.pairs = pair_by_stem(dataroot_raw, dataroot_rgb)
        self.rgb_crop = rgb_crop_size
        self.raw_crop = rgb_crop_size // 2
        self.crops_per_image = crops_per_image
        self.split = split
        self.seed = seed

        self.raw_cache, self.rgb_cache, self.wb_cache, self.global_cache = {}, {}, {}, {}
        for raw_path, rgb_path in tqdm(self.pairs, desc=f'Loading {split} images'):
            packed, wb = load_packed(raw_path, preprocessed_dir)
            self.raw_cache[raw_path] = packed
            self.wb_cache[raw_path] = wb
            self.global_cache[raw_path] = _global_view(packed)
            self.rgb_cache[raw_path] = _load_rgb(rgb_path)

    def __len__(self):
        return len(self.pairs) * self.crops_per_image

    def _crop_origin(self, index, max_y, max_x):
        if self.split == 'train':
            y = np.random.randint(0, max_y + 1)
            x = np.random.randint(0, max_x + 1)
        else:
            rng = np.random.RandomState(self.seed + index)
            y = rng.randint(0, max_y + 1)
            x = rng.randint(0, max_x + 1)
        return y - y % 2, x - x % 2

    def __getitem__(self, index):
        raw_path, _ = self.pairs[index // self.crops_per_image]
        packed = self.raw_cache[raw_path]
        rgb = self.rgb_cache[raw_path]

        rgb_h = min(rgb.shape[0], packed.shape[1] * 2)
        rgb_w = min(rgb.shape[1], packed.shape[2] * 2)
        y, x = self._crop_origin(index, rgb_h - self.rgb_crop, rgb_w - self.rgb_crop)

        rgb_crop = rgb[y:y + self.rgb_crop, x:x + self.rgb_crop, :]
        rgb_crop = rgb_crop.astype(np.float32) / _rgb_scale(rgb.dtype)
        raw_crop = np.asarray(
            packed[:, y // 2:y // 2 + self.raw_crop, x // 2:x // 2 + self.raw_crop],
            dtype=np.float32)

        sample = {
            'HQ': torch.from_numpy(np.ascontiguousarray(rgb_crop)).permute(2, 0, 1).float(),
            'LQ': torch.from_numpy(np.ascontiguousarray(raw_crop)).float(),
            'Global': self.global_cache[raw_path],
            'WB': torch.from_numpy(self.wb_cache[raw_path]).float(),
            'Index': index,
        }
        if self.split != 'train':
            sample['image_name'] = raw_path.rsplit('/', 1)[-1]
        return sample
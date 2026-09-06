import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .paired_dataset import _global_view
from .raw_utils import RAW_EXTENSIONS, list_files, load_packed, raw_packed_shape

PAD_MODES = {'reflect': 'reflect', 'replicate': 'edge', 'zero': 'constant'}


def tile_origins(height, width, tile):
    origins = []
    for j in range((height + tile - 1) // tile):
        for i in range((width + tile - 1) // tile):
            x = max(0, min(i * tile, width - tile))
            y = max(0, min(j * tile, height - tile))
            if (x, y) not in origins:
                origins.append((x, y))
    origins.sort(key=lambda p: (p[1], p[0]))
    return origins


class LargeRAWDataset(Dataset):
    """Splits each full-resolution RAW into tiles for memory-bounded inference.

    Coordinates and sizes returned by this dataset are in the OUTPUT (RGB) domain,
    i.e. already multiplied by `scale`.
    """

    def __init__(self, dataroot_raw, tile_size=512, border_padding=50,
                 border_padding_type='reflect', scale=2, cache_images=True,
                 preprocessed_dir=None):
        self.tile = tile_size
        self.pad = border_padding
        self.pad_mode = PAD_MODES.get(border_padding_type, 'reflect')
        self.scale = scale
        self.cache_images = cache_images
        self.preprocessed_dir = preprocessed_dir

        if os.path.isfile(dataroot_raw):
            self.paths = [dataroot_raw]
        else:
            self.paths = list_files(dataroot_raw, RAW_EXTENSIONS)
        if not self.paths:
            raise RuntimeError(f'No RAW files found in {dataroot_raw}')

        self.raw_cache, self.wb_cache, self.global_cache, self.shapes = {}, {}, {}, {}
        if cache_images:
            for path in tqdm(self.paths, desc='Caching RAW images'):
                packed, wb = load_packed(path, preprocessed_dir)
                self.raw_cache[path] = packed
                self.wb_cache[path] = wb
                self.global_cache[path] = _global_view(packed)
                self.shapes[path] = packed.shape[1:]
        else:
            for path in self.paths:
                self.shapes[path] = raw_packed_shape(path)

        self.tiles = []
        for path in self.paths:
            h, w = self.shapes[path]
            if h > self.tile or w > self.tile:
                for x, y in tile_origins(h, w, self.tile):
                    self.tiles.append((path, x, y, False))
            else:
                self.tiles.append((path, 0, 0, True))

    def __len__(self):
        return len(self.tiles)

    def _fetch(self, path):
        if path in self.raw_cache:
            return self.raw_cache[path], self.wb_cache[path], self.global_cache[path]
        packed, wb = load_packed(path, self.preprocessed_dir)
        return packed, wb, _global_view(packed)

    def __getitem__(self, index):
        path, x0, y0, is_full = self.tiles[index]
        packed, wb, global_view = self._fetch(path)
        _, height, width = packed.shape

        tile_h = height if is_full else min(self.tile, height)
        tile_w = width if is_full else min(self.tile, width)

        x_start, y_start = max(0, x0 - self.pad), max(0, y0 - self.pad)
        x_end = min(width, x0 + tile_w + self.pad)
        y_end = min(height, y0 + tile_h + self.pad)
        region = np.asarray(packed[:, y_start:y_end, x_start:x_end], dtype=np.float32)

        need_t = max(0, self.pad - (y0 - y_start))
        need_b = max(0, self.pad - (y_end - (y0 + tile_h)))
        need_l = max(0, self.pad - (x0 - x_start))
        need_r = max(0, self.pad - (x_end - (x0 + tile_w)))
        if any((need_t, need_b, need_l, need_r)):
            kwargs = {'mode': self.pad_mode}
            if self.pad_mode == 'constant':
                kwargs['constant_values'] = 0
            region = np.pad(region, ((0, 0), (need_t, need_b), (need_l, need_r)), **kwargs)

        is_last_tile = (index == len(self.tiles) - 1) or (self.tiles[index + 1][0] != path)

        return {
            'LQ': torch.from_numpy(np.ascontiguousarray(region)).float(),
            'Global': global_view,
            'WB': torch.from_numpy(wb).float(),
            'crop_X': x0 * self.scale,
            'crop_Y': y0 * self.scale,
            'height': height * self.scale,
            'width': width * self.scale,
            'Index': index,
            'last_tile': is_last_tile,
            'is_full_image': is_full,
            'image_name': os.path.basename(path),
            'image_path': path,
        }

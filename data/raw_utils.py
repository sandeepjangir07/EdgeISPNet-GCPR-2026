import os
import numpy as np
import rawpy

RAW_EXTENSIONS = ('.iiq', '.arw', '.cr2', '.cr3', '.nef', '.dng', '.rw2', '.srw', '.raf')
RGB_EXTENSIONS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')


def list_files(root, extensions):
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(extensions):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def pair_by_stem(raw_root, rgb_root):
    """Match RAW and RGB files by filename stem. Returns a sorted list of (raw, rgb)."""
    raws = {os.path.splitext(os.path.basename(p))[0]: p
            for p in list_files(raw_root, RAW_EXTENSIONS)}
    rgbs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in list_files(rgb_root, RGB_EXTENSIONS)}
    stems = sorted(set(raws) & set(rgbs))
    if not stems:
        raise RuntimeError(
            f'No RAW/RGB pairs found.\n  RAW dir: {raw_root} ({len(raws)} files)\n'
            f'  RGB dir: {rgb_root} ({len(rgbs)} files)\n'
            'Files are paired by filename stem, e.g. MM011187.iiq <-> MM011187.tif')
    return [(raws[s], rgbs[s]) for s in stems]


def pack_raw(path):
    """Load a Bayer RAW file and return (packed, wb).

    packed: float32 (4, H/2, W/2) in [0, 1], channel order R, G1, G2, B
    wb:     float32 (4,) green-normalised camera white-balance gains
    """
    with rawpy.imread(path) as raw:
        bayer = raw.raw_image_visible.astype(np.float32)
        pattern = raw.raw_pattern
        black = np.array(raw.black_level_per_channel, dtype=np.float32)
        white = float(raw.white_level)

        h, w = bayer.shape
        black_frame = np.zeros((h, w), dtype=np.float32)
        for r in range(2):
            for c in range(2):
                black_frame[r::2, c::2] = black[pattern[r, c]]

        bayer = np.maximum(bayer - black_frame, 0.0)
        bayer = np.clip(bayer / np.maximum(white - black_frame, 1e-5), 0.0, 1.0)

        out_h, out_w = h // 2, w // 2
        packed = np.zeros((4, out_h, out_w), dtype=np.float32)

        cam_wb = np.array(raw.camera_whitebalance, dtype=np.float32)
        if len(cam_wb) >= 4 and cam_wb[3] == 0:
            cam_wb[3] = cam_wb[1]
        wb = np.zeros(4, dtype=np.float32)

        color_desc = raw.color_desc
        green_seen = 0
        for r in range(2):
            for c in range(2):
                idx = pattern[r, c]
                color = chr(color_desc[idx]).upper()
                plane = bayer[r::2, c::2][:out_h, :out_w]
                if color == 'R':
                    packed[0], wb[0] = plane, cam_wb[idx]
                elif color == 'B':
                    packed[3], wb[3] = plane, cam_wb[idx]
                elif color == 'G':
                    slot = 1 if green_seen == 0 else 2
                    packed[slot], wb[slot] = plane, cam_wb[idx]
                    green_seen += 1

        if wb[1] > 1e-5:
            wb = wb / wb[1]

    return packed, wb.astype(np.float32)


def raw_packed_shape(path):
    with rawpy.imread(path) as raw:
        h, w = raw.raw_image_visible.shape
    return h // 2, w // 2


def load_packed(path, preprocessed_dir=None, mmap=True):
    """Return (packed, wb), reading a preprocessed .npy cache when available."""
    if preprocessed_dir:
        stem = os.path.splitext(os.path.basename(path))[0]
        packed_path = os.path.join(preprocessed_dir, f'{stem}.npy')
        wb_path = os.path.join(preprocessed_dir, f'{stem}.wb.npy')
        if os.path.exists(packed_path) and os.path.exists(wb_path):
            packed = np.load(packed_path, mmap_mode='r' if mmap else None)
            return packed, np.load(wb_path)
    return pack_raw(path)

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.raw_utils import RAW_EXTENSIONS, list_files, pack_raw


def main():
    parser = argparse.ArgumentParser(
        description='Decode Bayer RAW files once into packed .npy arrays so that '
                    'training memory-maps them instead of re-decoding every epoch.')
    parser.add_argument('--dataroot_RAW', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    paths = list_files(args.dataroot_RAW, RAW_EXTENSIONS)
    if not paths:
        raise SystemExit(f'No RAW files found in {args.dataroot_RAW}')
    os.makedirs(args.output, exist_ok=True)

    total_bytes = 0
    for path in tqdm(paths, desc='Preprocessing'):
        stem = os.path.splitext(os.path.basename(path))[0]
        packed_path = os.path.join(args.output, f'{stem}.npy')
        wb_path = os.path.join(args.output, f'{stem}.wb.npy')
        if os.path.exists(packed_path) and not args.overwrite:
            total_bytes += os.path.getsize(packed_path)
            continue

        packed, wb = pack_raw(path)
        np.save(packed_path, packed.astype(args.dtype))
        np.save(wb_path, wb)
        total_bytes += os.path.getsize(packed_path)

    print(f'{len(paths)} images -> {args.output} ({total_bytes / 1e9:.1f} GB, {args.dtype})')
    print('Set preprocessed_dir in your config to this path.')


if __name__ == '__main__':
    main()

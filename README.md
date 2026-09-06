# EdgeISPNet

Ultra-lightweight neural ISP for high-resolution RAW-to-RGB conversion on edge devices.

Official implementation of *"EdgeISPNet: Ultra-Lightweight Neural ISP for
High-Resolution RAW-to-RGB High Altitude Imaging Systems"* (GCPR 2026).

EdgeISPNet maps 16-bit Bayer RAW to RGB using only 3x3 convolutions, element-wise
operations and PixelShuffle, so it compiles natively on edge accelerators. Multi-branch
training blocks collapse into plain convolutions at inference time
(**73,578 training parameters -> 28,407 inference parameters**). A Global Scene
Representation Module (GSRM) is evaluated once per image and reused for every tile,
which makes tiled inference of very large images bit-exact with whole-image inference.

---

## Install

```bash
git clone https://github.com/sandeepjangir07/EdgeISPNet-GCPR-2026
cd EdgeISPNet-GCPR-2026
pip install -r requirements.txt
```

## Data layout

You provide two directories per split: one of Bayer RAW files, one of RGB reference
images. **Files are paired by filename stem** — no index or crop-coordinate file is
needed.

```
dataset/
├── train/
│   ├── raw/    MM011187.iiq  MM011188.iiq  ...
│   └── rgb/    MM011187.tif  MM011188.tif  ...
├── val/
│   ├── raw/
│   └── rgb/
└── test/
    └── raw/
```

RAW: `.iiq .arw .cr2 .cr3 .nef .dng .rw2 .srw .raf` (anything `rawpy` reads).
RGB: `.tif .tiff .png .jpg`, 8- or 16-bit, at **twice** the packed-RAW resolution
(i.e. the native sensor resolution).

RAW files are black-level subtracted, normalised by the white level, and packed into a
4-channel `R, G1, G2, B` tensor at half spatial resolution. Camera white-balance gains
are read from the file metadata and green-normalised so `G = 1`.

### Optional: preprocess for faster training

Decoding a 150 MP IIQ file takes seconds. Decode once instead:

```bash
python scripts/preprocess.py --dataroot_RAW dataset/train/raw --output cache/train
python scripts/preprocess.py --dataroot_RAW dataset/val/raw   --output cache/val
```

Then set `preprocessed_dir: cache/train` in the config. Arrays are memory-mapped, so
they are not all resident in RAM. `--dtype float16` (default) halves the cache size;
Use `--dtype float32` for an exact cache.

---

## Training

Two phases, matching the paper.

**Phase 1** — Charbonnier loss, 500k iterations:

```bash
python train.py -c config/train_phase1.yml \
  --dataroot_RAW dataset/train/raw --dataroot_RGB dataset/train/rgb
```

**Phase 2** — fine-tune with MS-SSIM, perceptual and adversarial losses, 400k iterations:

```bash
python train.py -c config/train_phase2.yml \
  --resume experiments/EdgeISPNet_phase1/best/edgeispnet_step500000.pth
```

Edit the config directly, or override the data roots and checkpoint from the command
line. Checkpoints, logs and validation strips are written to
`experiments/<name>/`; the best-PSNR checkpoint is kept separately in
`experiments/<name>/best/`.

Each training sample is a 512x512 RGB crop with the co-located 256x256 packed-RAW crop
and a 256x256 area-downsampled global view of the full frame. Crop positions are drawn
at random during training and seeded per index for validation, so validation is
reproducible across runs.

---

## Inference

```bash
python test.py -c config/test.yml \
  --dataroot_RAW dataset/test/raw \
  --checkpoint experiments/EdgeISPNet_phase2/best/edgeispnet_step400000.pth \
  --output results/
```

Full-resolution images are split into tiles, processed, and reassembled into a single
TIFF. Before inference the network merges its training branches into plain convolutions
(`set_deploy(True)`), and the GSRM runs once per image with its output reused for every
tile of that image.

### Tiling and `border_padding`

Each tile is read with a `border_padding` ring of surrounding context, which is trimmed
after the forward pass. 

---

## Citation

```bibtex
@inproceedings{jangir2026edgeispnet,
  title     = {EdgeISPNet: Ultra-Lightweight Neural ISP for High-Resolution
               RAW-to-RGB High Altitude Imaging Systems},
  author    = {Jangir, Sandeep Kumar and Gstaiger, Veronika and Bahmanyar, Reza},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  year      = {2026}
}
```

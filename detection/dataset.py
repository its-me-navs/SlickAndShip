"""
Dataset loader matching the actual Zenodo "Sentinel-1 SAR Oil spill image
dataset for train, validate, and test deep learning models" (Parts I-III,
Trujillo-Acatitla et al.) — this is the dataset linked from the PS.

Expected folder layout after downloading + extracting all three parts:

  data/sar_raw/
    01_Train_Val_Oil_Spill_images/       *.tif   (2048x2048x2, Sigma0 dB)
    01_Train_Val_Oil_Spill_mask/         *.tif   (2048x2048, 1=oil / 0=background)
    01_Train_Val_No_Oil_Images/          *.tif
    01_Train_Val_No_Oil_mask/            *.tif
    01_Train_Val_Lookalike_Images/       *.tif   (name may vary slightly — check
    01_Train_Val_Lookalike_mask/         *.tif    your actual extracted folder names)
    03_Test_*                            (Part III, same pattern, used as held-out test)

If your extracted folder names differ slightly from the above (Zenodo listings
occasionally rename between dataset versions), just point IMAGE_DIRS/MASK_DIRS
below at whatever you actually got — the loader only cares about matching
image/mask filename pairs, not exact folder names.

Lookalikes and no-oil scenes are included as negative examples with all-zero
masks (already the case for No-Oil; Lookalike masks are handled the same way
so the model learns oil-vs-not-oil, which is what item (a) of the PS needs).

Since the raw images are 2048x2048, training on random 256x256 crops keeps
GPU memory and epoch time reasonable — bump CROP_SIZE up if you have the
VRAM for it.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None  # only needed if your TIFFs are true GeoTIFFs; plain
    # TIFF reading still works via PIL/tifffile as a fallback (see _read_tiff)

CROP_SIZE = 256


def _read_tiff(path: str) -> np.ndarray:
    """Reads a TIFF as a numpy array, using rasterio if available (keeps any
    georeferencing metadata, which the inference step needs), falling back
    to tifffile/PIL if not."""
    if rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
            return np.moveaxis(arr, 0, -1)  # -> (H, W, bands)
    else:
        import tifffile
        return tifffile.imread(path)


class SARSpillDataset(Dataset):
    def __init__(self, image_dirs: list[str], mask_dirs: list[str], crop_size: int = CROP_SIZE,
                 augment: bool = True):
        assert len(image_dirs) == len(mask_dirs)
        self.pairs = []
        for img_dir, mask_dir in zip(image_dirs, mask_dirs):
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")) +
                                glob.glob(os.path.join(img_dir, "*.tiff")))
            for ip in img_paths:
                mp = os.path.join(mask_dir, os.path.basename(ip))
                if os.path.exists(mp):
                    self.pairs.append((ip, mp))
        if not self.pairs:
            raise FileNotFoundError(
                "No image/mask pairs found. Check that image_dirs/mask_dirs point at your "
                "actual extracted dataset folders and that filenames match between them."
            )
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = _read_tiff(img_path).astype(np.float32)      # (H, W, 2) Sigma0 dB
        mask = _read_tiff(mask_path).astype(np.float32)     # (H, W) or (H, W, 1)
        if mask.ndim == 3:
            mask = mask[..., 0]

        # normalize dB values to roughly [0, 1] — Sigma0 in dB for ocean
        # scenes typically falls in about [-35, 5]; clip then rescale
        img = np.clip(img, -35, 5)
        img = (img + 35) / 40.0

        h, w = mask.shape
        cs = self.crop_size
        if h > cs and w > cs:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
            img = img[top:top + cs, left:left + cs, :]
            mask = mask[top:top + cs, left:left + cs]

        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                img, mask = img[::-1, :, :].copy(), mask[::-1, :].copy()

        img_t = torch.from_numpy(np.moveaxis(img, -1, 0))   # (2, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)
        return img_t, mask_t

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

Real-world note: the archives don't always extract with Zenodo's listed
folder names — e.g. 7-Zip may give you 'Lookalike' and 'No_oil' (no
"image"/"img" in the name at all) alongside 'Mask_lookalike' and
'Mask_no_oil'. build_datasets() below is written to handle this: any
directory containing .tif/.tiff files is classified purely by whether
"mask" appears in its name — no requirement that "image"/"img" appear in
the non-mask one. See _discover_pairs().

Lookalikes and no-oil scenes are included as negative examples with all-zero
masks (already the case for No-Oil; Lookalike masks are handled the same way
so the model learns oil-vs-not-oil, which is what item (a) of the PS needs).

Since the raw images are 2048x2048, training on random 256x256 crops keeps
GPU memory and epoch time reasonable — bump CROP_SIZE up if you have the
VRAM for it.
"""

import difflib
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
    """
    Two ways to construct this:
      - image_dirs + mask_dirs: scans each pair of directories for matching
        filenames (original interface).
      - pairs: a precomputed list of (image_path, mask_path) tuples — used
        by build_datasets() below after it's done the train/val/test split,
        so each split gets an exact, non-overlapping list of files rather
        than re-scanning whole directories.
    """

    def __init__(self, image_dirs: list[str] | None = None, mask_dirs: list[str] | None = None,
                 pairs: list[tuple[str, str]] | None = None,
                 crop_size: int = CROP_SIZE, augment: bool = True):
        if pairs is not None:
            self.pairs = list(pairs)
        else:
            assert image_dirs is not None and mask_dirs is not None and len(image_dirs) == len(mask_dirs)
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
                "No image/mask pairs found. Check that image_dirs/mask_dirs (or data_root, "
                "if using build_datasets) point at your actual extracted dataset folders and "
                "that filenames match between image and mask directories."
            )
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    # fraction of the time a crop is deliberately centered on a real oil
    # region (when the full mask has one) rather than placed uniformly at
    # random — without this, random crops from a 2048x2048 image mostly miss
    # a spill that only covers a small area, starving the model of positive
    # examples and pushing it toward always predicting background (see
    # diagnose_unet.py / the training-collapse this was added to fix).
    POSITIVE_CROP_PROB = 0.85

    def _pick_crop_origin(self, mask: np.ndarray, h: int, w: int, cs: int) -> tuple[int, int]:
        ys, xs = np.where(mask > 0)
        if len(ys) > 0 and random.random() < self.POSITIVE_CROP_PROB:
            # anchor near a random oil pixel, with jitter so the crop isn't
            # always perfectly centered on it (keeps some edge/context
            # variety), then clip so the crop stays inside the image.
            idx = random.randrange(len(ys))
            anchor_y, anchor_x = int(ys[idx]), int(xs[idx])
            jitter = cs // 2
            top = anchor_y - cs // 2 + random.randint(-jitter, jitter)
            left = anchor_x - cs // 2 + random.randint(-jitter, jitter)
            top = max(0, min(h - cs, top))
            left = max(0, min(w - cs, left))
            return top, left
        # no oil in this image at all, or we're in the (1 - POSITIVE_CROP_PROB)
        # slice deliberately kept fully random for background/context diversity
        return random.randint(0, h - cs), random.randint(0, w - cs)

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
            top, left = self._pick_crop_origin(mask, h, w, cs)
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


# ──────────────────────────────────────────────────────────────────────────
# build_datasets() — discovers the Zenodo folder layout under data_root and
# returns (train_dataset, val_dataset, test_dataset), which is what
# training.py imports and calls.
# ──────────────────────────────────────────────────────────────────────────

def _leaf_image_dirs(root: str) -> list[str]:
    """Every directory anywhere under root that directly contains at least
    one .tif/.tiff file — regardless of what it's named."""
    leaves = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower().endswith((".tif", ".tiff")) for f in filenames):
            leaves.append(dirpath)
    return leaves


def _stem(name: str) -> str:
    """Lowercases and strips common noise tokens so e.g. '01_Train_Val_Oil_
    Spill_mask' and 'Oil_Spill' (or 'oil') produce comparable stems for
    fuzzy matching."""
    s = name.lower()
    for kw in ["01_train_val_", "train_val_", "mask", "masks", "images", "image", "img", "01_", "03_test_", "test_"]:
        s = s.replace(kw, "")
    return s.strip("_- ")

def _is_mask_dir(dirpath: str, data_root: str) -> bool:
    """True if 'mask' appears in any path component between data_root and
    dirpath — handles both flat layouts (Oil_Spill_mask) and nested ones
    (Mask/Oil), where the keyword may live on a parent folder, not the leaf."""
    rel = os.path.relpath(dirpath, data_root)
    return any("mask" in part.lower() for part in rel.split(os.sep))

def _is_test_dir(dirpath: str, data_root: str) -> bool:
    """True if 'test' appears in any path component between data_root and
    dirpath. Used to scope mask-dir candidates to the same test/trainval
    subtree as the image dir being matched, so e.g. Test_Images/Lookalike
    can never be scored against trainval's Lookalike_mask."""
    rel = os.path.relpath(dirpath, data_root)
    return any("test" in part.lower() for part in rel.split(os.sep))

def _mask_suffix_variants(image_filename: str) -> list[str]:
    """Candidate mask filenames for a given image filename, covering both
    naming conventions seen across dataset parts: exact match (Part I/II,
    e.g. 00000.tif -> 00000.tif) and a '_segmentation' suffix before the
    extension (Part III, e.g. 00000.tif -> 00000_segmentation.tif)."""
    stem, ext = os.path.splitext(image_filename)
    return [image_filename, f"{stem}_segmentation{ext}"]

def _discover_pairs(data_root: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Walks data_root, finds every directory that actually contains .tif/.tiff
    files, and classifies each as a mask dir (name contains 'mask') or an
    image dir (everything else — no requirement that 'image'/'img' appear
    in the name, since real extractions don't always include that word).
    Pairs each image dir with its best-matching mask dir by fuzzy name
    similarity. Directories whose path contains 'test' (case-insensitive)
    go into the held-out test pool; everything else into train/val.

    Returns (trainval_pairs, test_pairs), each a list of (img_path, mask_path).
    """
    leaf_dirs = _leaf_image_dirs(data_root)
    if not leaf_dirs:
        raise FileNotFoundError(
            f"No directories containing .tif/.tiff files found under {data_root}. "
            f"Make sure you've extracted the archives (not just downloaded the .7z files)."
        )

    mask_dirs = [d for d in leaf_dirs if _is_mask_dir(d, data_root)]
    image_dirs = [d for d in leaf_dirs if not _is_mask_dir(d, data_root)]

    if not image_dirs or not mask_dirs:
        raise FileNotFoundError(
            f"Found {len(leaf_dirs)} directories with .tif files under {data_root}, but "
            f"couldn't split them into image/mask groups (image_dirs={len(image_dirs)}, "
            f"mask_dirs={len(mask_dirs)}). Found dirs: {leaf_dirs}"
        )

    trainval_pairs, test_pairs = [], []

    for img_dir in image_dirs:
        img_is_test = _is_test_dir(img_dir, data_root)
        # Only ever consider mask dirs from the same test/trainval subtree —
        # prevents ties (e.g. two dirs both named "Lookalike") from being
        # broken by arbitrary os.walk order, and prevents a test image dir
        # from ever being scored against a trainval mask dir or vice versa.
        candidate_mask_dirs = [d for d in mask_dirs if _is_test_dir(d, data_root) == img_is_test]
        if not candidate_mask_dirs:
            print(f"  [build_datasets] WARNING: no {'test' if img_is_test else 'train/val'}-scoped "
                  f"mask dir candidates for '{img_dir}' — skipping.")
            continue

        img_stem = _stem(os.path.basename(img_dir))
        best_mask_dir, best_score = None, -1.0
        for mask_dir in candidate_mask_dirs:
            mask_stem = _stem(os.path.basename(mask_dir))
            score = difflib.SequenceMatcher(None, img_stem, mask_stem).ratio()
            if score > best_score:
                best_score, best_mask_dir = score, mask_dir

        if best_mask_dir is None or best_score < 0.4:
            print(f"  [build_datasets] WARNING: no confident mask-dir match for '{img_dir}' "
                  f"(best candidate: {best_mask_dir}, score={best_score:.2f}) — skipping.")
            continue

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")) +
                            glob.glob(os.path.join(img_dir, "*.tiff")))
        pairs = []
        for ip in img_paths:
            for candidate in _mask_suffix_variants(os.path.basename(ip)):
                mp = os.path.join(best_mask_dir, candidate)
                if os.path.exists(mp):
                    pairs.append((ip, mp))
                    break

        is_test = img_is_test
        print(f"  [build_datasets] {img_dir}  <->  {best_mask_dir}  "
              f"(score={best_score:.2f}, {len(pairs)} pairs, {'TEST' if is_test else 'train/val'})")
        (test_pairs if is_test else trainval_pairs).extend(pairs)

    return trainval_pairs, test_pairs


def build_datasets(data_root: str, seed: int = 42, crop_size: int = CROP_SIZE,
                    val_fraction: float = 0.15, test_fraction_fallback: float = 0.15):
    """
    Discovers the extracted Zenodo dataset under data_root, pools all
    categories (oil / no-oil / lookalike), and splits into train/val/test.

    If a genuine held-out test folder (path containing 'test') is found,
    it's used as-is for test_dataset. If none is found (e.g. you've only
    downloaded Parts I-II so far), trainval is instead split three ways
    (train/val/test_fraction_fallback) with a printed warning, so training
    still runs rather than crashing on a missing test set.
    """
    print(f"[build_datasets] Scanning {data_root} ...")
    trainval_pairs, test_pairs = _discover_pairs(data_root)

    if not trainval_pairs:
        raise FileNotFoundError(f"No train/val image-mask pairs discovered under {data_root}.")

    rng = random.Random(seed)
    rng.shuffle(trainval_pairs)

    if test_pairs:
        n_val = max(1, int(len(trainval_pairs) * val_fraction))
        val_pairs = trainval_pairs[:n_val]
        train_pairs = trainval_pairs[n_val:]
    else:
        print("  [build_datasets] WARNING: no dedicated test folder found — "
              "carving a test split out of train/val data instead.")
        n_val = max(1, int(len(trainval_pairs) * val_fraction))
        n_test = max(1, int(len(trainval_pairs) * test_fraction_fallback))
        val_pairs = trainval_pairs[:n_val]
        test_pairs = trainval_pairs[n_val:n_val + n_test]
        train_pairs = trainval_pairs[n_val + n_test:]

    print(f"[build_datasets] train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    train_dataset = SARSpillDataset(pairs=train_pairs, crop_size=crop_size, augment=True)
    val_dataset = SARSpillDataset(pairs=val_pairs, crop_size=crop_size, augment=False)
    test_dataset = SARSpillDataset(pairs=test_pairs, crop_size=crop_size, augment=False)

    return train_dataset, val_dataset, test_dataset
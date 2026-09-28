"""
Run this BEFORE training.py, in its own Colab cell, right after extracting
the dataset. Confirms build_datasets() actually discovered and paired the
image/mask folders correctly (catches naming mismatches, empty splits, or
a bad extraction) before you spend time on a real training run.

Usage (in Colab, after the !7z extraction steps):
    !python verify_dataset.py --data data/sar_raw
"""

import argparse
import numpy as np

from detection.dataset import build_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("DRY RUN — verifying dataset discovery before training")
    print("=" * 60)

    train_ds, val_ds, test_ds = build_datasets(data_root=args.data, seed=42, crop_size=256)

    print("\n--- Split sizes ---")
    print(f"Train: {len(train_ds)}")
    print(f"Val:   {len(val_ds)}")
    print(f"Test:  {len(test_ds)}")

    if len(train_ds) == 0:
        print("\n⚠️  WARNING: train split is empty — check that image/mask "
              "folders were actually extracted under --data.")
        return

    print("\n--- Checking a few pairs for sanity ---")
    for i in [0, len(train_ds) // 2, len(train_ds) - 1]:
        img_path, mask_path = train_ds.pairs[i]
        img, mask = train_ds[i]
        print(f"  [{i}] img={img_path}")
        print(f"       mask={mask_path}")
        print(f"       img shape={tuple(img.shape)} range=[{img.min():.3f}, {img.max():.3f}]")
        print(f"       mask shape={tuple(mask.shape)} unique_values={np.unique(mask.numpy())} "
              f"oil_pixel_fraction={mask.numpy().mean():.4f}")

    print("\n✓ If shapes look like (2, 256, 256) for images and (1, 256, 256) "
          "for masks, with mask values in {0.0, 1.0}, discovery worked correctly.")
    print("  Ready to run training.py.")


if __name__ == "__main__":
    main()
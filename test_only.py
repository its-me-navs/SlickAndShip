"""
Runs just the held-out test-set evaluation against an existing checkpoint,
single-threaded (num_workers=0) to avoid the OpenBLAS multi-process memory
crash that can happen when DataLoader workers each spawn their own BLAS
thread pool. Use this if training.py's automatic post-training test step
crashed — your checkpoint was already saved before that step runs, so
nothing is lost, this just re-does the final scoring safely.

Usage: python test_only.py --data data/sar_raw --checkpoint best_unet.pt
"""

import argparse
import os

# reduce BLAS thread contention proactively, even single-threaded
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from torch.utils.data import DataLoader

from detection.dataset import build_datasets
from detection.unet_model import UNet
from detection.training import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="best_unet.pt")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cpu")

    _, _, test_dataset = build_datasets(data_root=args.data, seed=42, crop_size=args.crop_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=2, out_channels=1, base=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, "
          f"best_positive_iou={ckpt.get('best_positive_iou', ckpt.get('best_iou')):.4f}\n")

    test_loss, test_iou, test_pos_iou, n_pos = evaluate(model, test_loader, device, desc="test")

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")
    print(f"Test loss              : {test_loss:.4f}")
    print(f"Test IoU (overall)     : {test_iou:.4f}")
    print(f"Test IoU (oil-only, n={n_pos}) : {test_pos_iou:.4f}  <- this is the number that matters")
    print("==============================")


if __name__ == "__main__":
    main()
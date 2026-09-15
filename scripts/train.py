#!/usr/bin/env python3
"""
train.py -- trains the learned stage of the Mora SP Cup 2026 denoiser.

The network is NOT asked to denoise raw pixels.  It operates inside the
variance-stabilised, opponent-colour domain produced by the classical
front-end in ``noise_model.py``:

    noisy RGB
      -> impulse / hot-pixel repair         (classical)
      -> blind Poisson-Gaussian estimation  (classical)  alpha, sigma_r^2
      -> Generalised Anscombe Transform     (classical)  noise -> unit variance
      -> orthonormal opponent rotation      (classical)  luma / chroma split
      -> [ NETWORK predicts the clean signal ]
      -> inverse rotation                   (classical)
      -> exact unbiased inverse GAT         (classical)
    clean RGB

Why: the measured noise is signal dependent (Var = alpha*clean + sigma_r^2)
with alpha and sigma_r varying per image.  A network trained on raw pixels
must learn that variation from only 460 images.  After the GAT the noise is
unit variance everywhere, so the network only has to learn image structure.
That is what makes a 0.5M-parameter model sufficient, and what keeps CPU
inference fast enough for the competition's runtime requirement.

Images 001-420 are used for training; 421-460 are held out and never
trained on.

Usage
-----
    python scripts/train.py --data_root competition_data/public

The first run builds a preprocessed cache on disk (about 5 GB).  Later runs
reuse it, so retraining is fast.  Delete the cache folder to rebuild it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_model import (OPPONENT, estimate_poisson_gaussian, gat_forward,
                         repair_impulses)

SEED = 2026
VAL_IDS = {f"{i:03d}" for i in range(421, 461)}     # held out, never trained on


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Shared front-end: identical at training and at inference time
# ---------------------------------------------------------------------------

def to_model_domain(noisy_u8: np.ndarray, clean_u8: np.ndarray | None = None):
    """Apply the classical front-end.

    The Poisson-Gaussian parameters are estimated from the NOISY image only,
    never from the clean one, so training inputs match inference inputs
    exactly.
    """
    work = repair_impulses(noisy_u8.astype(np.float32))
    params = estimate_poisson_gaussian(np.clip(work, 0, 255))

    noisy_gat = np.stack([gat_forward(work[..., c], a, v)
                          for c, (a, v) in enumerate(params)], axis=-1)
    noisy_opp = noisy_gat @ OPPONENT.T

    if clean_u8 is None:
        return noisy_opp, params

    clean_gat = np.stack([gat_forward(clean_u8[..., c].astype(np.float32), a, v)
                          for c, (a, v) in enumerate(params)], axis=-1)
    return noisy_opp, clean_gat @ OPPONENT.T, params


# ---------------------------------------------------------------------------
# Disk-backed cache
# ---------------------------------------------------------------------------

def build_cache(data_root: Path, cache_dir: Path, ids: list[str]) -> None:
    """Preprocess every training pair once and store it as a memory-map.

    The front-end is deterministic, so running it every epoch would be pure
    waste.  Caching it as float16 on disk keeps RAM usage near zero (the OS
    pages the file in on demand) while still allowing a fresh random crop
    from anywhere in the image on every sample.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = np.asarray(Image.open(data_root / "ground_truth" / f"{ids[0]}.png")
                       .convert("RGB"))
    height, width = probe.shape[:2]
    shape = (len(ids), height, width, 3)

    print(f"Building cache for {len(ids)} images "
          f"({2 * np.prod(shape) * 2 / 1e9:.1f} GB on disk). One-off cost.")

    noisy_mm = np.lib.format.open_memmap(cache_dir / "noisy.npy", mode="w+",
                                         dtype=np.float16, shape=shape)
    clean_mm = np.lib.format.open_memmap(cache_dir / "clean.npy", mode="w+",
                                         dtype=np.float16, shape=shape)

    start = time.time()
    for index, image_id in enumerate(ids):
        noisy = np.asarray(Image.open(data_root / "noisy" / f"{image_id}_noise.png")
                           .convert("RGB"), dtype=np.uint8)
        clean = np.asarray(Image.open(data_root / "ground_truth" / f"{image_id}.png")
                           .convert("RGB"), dtype=np.uint8)
        x, y, _ = to_model_domain(noisy, clean)
        noisy_mm[index] = x.astype(np.float16)
        clean_mm[index] = y.astype(np.float16)

        if (index + 1) % 20 == 0 or index + 1 == len(ids):
            done = index + 1
            rate = (time.time() - start) / done
            print(f"  {done}/{len(ids)}  ({rate * (len(ids) - done) / 60:.1f} "
                  f"min remaining)", flush=True)

    noisy_mm.flush()
    clean_mm.flush()
    (cache_dir / "ids.json").write_text(json.dumps(ids))
    print("Cache built.\n")


class PatchDataset(Dataset):
    """Random crops drawn from the memory-mapped cache.

    The memory-maps are opened LAZILY, on first access inside whichever
    process is doing the reading.  On Windows, DataLoader workers are
    spawned rather than forked, so the dataset object has to be pickled and
    sent to each worker -- and an open memory-map cannot be pickled.  Keeping
    the handles out of the object's state until they are needed makes the
    dataset picklable and lets num_workers > 0 work on every platform.
    """

    def __init__(self, cache_dir: Path, patch: int = 128,
                 patches_per_image: int = 24):
        self.cache_dir = Path(cache_dir)
        self.patch = patch
        self.per_image = patches_per_image
        self._noisy = None
        self._clean = None
        # read the length without keeping a handle open
        with open(self.cache_dir / "ids.json") as fh:
            self.count = len(json.load(fh))

    def _maps(self):
        if self._noisy is None:
            self._noisy = np.load(self.cache_dir / "noisy.npy", mmap_mode="r")
            self._clean = np.load(self.cache_dir / "clean.npy", mmap_mode="r")
        return self._noisy, self._clean

    def __len__(self):
        return self.count * self.per_image

    def __getitem__(self, index):
        noisy, clean = self._maps()
        i = index // self.per_image
        p = self.patch
        top = random.randint(0, noisy.shape[1] - p)
        left = random.randint(0, noisy.shape[2] - p)

        x = np.asarray(noisy[i, top:top + p, left:left + p], dtype=np.float32)
        y = np.asarray(clean[i, top:top + p, left:left + p], dtype=np.float32)

        k = random.randint(0, 3)                   # dihedral augmentation
        if k:
            x, y = np.rot90(x, k), np.rot90(y, k)
        if random.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]

        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))))


# ---------------------------------------------------------------------------
# The network: a small residual U-Net
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.body(x)


class SmallUNet(nn.Module):
    """Three-scale U-Net predicting the residual (i.e. the noise).

    Predicting the residual rather than the clean image means the network
    starts from an identity mapping, which trains far faster on a small
    dataset.  Width 32 gives roughly 0.5M parameters -- small enough to run
    quickly on a plain CPU, as the competition requires.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.enc1 = ConvBlock(3, w)
        self.enc2 = ConvBlock(w, w * 2)
        self.enc3 = ConvBlock(w * 2, w * 4)
        self.dec2 = ConvBlock(w * 4 + w * 2, w * 2)
        self.dec1 = ConvBlock(w * 2 + w, w)
        self.out = nn.Conv2d(w, 3, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat(
            [F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat(
            [F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        return x - self.out(d1)                    # residual connection


def charbonnier(pred, target, eps: float = 1e-3):
    """Smooth L1-like loss, robust to the outliers left by clipping."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps * eps))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("competition_data/public"))
    ap.add_argument("--cache_dir", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "model.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int,
                    default=0 if sys.platform == "win32" else 2,
                    help="data-loader workers; 0 is safest on Windows")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N training images (quick test)")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + (torch.cuda.get_device_name(0) if device.type == "cuda"
                        else "CPU"))

    all_ids = sorted(p.stem for p in (args.data_root / "ground_truth").glob("*.png"))
    train_ids = [i for i in all_ids if i not in VAL_IDS]
    if args.limit:
        train_ids = train_ids[:args.limit]
    print(f"Training on {len(train_ids)} images; "
          f"{len(VAL_IDS & set(all_ids))} held out for validation.")

    cached_ids = None
    if (args.cache_dir / "ids.json").exists():
        cached_ids = json.loads((args.cache_dir / "ids.json").read_text())
    if cached_ids != train_ids:
        build_cache(args.data_root, args.cache_dir, train_ids)
    else:
        print(f"Reusing cache in {args.cache_dir}\n")

    loader = DataLoader(PatchDataset(args.cache_dir, args.patch),
                        batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)

    model = SmallUNet(args.width).to(device)
    print(f"Model parameters: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = charbonnier(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()

        mean_loss = running / seen
        marker = ""
        if mean_loss < best:
            best = mean_loss
            torch.save({"state_dict": model.state_dict(), "width": args.width,
                        "seed": SEED, "epoch": epoch}, args.out)
            marker = "  <- saved"
        print(f"epoch {epoch:3d}/{args.epochs}   loss {mean_loss:.5f}   "
              f"{time.time() - t0:.0f}s{marker}", flush=True)

    print(f"\nBest checkpoint written to {args.out}")
    print("Run scripts/denoise.py -- it picks the checkpoint up automatically.")
    print("IMPORTANT: validate on images 421-460 before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

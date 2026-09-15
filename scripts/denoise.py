#!/usr/bin/env python3
"""
denoise.py -- Mora SP Cup 2026, low-light image denoising.

Reads every ``<id>_noise.png`` from the noisy directory, denoises it, and
writes ``<id>.png`` (suffix removed, as required) to the output directory.

Usage
-----
    python scripts/denoise.py --noise_dir  <in> --denoised_dir <out>
    python scripts/denoise.py --input_dir  <in> --output_dir   <out>

Both spellings are accepted: the participant handbook specifies the first
pair, the provided baseline uses the second.

Runs on CPU by default.  If a trained checkpoint and PyTorch are both
available the learned stage is used, on GPU when one is present and on CPU
otherwise; if either is missing the pipeline falls back to a purely
classical denoiser so the script always produces valid output.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_model import (OPPONENT, OPPONENT_INV, destripe,
                         estimate_poisson_gaussian, gat_forward,
                         gat_inverse_exact, repair_impulses)

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
cv2.setRNGSeed(SEED)

VALID_EXT = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"

# The GAT maps the noise to unit variance.  Working in 8-bit for the fast
# OpenCV NLM implementation needs an integer scale; 12 DN per noise sigma
# keeps the full GAT range inside 0..255 while leaving enough resolution.
GAT_SCALE = 12.0


# --------------------------------------------------------------------------
# classical denoiser, applied in the variance-stabilised domain
# --------------------------------------------------------------------------

def nlm_unit_variance(plane: np.ndarray, strength: float) -> np.ndarray:
    """Non-local means on a plane whose noise has unit standard deviation."""
    offset = plane.min()
    scaled = (plane - offset) * GAT_SCALE
    if scaled.max() > 255.0:                       # keep inside 8-bit range
        squeeze = 255.0 / scaled.max()
        scaled *= squeeze
    else:
        squeeze = 1.0
    as_u8 = np.clip(scaled, 0, 255).astype(np.uint8)
    filtered = cv2.fastNlMeansDenoising(
        as_u8, None, h=strength * GAT_SCALE * squeeze,
        templateWindowSize=7, searchWindowSize=21)
    return filtered.astype(np.float32) / (GAT_SCALE * squeeze) + offset




def denoise_classical(noisy_u8: np.ndarray, strength: float = 0.9,
                      chroma_boost: float = 3.0,
                      do_destripe: bool = False,
                      do_impulse: bool = True) -> np.ndarray:
    """Full classical pipeline.  Input uint8 RGB, output float DN in [0, 255]."""
    work = noisy_u8.astype(np.float32)
    if do_impulse:
        work = repair_impulses(work)
    if do_destripe:
        work = destripe(work)

    params = estimate_poisson_gaussian(np.clip(work, 0, 255))

    # 1. variance stabilisation, per channel
    stabilised = np.stack(
        [gat_forward(work[..., c], a, v) for c, (a, v) in enumerate(params)],
        axis=-1)

    # 2. rotate to opponent colour space (noise stays unit variance)
    opponent = stabilised @ OPPONENT.T

    # 3. denoise: gently on luma, hard on the two chroma planes
    strengths = (strength, strength * chroma_boost, strength * chroma_boost)
    filtered = np.stack(
        [nlm_unit_variance(opponent[..., k], strengths[k]) for k in range(3)],
        axis=-1)

    # 4. rotate back and undo the stabilisation
    rgb = filtered @ OPPONENT_INV.T
    planes = [gat_inverse_exact(rgb[..., c], a, v)
              for c, (a, v) in enumerate(params)]
    return np.clip(np.stack(planes, axis=-1), 0.0, 255.0)


# --------------------------------------------------------------------------
# learned denoiser (used when a checkpoint is present)
# --------------------------------------------------------------------------

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "model.pt"


def load_model(path: Path, force_cpu: bool = False):
    """Return (model, torch) if a usable checkpoint exists, else (None, None).

    Any failure here -- PyTorch missing, checkpoint missing, checkpoint
    unreadable -- falls back to the classical path rather than crashing, so
    the script always produces valid output.
    """
    if not path.exists():
        return None, None
    try:
        import torch
        from train import SmallUNet
    except Exception as exc:                       # pragma: no cover
        print(f"note: learned stage unavailable ({exc}); using classical path")
        return None, None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = SmallUNet(ckpt.get("width", 32))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        torch.set_grad_enabled(False)
        if torch.cuda.is_available() and not force_cpu:
            model = model.cuda()
        return model, torch
    except Exception as exc:                       # pragma: no cover
        print(f"note: could not load {path} ({exc}); using classical path")
        return None, None


def denoise_learned(noisy_u8: np.ndarray, model, torch, tile: int = 496,
                    overlap: int = 32) -> np.ndarray:
    """Same front-end as the classical path, network in place of the NLM.

    The image is processed in overlapping tiles so that peak memory stays
    small on a modest CPU; the overlap is cropped away so tile boundaries
    leave no seam.
    """
    work = repair_impulses(noisy_u8.astype(np.float32))
    params = estimate_poisson_gaussian(np.clip(work, 0, 255))

    stabilised = np.stack([gat_forward(work[..., c], a, v)
                           for c, (a, v) in enumerate(params)], axis=-1)
    opponent = stabilised @ OPPONENT.T

    device = next(model.parameters()).device
    height, width = opponent.shape[:2]
    out = np.zeros_like(opponent)

    for top in range(0, height, tile):
        for left in range(0, width, tile):
            t0, l0 = max(top - overlap, 0), max(left - overlap, 0)
            t1 = min(top + tile + overlap, height)
            l1 = min(left + tile + overlap, width)
            patch = opponent[t0:t1, l0:l1]
            tensor = (torch.from_numpy(patch.transpose(2, 0, 1))
                      .unsqueeze(0).float().to(device))
            pred = model(tensor)[0].cpu().numpy().transpose(1, 2, 0)
            out[top:min(top + tile, height), left:min(left + tile, width)] = \
                pred[top - t0:min(top + tile, height) - t0,
                     left - l0:min(left + tile, width) - l0]

    rgb = out @ OPPONENT_INV.T
    planes = [gat_inverse_exact(rgb[..., c], a, v)
              for c, (a, v) in enumerate(params)]
    return np.clip(np.stack(planes, axis=-1), 0.0, 255.0)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def strip_noise_suffix(stem: str) -> str:
    return stem[:-len(NOISE_SUFFIX)] if stem.lower().endswith(NOISE_SUFFIX) else stem


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--noise_dir", "--input_dir", dest="noise_dir",
                    required=True, type=Path,
                    help="directory holding <id>_noise.png inputs")
    ap.add_argument("--denoised_dir", "--output_dir", dest="denoised_dir",
                    required=True, type=Path,
                    help="directory to write <id>.png outputs to")
    ap.add_argument("--strength", type=float, default=0.9,
                    help="classical denoiser strength, in units of the "
                         "stabilised noise sigma (tuned on the public set)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help="trained model; if absent the classical path is used")
    ap.add_argument("--classical", action="store_true",
                    help="force the classical path even if a model exists")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU inference even when a GPU is present")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.denoised_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.noise_dir.glob("*")
                   if p.suffix.lower() in VALID_EXT)
    if not paths:
        raise SystemExit(f"ERROR: no images found in {args.noise_dir}")

    model, torch = ((None, None) if args.classical
                    else load_model(args.checkpoint, force_cpu=args.cpu))
    if model is not None:
        device = str(next(model.parameters()).device).upper()
        print(f"Stage: learned (GAT + opponent domain)   device: {device}")
    else:
        print("Stage: classical (GAT + opponent domain + NLM)   device: CPU")

    total = 0.0
    for path in paths:
        noisy = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        start = time.time()
        if model is not None:
            result = denoise_learned(noisy, model, torch)
        else:
            result = denoise_classical(noisy, strength=args.strength)
        total += time.time() - start
        out = np.clip(np.rint(result), 0, 255).astype(np.uint8)
        Image.fromarray(out).save(args.denoised_dir / f"{strip_noise_suffix(path.stem)}.png")

    print(f"Denoised {len(paths)} images in {total:.1f}s "
          f"({total / len(paths):.2f}s per image)")
    print(f"Output written to {args.denoised_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
validate.py -- honest self-evaluation on the held-out split.

Scores the classical and the learned pipelines on images 421-460, which
``train.py`` never trains on, using the official composite metric from
``evaluation/evaluate.py``.  Also scores the provided baseline for
reference.

    python scripts/validate.py --data_root competition_data/public

Whatever wins here is what should be used to generate the submission.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
import denoise as D

VAL_IDS = [f"{i:03d}" for i in range(421, 461)]


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def ssim_value(a: np.ndarray, b: np.ndarray) -> float:
    """Exactly the settings used by evaluation/evaluate.py."""
    return float(sk_ssim(a, b, channel_axis=-1, data_range=1.0, win_size=7,
                         gaussian_weights=False, use_sample_covariance=True,
                         K1=0.01, K2=0.03))


def composite(gt, noisy, pred) -> tuple[float, float, float]:
    delta_psnr = (sk_psnr(gt, pred, data_range=1.0)
                  - sk_psnr(gt, noisy, data_range=1.0))
    delta_ssim = ssim_value(gt, pred) - ssim_value(gt, noisy)
    score = (0.6 * float(np.clip(delta_psnr / 15.0, 0.0, 1.0))
             + 0.4 * max(delta_ssim, 0.0))
    return score, delta_psnr, delta_ssim


def run(name: str, fn, data_root: Path, ids: list[str]) -> None:
    scores, dps, dss = [], [], []
    start = time.time()
    for image_id in ids:
        gt = load(data_root / "ground_truth" / f"{image_id}.png")
        noisy = load(data_root / "noisy" / f"{image_id}_noise.png")
        raw = fn((noisy * 255).astype(np.uint8))
        # round-trip through uint8 exactly as saving a PNG would
        pred = (np.clip(np.rint(raw), 0, 255).astype(np.uint8)
                .astype(np.float32) / 255.0)
        s, dp, ds = composite(gt, noisy, pred)
        scores.append(s)
        dps.append(dp)
        dss.append(ds)
    print(f"{name:34s} score={np.mean(scores):.4f}  "
          f"dPSNR={np.mean(dps):+6.2f} dB  dSSIM={np.mean(dss):+.4f}  "
          f"{(time.time() - start) / len(ids):5.2f}s/img", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("competition_data/public"))
    ap.add_argument("--checkpoint", type=Path, default=D.DEFAULT_CHECKPOINT)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N validation images")
    args = ap.parse_args()

    ids = [i for i in VAL_IDS
           if (args.data_root / "ground_truth" / f"{i}.png").exists()]
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise SystemExit("ERROR: no validation images found (expected 421-460)")

    print(f"Held-out validation on {len(ids)} images "
          f"({ids[0]}-{ids[-1]}), never used for training.\n")

    import cv2
    run("provided baseline",
        lambda u: cv2.fastNlMeansDenoisingColored(u, None, 10, 10, 7, 21)
        .astype(np.float32), args.data_root, ids)

    run("classical pipeline",
        lambda u: D.denoise_classical(u), args.data_root, ids)

    model, torch_mod = D.load_model(args.checkpoint)
    if model is None:
        print("\n(no usable checkpoint yet -- learned stage not scored)")
    else:
        run("hybrid pipeline (learned)",
            lambda u: D.denoise_learned(u, model, torch_mod),
            args.data_root, ids)
        print("\nUse whichever scored highest to generate the submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

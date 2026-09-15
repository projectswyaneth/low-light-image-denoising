# Scripts — Mora SP Cup 2026 submission

## Overview

Low-light denoising by a **hybrid classical / learned pipeline**. Every stage
was chosen from measurements made on the 460 public pairs, not by default.

```
noisy RGB (992x992, uint8)
   |
   |  1. impulse / hot-pixel repair        (classical, MAD-thresholded)
   |  2. blind Poisson-Gaussian estimation (classical, flat-block regression)
   |  3. Generalised Anscombe Transform    (classical, variance stabilisation)
   |  4. orthonormal opponent rotation     (classical, luma/chroma separation)
   |
   |  5. denoiser:  small residual U-Net   (learned)
   |                or non-local means     (classical fallback)
   |
   |  6. inverse opponent rotation         (classical)
   |  7. exact unbiased inverse GAT        (classical, Makitalo-Foi)
   v
denoised RGB -> <id>.png
```

### Why this structure

The measured corruption is

```
noisy = clip( clean + n, 0, 255 ),    Var[n | clean] = alpha * clean + sigma_r^2
```

with **alpha and sigma_r varying per image** (alpha ~ 3.6–13, sigma_r ~ 13–22 DN
across the public set). Because the noise is signal dependent, its strength
changes both within an image and between images. Steps 2–3 remove that
variation: after the GAT the noise is unit variance everywhere, so a single
denoiser setting is correct across the whole intensity range, and the learned
stage only has to model image structure rather than also modelling the noise
level. That is what makes a 0.5M-parameter network sufficient — and what keeps
CPU inference fast.

Step 4 is valid only *because* step 3 made the noise unit variance: an
orthonormal rotation then preserves that variance exactly while separating
luma from chroma, letting chroma be smoothed far harder than luma.

There is **no brightness or tone correction**. Regressing noisy on clean over
clipping-safe mid-tones gives a slope of 0.98–1.00 per channel, so the noisy
and clean images share the same exposure.

## Requirements

```
pip install -r scripts/requirements.txt
```

Inference needs only `numpy`, `opencv-python` and `Pillow`. PyTorch is needed
for the learned stage; **if PyTorch or the checkpoint is missing, `denoise.py`
falls back automatically to the classical pipeline and still produces valid
output.** No network access is required at inference time.

## Usage

```bash
python scripts/denoise.py --noise_dir  competition_data/submissions/noisy \
                          --denoised_dir competition_data/submissions/denoised
```

`--input_dir` / `--output_dir` are accepted as aliases for the same arguments.

The script reads every `<id>_noise.png` from the input directory and writes
`<id>.png` to the output directory, performing the required rename itself
(`461_noise.png` -> `461.png`). Provided images are never renamed or modified.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--classical` | force the classical path even when a checkpoint exists |
| `--checkpoint PATH` | use a checkpoint other than `scripts/model.pt` |
| `--strength S` | classical denoiser strength in stabilised sigma units |

## Output

20 files, `461.png`–`480.png`, 8-bit RGB PNG, 992×992 — identical in size and
format to the inputs.

## Reproducing the submission

```bash
# 1. dataset into the required layout (copies only, no renaming)
python scripts/setup_data.py --downloads <folder with the 3 downloaded folders>

# 2. train the learned stage (images 001-420; 421-460 held out)
python scripts/train.py --data_root competition_data/public

# 3. generate the submission images
python scripts/denoise.py --noise_dir competition_data/submissions/noisy \
                          --denoised_dir competition_data/submissions/denoised

# 4. self-evaluate on the held-out validation split
python evaluation/evaluate.py --noisy_dir competition_data/public/noisy \
                              --pred_dir  <validation predictions> \
                              --gt_dir    <validation ground truth>
```

All random seeds are fixed (`SEED = 2026`), so the pipeline is deterministic
and the submitted code regenerates the submitted images exactly.

## Validation strategy

Images **421–460 are held out** and never trained on; all reported scores come
from that split. Images 001–420 are used for training. The 20 submission
images (461–480) have no ground truth and are never used for any fitting or
tuning.

## Results (held-out validation split, official composite metric)

| Configuration | Composite | ΔPSNR | ΔSSIM |
| --- | --- | --- | --- |
| Provided baseline (defect repair + NLM h=10) | 0.2156 | +3.70 dB | +0.1687 |
| Classical pipeline (this repo, `--classical`) | 0.5053 | +8.42 dB | +0.4236 |
| Hybrid pipeline (this repo, default) | **0.5844** | **+9.84 dB** | **+0.4790** |

Mean CPU runtime: **1.43** s per image (20 images in 28.6 s, CPU-only, on an
Intel Core i5-11260H; the script prints the device it selected). The hybrid
pipeline scores 2.7x the provided baseline on the held-out split.

## Official Submission Information

Git Commit SHA:
`a246fb2abd1216b91eac325b902ce7fd6631b2b7`

This is the commit containing the complete solution and the one that produced
the submitted images. A file cannot record the identifier of the commit that
adds it, so the single commit following it -- the tip of `main` -- changes
nothing but the identifier lines in this file and in the root `README.md`.
Both commits are tagged; `git diff a246fb2 HEAD` touches no code.

Model Checkpoint:
`model.pt`

Model Drive Link:
`N/A — checkpoint is small enough to be committed directly`

Expected Model Path:
`scripts/model.pt`

Model SHA-256:
`f5ed44cb5ccc30aa2db7a54d3a72caffe8faf23db40cab8769efb289565ebae3`

Verify with `sha256sum scripts/model.pt`, or on Windows PowerShell
`(Get-FileHash scripts\model.pt -Algorithm SHA256).Hash`.

> **Submission freeze.** After the preliminary-round deadline no new commits
> may be made to this repository, and the checkpoint must not be modified or
> replaced. The commit SHA above is the official submitted code version.

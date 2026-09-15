"""
noise_model.py
==============
Classical signal-processing front-end for the Mora SP Cup 2026
low-light denoising challenge.

Everything in this module is derived from measurements made on the 460
public image pairs.  The measured corruption model is

    noisy = clip( clean + n ,  0 , 255 )
    Var[n | clean] = alpha * clean + sigma_r**2          (Poisson-Gaussian)

with alpha (photon/shot gain) and sigma_r (read noise) BOTH VARYING PER
IMAGE -- this is the "different severity levels of noise" referred to in
the participant handbook.  Superimposed on that are

    * per-image random row / column stripe noise (fixed-pattern-like), and
    * sparse impulse ("hot pixel") outliers.

No brightness or tone correction is performed: regressing noisy on clean
over clipping-safe mid-tones gives a slope of 0.98-1.00 per channel, so
the noisy and clean images share the same exposure.

All parameters are estimated blind, from the noisy image alone.  Nothing
in this file uses ground truth, image identity, or file names.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "OPPONENT",
    "OPPONENT_INV",
    "repair_impulses",
    "destripe",
    "estimate_poisson_gaussian",
    "gat_forward",
    "gat_inverse_exact",
]


# ---------------------------------------------------------------------------
# 1. Impulse / hot-pixel repair
# ---------------------------------------------------------------------------

def repair_impulses(img: np.ndarray, n_sigma: float = 4.0) -> np.ndarray:
    """Replace sparse outliers by the local median.

    A pixel is an outlier when it deviates from its 3x3 median by more than
    ``n_sigma`` times a robust local scale estimate.  The scale is derived
    from the median absolute deviation of the median-residual, so the
    threshold adapts to the noise level of each image instead of being a
    fixed constant as in the provided baseline.
    """
    out = img.copy()
    for c in range(img.shape[2]):
        ch = img[..., c]
        med = cv2.medianBlur(ch.astype(np.float32), 3)
        dev = ch - med
        # robust sigma from the MAD (0.6745 = Phi^-1(0.75))
        mad = np.median(np.abs(dev - np.median(dev)))
        sigma = max(mad / 0.6745, 1e-3)
        out[..., c] = np.where(np.abs(dev) > n_sigma * sigma, med, ch)
    return out


# ---------------------------------------------------------------------------
# 2. Row / column destriping
# ---------------------------------------------------------------------------

def destripe(img: np.ndarray, blur_sigma: float = 6.0) -> np.ndarray:
    """Remove per-row and per-column stripe noise.

    Measured on the public set, the per-row and per-column means of the
    noise residual have a standard deviation of 3-4.6 DN where independent
    pixel noise would predict only 0.45-0.64 DN, i.e. a 6-8x excess.  The
    stripe pattern is uncorrelated between images, so it must be estimated
    from each image separately.

    The estimate is taken on a high-pass version of the channel so that
    genuine image content (which is low-frequency along a row) is not
    mistaken for a stripe, and the median is used so that a few strong
    edges crossing the row cannot drag the estimate.
    """
    out = img.copy()
    for c in range(img.shape[2]):
        ch = out[..., c].astype(np.float32)
        for axis in (1, 0):                       # rows first, then columns
            high_pass = ch - cv2.GaussianBlur(ch, (0, 0), blur_sigma)
            offset = np.median(high_pass, axis=axis, keepdims=True)
            ch = ch - offset
        out[..., c] = ch
    return out


# ---------------------------------------------------------------------------
# 3. Blind Poisson-Gaussian parameter estimation
# ---------------------------------------------------------------------------

def estimate_poisson_gaussian(img: np.ndarray, block: int = 8,
                              percentile: float = 10.0
                              ) -> list[tuple[float, float]]:
    """Estimate (alpha, sigma_r^2) per channel from the noisy image alone.

    The image is tiled into ``block`` x ``block`` patches and the (mean,
    variance) pair of every patch is computed.  Within each mean-intensity
    bin the LOW percentile of the patch variances is taken: those patches
    are the flat ones, whose variance is due to noise rather than to image
    structure.  A straight line fitted through (mean, variance) then gives

        variance = alpha * mean + sigma_r^2

    which is exactly the Poisson-Gaussian relation.  Returns one
    (alpha, sigma_r^2) pair per channel.
    """
    params: list[tuple[float, float]] = []
    for c in range(img.shape[2]):
        x = img[..., c]
        h, w = x.shape[0] // block, x.shape[1] // block
        tiles = (x[:h * block, :w * block]
                 .reshape(h, block, w, block)
                 .transpose(0, 2, 1, 3)
                 .reshape(-1, block * block))
        mean = tiles.mean(axis=1)
        var = tiles.var(axis=1)

        xs, ys = [], []
        for lo in range(0, 256, 8):
            sel = (mean >= lo) & (mean < lo + 8)
            if sel.sum() >= 20:
                xs.append(mean[sel].mean())
                ys.append(np.percentile(var[sel], percentile))

        if len(xs) < 4:                       # degenerate image: safe default
            params.append((5.0, 225.0))
            continue

        design = np.vstack([np.asarray(xs), np.ones(len(xs))]).T
        alpha, read_var = np.linalg.lstsq(design, np.asarray(ys), rcond=None)[0]
        params.append((float(np.clip(alpha, 0.5, 30.0)),
                       float(np.clip(read_var, 1.0, 2000.0))))
    return params


# ---------------------------------------------------------------------------
# 4. Generalised Anscombe Transform and its exact unbiased inverse
# ---------------------------------------------------------------------------

_SQRT_1_5 = np.sqrt(1.5)


def gat_forward(x: np.ndarray, alpha: float, read_var: float) -> np.ndarray:
    """Generalised Anscombe Transform.

    Maps signal-dependent Poisson-Gaussian noise to approximately additive
    white Gaussian noise of UNIT variance, so that a single denoiser
    setting is correct across the whole intensity range of the image.
    """
    inner = alpha * x + 0.375 * alpha * alpha + read_var
    return (2.0 / alpha) * np.sqrt(np.maximum(inner, 0.0))


def gat_inverse_exact(y: np.ndarray, alpha: float, read_var: float) -> np.ndarray:
    """Closed-form exact unbiased inverse of the GAT (Makitalo & Foi).

    The naive algebraic inverse is biased, because the expectation of the
    transform is not the transform of the expectation.  This closed form
    corrects that bias and is the standard companion to the forward GAT.
    """
    y = np.maximum(y, 1e-6)
    poisson_estimate = (y * y / 4.0
                        + 0.25 * _SQRT_1_5 / y
                        - 1.375 / (y * y)
                        + 0.625 * _SQRT_1_5 / (y ** 3)
                        - 0.125)
    return alpha * poisson_estimate - read_var / alpha


# ---------------------------------------------------------------------------
# 5. Opponent-colour rotation
# ---------------------------------------------------------------------------
#
# After the GAT each RGB plane carries unit-variance noise, and the measured
# residual is nearly uncorrelated between channels (cross-channel correlation
# 0.02-0.13 on the public set).  An ORTHONORMAL rotation therefore preserves
# that unit variance exactly, while concentrating image structure into the
# luma axis.  The two chroma planes are much smoother than luma, so they
# tolerate -- and benefit from -- considerably stronger smoothing.
#
# This is the classical luma/chroma denoising argument; what makes it valid
# here is the variance stabilisation that precedes it.

OPPONENT = np.array([[1.0,  1.0,  1.0],      # luma
                     [1.0,  0.0, -1.0],      # red - blue
                     [1.0, -2.0,  1.0]],     # red - 2*green + blue
                    dtype=np.float32)
OPPONENT /= np.linalg.norm(OPPONENT, axis=1, keepdims=True)
OPPONENT_INV = OPPONENT.T                    # orthonormal => inverse = transpose

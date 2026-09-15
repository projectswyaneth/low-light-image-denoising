#!/usr/bin/env python3
"""
setup_data.py -- copy the downloaded dataset into the required layout.

The competition requires this exact structure:

    competition_data/public/ground_truth/001.png ... 460.png
    competition_data/public/noisy/001_noise.png ... 460_noise.png
    competition_data/submissions/noisy/461_noise.png ... 480_noise.png
    competition_data/submissions/denoised/

Run it once, pointing at wherever the three downloaded folders live:

    python scripts/setup_data.py --downloads "C:/Users/User/Downloads"

Files are COPIED, never moved or renamed -- the handbook forbids renaming
the provided noisy images.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TARGETS = {
    "ground_truth": Path("competition_data/public/ground_truth"),
    "noisy":        Path("competition_data/public/noisy"),
    "submissions":  Path("competition_data/submissions/noisy"),
}
EXPECTED = {"ground_truth": 460, "noisy": 460, "submissions": 20}


def locate(downloads: Path, marker: str) -> Path | None:
    """Find the folder actually containing the PNGs, whatever it is nested in."""
    for candidate in sorted(downloads.glob(f"*{marker}*")):
        if not candidate.is_dir():
            continue
        for sub in [candidate, *candidate.rglob("*")]:
            if sub.is_dir() and any(sub.glob("*.png")):
                return sub
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, required=True,
                    help="folder holding the three downloaded dataset folders")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="repository root (default: current directory)")
    args = ap.parse_args()

    ok = True
    for marker, target in TARGETS.items():
        source = locate(args.downloads, marker)
        destination = args.root / target
        destination.mkdir(parents=True, exist_ok=True)
        if source is None:
            print(f"  [MISSING] no folder matching '*{marker}*' with PNGs in "
                  f"{args.downloads}")
            ok = False
            continue
        pngs = sorted(source.glob("*.png"))
        for p in pngs:
            shutil.copy2(p, destination / p.name)
        status = "OK" if len(pngs) == EXPECTED[marker] else "CHECK"
        print(f"  [{status}] {marker}: copied {len(pngs)} files "
              f"(expected {EXPECTED[marker]}) -> {target}")
        if len(pngs) != EXPECTED[marker]:
            ok = False

    (args.root / "competition_data/submissions/denoised").mkdir(parents=True,
                                                                exist_ok=True)
    (args.root / "competition_data/public/denoised").mkdir(parents=True,
                                                           exist_ok=True)
    print("\nDataset layout " + ("is correct." if ok else "HAS PROBLEMS - see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

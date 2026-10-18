#!/usr/bin/env python3
"""
run_validation.py — reproduce the ground-truth descriptor checks.

Builds ten images whose descriptor values are known analytically or by
construction, measures them with frame_descriptors.py, and reports the pass
table quoted in references/descriptors.md and in the accompanying paper.

Run from the release root:

    python examples/run_validation.py            # pass table
    python examples/run_validation.py --csv out.csv

Expected result: 30 of 31 checks pass. The single failure is X01
(figure-to-ground area), which fails on structural grounds — spectral-residual
saliency responds to boundaries, not filled regions — and is excluded from the
confirmatory descriptor set for exactly that reason. A release in which this
check "passes" has been tampered with or has a different saliency model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import frame_descriptors as fd  # noqa: E402
from skimage.color import lab2rgb  # noqa: E402

H, W = 360, 640


def lab_colour(lightness, a, b):
    return np.clip(lab2rgb(np.array([[[lightness, a, b]]], float))[0, 0], 0, 1)


def build_images():
    """Ten constructed images with known answers."""
    rng = np.random.default_rng(1)
    warm, cold = lab_colour(60, 40, 45), lab_colour(60, -10, -45)
    yy, xx = np.mgrid[0:H, 0:W]
    img = {}

    img["flat_grey"] = np.full((H, W, 3), 0.5)

    a = np.zeros((H, W, 3)); a[:, :W // 2] = warm; a[:, W // 2:] = cold
    img["split_5050"] = a

    chk = ((yy // 8 + xx // 8) % 2).astype(bool)
    a = np.zeros((H, W, 3)); a[chk] = warm; a[~chk] = cold
    img["checker_5050"] = a

    a = np.zeros((H, W, 3)); k = int(0.85 * W); a[:, :k] = warm; a[:, k:] = cold
    img["split_8515"] = a

    ramp = np.linspace(10, 90, H)[:, None] * np.ones((1, W))
    zeros = np.zeros_like(ramp)
    img["lum_ramp_10_90"] = np.clip(lab2rgb(np.stack([ramp, zeros, zeros], -1)), 0, 1)

    a = np.zeros((H, W, 3)); r = int(0.40 * H)
    a[:r] = lab_colour(80, -5, -25); a[r:] = lab_colour(35, 5, 25)
    img["horizon_40"] = a

    a = np.full((H, W, 3), 0.25); a[:, ::12] = 0.85
    img["stripes_vertical"] = a

    img["noise_isotropic"] = rng.random((H, W, 3))

    r0 = 0.125 * H
    disc = (yy - H / 2) ** 2 + (xx - W / 2) ** 2 < r0 ** 2
    a = np.zeros((H, W, 3)); a[disc] = 1.0
    img["disc_centre"] = a
    img["_disc_area"] = float(disc.mean())

    d2 = (yy - H / 3) ** 2 + (xx - W / 3) ** 2 < r0 ** 2
    a = np.zeros((H, W, 3)); a[d2] = 1.0
    img["disc_thirds"] = a
    return img


def checks(d, disc_area):
    """(image, descriptor, expectation, predicate) -> evaluated against measurements d."""
    g = lambda im, k: d[im][k]  # noqa: E731
    near = lambda x, t, tol: np.isfinite(x) and abs(x - t) < tol  # noqa: E731
    return [
        ("flat grey field", "C01 palette_entropy", "== 0", near(g("flat_grey", "palette_entropy"), 0, 1e-3)),
        ("flat grey field", "C06 luminance_range", "== 0", near(g("flat_grey", "luminance_range"), 0, 1e-2)),
        ("flat grey field", "C07 rms_contrast", "== 0", near(g("flat_grey", "rms_contrast"), 0, 1e-3)),
        ("flat grey field", "C10 edge_density", "== 0", near(g("flat_grey", "edge_density"), 0, 1e-3)),
        ("flat grey field", "C09 colourfulness", "== 0", near(g("flat_grey", "colourfulness"), 0, 0.5)),
        ("flat grey field", "C03 warm_ratio", "NaN (no chroma)", not np.isfinite(g("flat_grey", "warm_ratio"))),
        ("flat grey field", "C14 saliency_offset", "NaN (degenerate)", not np.isfinite(g("flat_grey", "saliency_offset"))),
        ("flat grey field", "C12 glcm_energy", "== 1", near(g("flat_grey", "glcm_energy"), 1, 1e-3)),
        ("warm/cold split 50:50", "C03 warm_ratio", "== 0.50", near(g("split_5050", "warm_ratio"), 0.5, 0.01)),
        ("warm/cold split 85:15", "C03 warm_ratio", "== 0.85", near(g("split_8515", "warm_ratio"), 0.85, 0.01)),
        ("warm/cold split 50:50", "C04 warm_cold_moran", "> 0.9 (segregated)", g("split_5050", "warm_cold_moran") > 0.9),
        ("warm/cold checkerboard", "C04 warm_cold_moran", "< 0 (interleaved)", g("checker_5050", "warm_cold_moran") < 0),
        ("isotropic noise", "C04 warm_cold_moran", "~ 0 (no structure)", abs(g("noise_isotropic", "warm_cold_moran")) < 0.05),
        ("warm/cold split 50:50", "C02 palette_bimodality", "> 0.9", g("split_5050", "palette_bimodality") > 0.9),
        ("warm/cold split 85:15", "C02 palette_bimodality", "~ 0.30 (= balance)", near(g("split_8515", "palette_bimodality"), 0.30, 0.05)),
        ("warm/cold checkerboard", "C02 palette_bimodality", "= split value (population, not layout)",
         abs(g("checker_5050", "palette_bimodality") - g("split_5050", "palette_bimodality")) < 0.02),
        ("luminance ramp L*10-90", "C06 luminance_range", "== 72.0 (analytic)", near(g("lum_ramp_10_90", "luminance_range"), 72.0, 0.5)),
        ("luminance ramp L*10-90", "C01 palette_entropy", "== 1.0 (8 equal masses)", near(g("lum_ramp_10_90", "palette_entropy"), 1.0, 0.02)),
        ("synthetic horizon at 0.40", "C16 horizon_height", "== 0.40", near(g("horizon_40", "horizon_height"), 0.40, 0.02)),
        ("synthetic horizon at 0.40", "horizon_confidence", "> 1.5", g("horizon_40", "horizon_confidence") > 1.5),
        ("luminance ramp L*10-90", "horizon_confidence", "< 1.5 (no horizon to find)", g("lum_ramp_10_90", "horizon_confidence") < 1.5),
        ("vertical stripes", "C11 orientation_entropy", "< 0.10 (one angle)", g("stripes_vertical", "orientation_entropy") < 0.10),
        ("isotropic noise", "C11 orientation_entropy", "> 0.90 (all angles)", g("noise_isotropic", "orientation_entropy") > 0.90),
        ("isotropic noise", "C12 glcm_energy", "< 0.25", g("noise_isotropic", "glcm_energy") < 0.25),
        ("isotropic noise", "C13 glcm_contrast", "> stripes value", g("noise_isotropic", "glcm_contrast") > g("stripes_vertical", "glcm_contrast")),
        ("disc at frame centre", "C14 saliency_offset", "< 0.02", g("disc_centre", "saliency_offset") < 0.02),
        ("disc at thirds point", "C14 saliency_offset", "> 0.10", g("disc_thirds", "saliency_offset") > 0.10),
        ("disc at thirds point", "C17 thirds_alignment", "> centre-disc value", g("disc_thirds", "thirds_alignment") > g("disc_centre", "thirds_alignment")),
        ("disc at frame centre", "C15 saliency_dispersion", "< noise value (focal)", g("disc_centre", "saliency_dispersion") < g("noise_isotropic", "saliency_dispersion")),
        ("disc at frame centre", "X01 figure_ground_proxy", f"== {disc_area:.4f} (true disc area)", near(g("disc_centre", "figure_ground_proxy"), disc_area, 0.005)),
        ("warm/cold split 50:50", "C10 edge_density", "== 0 (equiluminant: no L* edge)", near(g("split_5050", "edge_density"), 0, 1e-3)),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None, help="write the pass table to CSV")
    args = ap.parse_args()

    images = build_images()
    disc_area = images.pop("_disc_area")
    measured = {name: fd.describe_frame(arr) for name, arr in images.items()}

    rows = checks(measured, disc_area)
    width = max(len(r[0]) for r in rows)
    print(f"{'test image'.ljust(width)}  {'descriptor':<26} {'expectation':<42} result")
    print("-" * (width + 78))
    for image, descriptor, expectation, ok in rows:
        print(f"{image.ljust(width)}  {descriptor:<26} {expectation:<42} {'pass' if ok else 'FAIL'}")
    n_pass = sum(1 for r in rows if r[3])
    print(f"\n{n_pass} of {len(rows)} checks pass.")
    if n_pass == len(rows) - 1 and not rows[-2][3]:
        print("Expected outcome: the one failure is X01 (figure-to-ground area), which cannot work in\n"
              "principle and is excluded from the confirmatory set. See references/descriptors.md section\n"
              "'A descriptor that cannot work'.")
    elif n_pass != len(rows) - 1:
        print("UNEXPECTED: this release is validated at 30/31. Investigate before using the descriptors.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["test_image", "descriptor", "expectation", "result"])
            for image, descriptor, expectation, ok in rows:
                w.writerow([image, descriptor, expectation, "pass" if ok else "FAIL"])
        print(f"wrote {args.csv}")
    return 0 if n_pass >= len(rows) - 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())

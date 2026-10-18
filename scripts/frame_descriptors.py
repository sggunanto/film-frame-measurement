#!/usr/bin/env python3
"""
frame_descriptors.py — formal descriptors (C01-C18) for still frames.

Computes the measured half of the instrument described in `codebook_v1.md` §6:
the quantities a human coder cannot hold a stable threshold for across 108
frames and six colour regimes.

Every descriptor is defined in the docstring of the function that computes it,
with the convention, the unit, and the behaviour on degenerate input stated
explicitly. Nothing here is a black box: where a descriptor is a proxy for the
construct it serves (C18), the docstring says so.

Dependencies: numpy, scipy, scikit-image, scikit-learn, pandas.

Usage
-----
    python frame_descriptors.py --frames ./frames --out frame_descriptors.csv \
        [--editions editions.csv] [--work-height 360]

    from frame_descriptors import describe_frame, load_frame
    d = describe_frame(load_frame("frames/MON_MON03_0091233_peak_MON03p.png"))
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.feature import canny, graycomatrix, graycoprops, structure_tensor
from skimage.transform import resize
from sklearn.cluster import KMeans

# ------------------------------------------------------------------ constants
CHROMA_MIN = 8.0        # Lab chroma below which a pixel is treated as achromatic
WARM_AXIS_DEG = 60.0    # hue angle of the warm pole (orange-red); cold pole is +180
N_PALETTE = 8           # k for the palette descriptors
WORK_HEIGHT = 360       # working height for scale-dependent descriptors
GLCM_LEVELS = 8
RANDOM_STATE = 0


# -------------------------------------------------------------------- loading
def load_frame(path, crop_top: int = 0, crop_bottom: int = 0,
               auto_letterbox: bool = True) -> np.ndarray:
    """Load a PNG as float RGB in [0, 1], with letterbox bars removed.

    crop_top/crop_bottom come from editions.csv and take precedence. If both are
    zero and auto_letterbox is True, near-black rows at the top and bottom are
    detected and removed (threshold: mean channel value below 0.02 across the row).
    """
    from imageio.v3 import imread
    arr = np.asarray(imread(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3].astype(np.float64)
    arr /= 65535.0 if arr.max() > 255 else 255.0
    if crop_top or crop_bottom:
        h = arr.shape[0]
        arr = arr[crop_top:h - crop_bottom if crop_bottom else h]
    elif auto_letterbox:
        rowmean = arr.mean(axis=(1, 2))
        dark = rowmean < 0.02
        top = 0
        while top < len(dark) and dark[top]:
            top += 1
        bot = len(dark)
        while bot > top and dark[bot - 1]:
            bot -= 1
        if bot - top >= 0.5 * len(dark):      # never crop away more than half
            arr = arr[top:bot]
    return arr


def _work(rgb: np.ndarray, work_height: int) -> np.ndarray:
    """Resize to a common height for scale-dependent descriptors (C10-C15, C17)."""
    h, w = rgb.shape[:2]
    if h == work_height:
        return rgb
    nw = max(1, int(round(w * work_height / h)))
    return resize(rgb, (work_height, nw), order=1, anti_aliasing=True,
                  preserve_range=True)


# -------------------------------------------------------------- colour family
def palette_descriptors(lab: np.ndarray, rng_pixels: int = 20000) -> dict:
    """C01 palette entropy, C02 palette bimodality, from CIELAB k-means.

    C01 `palette_entropy`: Shannon entropy of the 8-cluster mass proportions,
        divided by log(8). 0 = one colour holds the frame; 1 = eight equal masses.

    C02 `palette_bimodality`: how strongly the frame splits into two colour
        populations. With a 2-means partition (centroids c1, c2; masses p1, p2;
        mean within-cluster distance s, and dE = ||c1 - c2||_CIE76):

            separation = dE / (dE + 2 * s)        in [0, 1)
            balance    = 2 * min(p1, p2)          in [0, 1]
            C02        = separation * balance     in [0, 1]

        Bounded on purpose, and each limit is a real case:
          * flat field            -> dE = 0            -> C02 = 0
          * two flat colours 50/50 -> s = 0, dE large  -> C02 -> 1
          * the same two at 85/15 -> balance 0.3       -> C02 ~ 0.3
          * fine-grained noise    -> s >> dE           -> C02 ~ 0

        An earlier formulation divided by s directly; with two flat colours s = 0,
        so perfect bimodality returned 0 instead of 1. The regularised ratio above
        removes that discontinuity.

        Note that C02 is a property of the colour *population*, not of its spatial
        arrangement: a left-right split and a fine checkerboard of the same two
        colours have the same C02. Telling those apart is C04's job.
    """
    flat = lab.reshape(-1, 3)
    if flat.shape[0] > rng_pixels:
        rs = np.random.default_rng(RANDOM_STATE)
        flat = flat[rs.choice(flat.shape[0], rng_pixels, replace=False)]

    out = {}
    if np.allclose(flat.std(axis=0), 0):
        return {"palette_entropy": 0.0, "palette_bimodality": 0.0,
                "palette_degenerate": True}

    n_distinct = len(np.unique(np.round(flat, 3), axis=0))
    k_eff = int(min(N_PALETTE, max(1, n_distinct)))
    out["palette_k_effective"] = k_eff
    if k_eff == 1:
        out.update(palette_entropy=0.0, palette_bimodality=0.0, palette_degenerate=True)
        return out

    km = KMeans(n_clusters=k_eff, n_init=10, random_state=RANDOM_STATE).fit(flat)
    _, counts = np.unique(km.labels_, return_counts=True)
    p = counts / counts.sum()
    # Normalised by log(N_PALETTE), not log(k_eff), so that a frame with few
    # distinct colours scores low rather than being rescaled up to 1.
    out["palette_entropy"] = float(-(p * np.log(p)).sum() / np.log(N_PALETTE))

    km2 = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(flat)
    c1, c2 = km2.cluster_centers_
    m = km2.labels_ == 0
    p1 = m.mean()
    within = np.concatenate([
        np.linalg.norm(flat[m] - c1, axis=1) if m.any() else np.array([0.0]),
        np.linalg.norm(flat[~m] - c2, axis=1) if (~m).any() else np.array([0.0]),
    ]).mean()
    dE = float(np.linalg.norm(c1 - c2))
    sep = dE / (dE + 2.0 * within) if (dE + 2.0 * within) > 1e-12 else 0.0
    out["palette_bimodality"] = float(sep * 2 * min(p1, 1 - p1))
    out["palette_degenerate"] = False
    return out


def warm_cold_descriptors(lab: np.ndarray, grid: int = 64) -> dict:
    """C03 warm-cold mass ratio, C04 warm-cold spatial segregation.

    Hue angle h = atan2(b*, a*). A pixel is chromatic when chroma = hypot(a*, b*)
    exceeds CHROMA_MIN (8 Lab units); achromatic pixels are excluded from C03 and
    contribute 0 to the warmth field.

    C03 `warm_ratio`: the fraction of *chromatic* pixels whose hue lies within
        90 degrees of the warm axis (60 deg, orange-red). 0.5 = balanced warm and
        cold mass; 1 = entirely warm. NaN when the frame has no chromatic pixels
        (reported alongside `achromatic_fraction`).

    C04 `warm_cold_moran`: Moran's I of the signed warmth field
        w = chroma * cos(h - 60 deg), on a downsampled grid with rook adjacency.

        This is the descriptor that separates *where* warm and cold sit from *how
        much* of each there is. A frame split left-warm / right-cold and a frame
        of fine warm-cold checkerboard have the same C03 (0.5) and opposite C04:
        the split approaches +1 (like next to like), the checkerboard goes
        negative (like next to unlike). Contradiction that occupies distinct
        regions of the image is the high-C04 case.

        NaN when the warmth field has no variance (a flat field).
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a)
    chromatic = chroma > CHROMA_MIN
    warm_axis = np.deg2rad(WARM_AXIS_DEG)
    is_warm = np.cos(hue - warm_axis) > 0

    out = {"achromatic_fraction": float(1.0 - chromatic.mean())}
    out["warm_ratio"] = (float(is_warm[chromatic].mean()) if chromatic.any() else np.nan)

    warmth = chroma * np.cos(hue - warm_axis)
    warmth = np.where(chromatic, warmth, 0.0)
    h, w = warmth.shape
    # Block-mean downsampling, NOT resize(). An anti-aliased resize applies a
    # smoothing kernel wider than one output cell, which injects spatial
    # autocorrelation and inflates Moran's I: white noise measured 0.30 instead
    # of ~0 before this was changed. Disjoint block means do not share pixels.
    from skimage.measure import block_reduce
    blk = max(1, int(round(max(h, w) / grid)))
    z = block_reduce(warmth, (blk, blk), func=np.mean)
    out["warm_cold_grid"] = f"{z.shape[0]}x{z.shape[1]}"
    out["warm_cold_moran"] = float(_morans_i(z))
    return out


def _morans_i(z: np.ndarray) -> float:
    """Moran's I with rook adjacency on a regular grid. NaN if z has no variance."""
    x = z - z.mean()
    denom = (x ** 2).sum()
    if denom < 1e-12:
        return np.nan
    num = 0.0
    n_w = 0
    for shift, axis in ((1, 0), (1, 1)):
        a = np.take(x, range(0, x.shape[axis] - shift), axis=axis)
        b = np.take(x, range(shift, x.shape[axis]), axis=axis)
        num += 2.0 * (a * b).sum()        # each pair counted in both directions
        n_w += 2 * a.size
    return float((x.size / n_w) * (num / denom))


def luminance_descriptors(lab: np.ndarray) -> dict:
    """C05-C07 from L* (0-100).

    C05 `luminance_mean`  : mean L*.
    C06 `luminance_range` : p95(L*) - p05(L*). Percentiles, not min-max, so one
        specular highlight or one crushed shadow cannot set the range.
    C07 `rms_contrast`    : standard deviation of L*/100 (dimensionless).
    """
    L = lab[..., 0]
    return {"luminance_mean": float(L.mean()),
            "luminance_range": float(np.percentile(L, 95) - np.percentile(L, 5)),
            "rms_contrast": float((L / 100.0).std())}


def hue_descriptors(lab: np.ndarray) -> dict:
    """C08 hue dispersion, C09 colourfulness.

    C08 `hue_dispersion`: circular variance 1 - R of the hue angles, weighted by
        chroma, over chromatic pixels. 0 = one hue; 1 = hues spread evenly round
        the circle. NaN when no pixel is chromatic.

    C09 `colourfulness`: the Hasler and Susstrunk (2003) metric, computed on sRGB
        in 0-255 units:
            rg = R - G,  yb = 0.5*(R + G) - B
            M  = sqrt(var_rg + var_yb) + 0.3 * sqrt(mean_rg^2 + mean_yb^2)
        Reported in the original scale, where roughly <15 reads as not colourful
        and >60 as extremely colourful. Implemented here, not approximated.
    """
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    m = chroma > CHROMA_MIN
    if m.any():
        hue = np.arctan2(b[m], a[m])
        wgt = chroma[m]
        R = np.hypot((wgt * np.cos(hue)).sum(), (wgt * np.sin(hue)).sum()) / wgt.sum()
        hue_disp = float(1.0 - R)
    else:
        hue_disp = np.nan
    return {"hue_dispersion": hue_disp}


def colourfulness(rgb: np.ndarray) -> dict:
    """C09, Hasler-Susstrunk. Kept separate because it is defined on sRGB, not Lab."""
    r, g, b = (rgb[..., i] * 255.0 for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    M = np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return {"colourfulness": float(M)}


# ------------------------------------------------------- line, texture family
def edge_descriptors(gray: np.ndarray, sigma: float = 2.0) -> dict:
    """C10 edge density, C11 orientation entropy. Computed at the working height.

    C10 `edge_density`: fraction of pixels flagged by the Canny detector
        (sigma = 2, automatic thresholds). Scale-dependent, which is why the frame
        is resized to a common height first.

    C11 `orientation_entropy`: Shannon entropy of a 36-bin histogram of structure-
        tensor orientations over the 180-degree range, each pixel weighted by
        tensor coherence, divided by log(36). 0 = all structure at one angle
        (stripes, an architectural grid); 1 = isotropic (rubble, foliage, noise).
    """
    e = canny(gray, sigma=sigma)
    out = {"edge_density": float(e.mean())}
    Axx, Axy, Ayy = structure_tensor(gray, sigma=sigma, order="rc")
    theta = 0.5 * np.arctan2(2 * Axy, (Axx - Ayy))          # in [-pi/2, pi/2]
    tr = Axx + Ayy
    det = Axx * Ayy - Axy ** 2
    disc = np.clip(tr ** 2 - 4 * det, 0, None)
    coherence = np.where(tr > 1e-12, np.sqrt(disc) / (tr + 1e-12), 0.0)
    bins = 36
    idx = np.clip(((theta + np.pi / 2) / np.pi * bins).astype(int), 0, bins - 1)
    hist = np.bincount(idx.ravel(), weights=coherence.ravel(), minlength=bins)
    if hist.sum() <= 1e-12:
        out["orientation_entropy"] = np.nan
    else:
        p = hist / hist.sum()
        nz = p[p > 0]
        out["orientation_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(bins))
    return out


def texture_descriptors(gray: np.ndarray) -> dict:
    """C12 GLCM energy, C13 GLCM contrast.

    Grey levels quantised to 8, co-occurrence over distances 1, 2, 4 at 0, 45, 90
    and 135 degrees, symmetric and normalised, then averaged over all
    distance-angle pairs.

    C12 `glcm_energy`  : sqrt of angular second moment. High = uniform, repetitive
        or flat; low = varied.
    C13 `glcm_contrast`: intensity difference between neighbouring pixels. High =
        abrupt local change (rubble, debris, high-frequency detail).
    """
    q = np.clip((gray * GLCM_LEVELS).astype(np.uint8), 0, GLCM_LEVELS - 1)
    P = graycomatrix(q, distances=[1, 2, 4],
                     angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    return {"glcm_energy": float(graycoprops(P, "energy").mean()),
            "glcm_contrast": float(graycoprops(P, "contrast").mean())}


# ----------------------------------------------------- composition family
def saliency_map(gray: np.ndarray) -> np.ndarray:
    """Spectral-residual saliency (Hou and Zhang 2007), blurred and normalised to [0,1].

    A deliberately simple, deterministic model with no training data and no
    weights to version. It predicts human fixations far less well than current
    deep models (see the MIT saliency benchmark), and is used here only as a
    *compositional attention proxy*: where image energy departs from the natural
    amplitude spectrum. Descriptors built on it (C14, C15, C17, C18) inherit that
    limitation and are reported as proxies.
    """
    if gray.std() < 1e-6:
        # A constant image has no spectral residual. Normalising the numerical
        # dust that remains produced a spurious saliency field (offset 0.17,
        # dispersion 0.62 on a flat grey frame), so return an empty map and let
        # the descriptors report NaN.
        return np.zeros_like(gray)
    F = np.fft.fft2(gray)
    logA = np.log(np.abs(F) + 1e-12)
    phase = np.angle(F)
    smooth = ndi.uniform_filter(logA, size=3)
    S = np.abs(np.fft.ifft2(np.exp((logA - smooth) + 1j * phase))) ** 2
    S = ndi.gaussian_filter(S, sigma=3)
    rng = S.max() - S.min()
    return (S - S.min()) / rng if rng > 1e-12 else np.zeros_like(S)


def composition_descriptors(gray: np.ndarray) -> dict:
    """C14-C18: where attention sits, how spread it is, the horizon, thirds, figure area.

    C14 `saliency_offset`   : distance from the saliency centre of mass to the frame
        centre, divided by half the frame diagonal. 0 = centred composition.
    C15 `saliency_dispersion`: root mean square spread of saliency mass about its
        own centroid, divided by half the diagonal. Low = one focal subject;
        high = attention distributed across the frame, which is the expected
        signature of a space-carried rather than figure-carried image.
    C16 `horizon_height`    : row of maximum horizontal-edge energy, as a fraction of
        frame height (0 = top). Searched between 0.15 and 0.85 only.
        `horizon_confidence` = peak energy / mean energy; below about 1.5 the
        estimate is not meaningful and C16 should be treated as missing.
    C17 `thirds_alignment`  : 1 - (distance from the saliency centroid to the nearest
        rule-of-thirds intersection) / half-diagonal. 1 = exactly on an
        intersection.
    X01 `figure_ground_proxy`: fraction of frame area above Otsu's threshold on the
        saliency map. **EXPLORATORY ONLY - this descriptor failed validation.**
        It was intended as the machine counterpart of the manual code M05, and
        against discs of known area it recovered only rank order (Spearman 0.89,
        Pearson 0.17) with one outright inversion. Saliency here responds to
        boundaries, not to filled regions, so it cannot measure how much of the
        frame a figure occupies. Kept in the output so the failure is auditable;
        excluded from DESCRIPTOR_ORDER and from every confirmatory test. Do not
        report it as "the size of the figure".
    """
    h, w = gray.shape
    S = saliency_map(gray)
    tot = S.sum()
    yy, xx = np.mgrid[0:h, 0:w]
    half_diag = 0.5 * np.hypot(h, w)
    out = {}
    if tot <= 1e-12:
        out.update(saliency_offset=np.nan, saliency_dispersion=np.nan,
                   thirds_alignment=np.nan, figure_ground_proxy=np.nan)
    else:
        cy = float((S * yy).sum() / tot)
        cx = float((S * xx).sum() / tot)
        out["saliency_offset"] = float(np.hypot(cy - (h - 1) / 2, cx - (w - 1) / 2) / half_diag)
        var = float((S * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot)
        out["saliency_dispersion"] = float(np.sqrt(var) / half_diag)
        thirds = [(fy * (h - 1), fx * (w - 1))
                  for fy in (1 / 3, 2 / 3) for fx in (1 / 3, 2 / 3)]
        d = min(np.hypot(cy - ty, cx - tx) for ty, tx in thirds)
        out["thirds_alignment"] = float(1.0 - d / half_diag)
        from skimage.filters import threshold_otsu
        try:
            t = threshold_otsu(S)
            out["figure_ground_proxy"] = float((S > t).mean())
        except ValueError:
            out["figure_ground_proxy"] = np.nan

    gy = np.abs(ndi.sobel(gray, axis=0))
    prof = ndi.uniform_filter1d(gy.mean(axis=1), size=max(3, h // 60))
    lo, hi = int(0.15 * h), int(0.85 * h)
    seg = prof[lo:hi]
    if seg.size and prof.mean() > 1e-12:
        r = int(np.argmax(seg)) + lo
        out["horizon_height"] = float(r / (h - 1))
        out["horizon_confidence"] = float(seg.max() / prof.mean())
    else:
        out["horizon_height"] = np.nan
        out["horizon_confidence"] = np.nan
    return out


# ------------------------------------------------------------------- assembly
DESCRIPTOR_ORDER = [
    "palette_entropy", "palette_bimodality", "warm_ratio", "warm_cold_moran",
    "luminance_mean", "luminance_range", "rms_contrast", "hue_dispersion",
    "colourfulness", "edge_density", "orientation_entropy", "glcm_energy",
    "glcm_contrast", "saliency_offset", "saliency_dispersion", "horizon_height",
    "thirds_alignment",
]

# Computed and stored, but NOT part of the confirmatory descriptor matrix.
#
# figure_ground_proxy was designed as the machine counterpart of the manual code
# M05 (body-to-space scale) and failed validation: against discs of known area
# spanning 0.01-0.45 of the frame it gave Pearson r = 0.17 (Spearman 0.89), with
# a gross inversion at one radius. The cause is structural, not a thresholding
# choice - spectral-residual saliency responds to *boundaries*, so a large object
# produces a thin rim of saliency rather than a filled region, and a
# threshold-free variant (area holding 50% of the saliency mass) failed the same
# way. Measuring figure area needs an actual segmentation model, which this
# pipeline deliberately does not carry.
#
# Body-to-space scale is therefore measured by the manual code M05 alone. This
# value is retained only so the failure stays visible and reproducible.
EXPLORATORY_DESCRIPTORS = ["figure_ground_proxy"]

DESCRIPTOR_IDS = {name: f"C{i + 1:02d}" for i, name in enumerate(DESCRIPTOR_ORDER)}
DESCRIPTOR_IDS.update({name: f"X{i + 1:02d}" for i, name in enumerate(EXPLORATORY_DESCRIPTORS)})


def describe_frame(rgb: np.ndarray, work_height: int = WORK_HEIGHT) -> dict:
    """Compute all descriptors for one RGB frame in [0, 1].

    Colour and luminance descriptors (C01-C09) are computed at native resolution;
    scale-dependent descriptors (C10-C18) at a common working height, because
    Canny, GLCM and the saliency spectrum all depend on pixel scale and the corpus
    spans SD and HD transfers.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB array")
    if rgb.max() > 1.001:
        raise ValueError("expected RGB in [0, 1]; divide by 255 first")

    lab = rgb2lab(rgb)
    work = _work(rgb, work_height)
    gray = rgb2lab(work)[..., 0] / 100.0

    d: dict = {}
    d.update(palette_descriptors(lab))
    d.update(warm_cold_descriptors(lab))
    d.update(luminance_descriptors(lab))
    d.update(hue_descriptors(lab))
    d.update(colourfulness(rgb))
    d.update(edge_descriptors(gray))
    d.update(texture_descriptors(gray))
    d.update(composition_descriptors(gray))
    d["px_height"], d["px_width"] = int(rgb.shape[0]), int(rgb.shape[1])
    d["work_height"] = int(work_height)
    return d


# ------------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("frame_descriptors.csv"))
    ap.add_argument("--editions", type=Path, default=None,
                    help="apply crop_top/crop_bottom per film")
    ap.add_argument("--work-height", type=int, default=WORK_HEIGHT)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="frames/manifest.csv; joins frame_id, seq_id and film_id")
    args = ap.parse_args()

    crops = {}
    if args.editions and args.editions.is_file():
        with open(args.editions, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                crops[r["film_id"].strip()] = (int(float(r.get("crop_top") or 0)),
                                               int(float(r.get("crop_bottom") or 0)))

    man = {}
    man_path = args.manifest or (args.frames / "manifest.csv")
    if man_path.is_file():
        with open(man_path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                man[Path(r["path"]).name] = r

    pngs = sorted(p for p in args.frames.glob("*.png"))
    if not pngs:
        print(f"no PNGs in {args.frames}", file=sys.stderr)
        sys.exit(2)

    rows = []
    for i, p in enumerate(pngs, 1):
        meta = man.get(p.name, {})
        film = meta.get("film_id") or p.name.split("_")[0]
        ct, cb = crops.get(film, (0, 0))
        rgb = load_frame(p, ct, cb)
        d = describe_frame(rgb, args.work_height)
        row = {"frame_id": meta.get("frame_id", p.stem), "seq_id": meta.get("seq_id", ""),
               "film_id": film, "frame_role": meta.get("frame_role", ""),
               "frame_index": meta.get("frame_index", ""), "file": p.name,
               "sha256": meta.get("sha256", "")}
        row.update(d)
        rows.append(row)
        print(f"[{i}/{len(pngs)}] {p.name}")

    cols = (["frame_id", "seq_id", "film_id", "frame_role", "frame_index", "file", "sha256"]
            + DESCRIPTOR_ORDER + EXPLORATORY_DESCRIPTORS
            + ["achromatic_fraction", "horizon_confidence", "palette_degenerate",
               "palette_k_effective", "warm_cold_grid", "px_height", "px_width",
               "work_height"])
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    sidecar = args.out.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({"descriptor_ids": DESCRIPTOR_IDS,
                                   "confirmatory": DESCRIPTOR_ORDER,
                                   "exploratory": EXPLORATORY_DESCRIPTORS,
                                   "chroma_min": CHROMA_MIN,
                                   "warm_axis_deg": WARM_AXIS_DEG,
                                   "n_palette": N_PALETTE,
                                   "glcm_levels": GLCM_LEVELS,
                                   "work_height": args.work_height,
                                   "random_state": RANDOM_STATE}, indent=2))
    print(f"\nwrote {args.out} ({len(rows)} frames) and {sidecar.name}")


if __name__ == "__main__":
    main()

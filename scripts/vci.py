#!/usr/bin/env python3
"""
vci.py — the Visual Contradiction Index.

VCI operationalises the construct at the centre of the study: the co-presence,
within one frame, of two image codes in tension. It has a measured component
computed from the pixels and a coded component supplied by the human coder.

    VCI = w * VCI_C + (1 - w) * VCI_M          w = 0.5 preregistered

Both components lie in [0, 1], so VCI does too.

VCI_C — measured (pixels)
-------------------------
    VCI_C = bimodality * max(0, segregation)

    bimodality  = C02 (frame_descriptors.palette_descriptors): the frame splits
                  into two balanced, well-separated CIELAB populations.
    segregation = Moran's I of the 2-means *label* field (+1 / -1), block-reduced
                  to a coarse grid with rook adjacency.

    The two factors are both necessary and neither is sufficient:

      * flat field                  -> bimodality 0            -> VCI_C = 0
      * fine checkerboard of two    -> bimodality ~1,
        flat colours                   segregation < 0         -> VCI_C = 0
      * the same two colours as a   -> bimodality ~1,
        left/right split               segregation ~ +1        -> VCI_C ~ 1

    The checkerboard case is the point. Stippling two opposed colours together is
    texture; the construct requires the two codes to *occupy* distinguishable
    parts of the image. Segregation is clipped at 0 because negative
    autocorrelation is not "negative contradiction" — it is a different thing
    (dithering), and folding it in as a negative score would let fine texture pull
    an otherwise contradictory frame downward.

    Segregation is computed here rather than taken from C04 because C04 is
    specific to the warm-cold hue axis, while the 2-means label field finds
    whatever two populations the frame actually has — which may be dark/light,
    saturated/desaturated, or green/grey rather than warm/cold.

VCI_M — coded (human)
---------------------
    VCI_M = M10 / 2,  i.e. 0 for a single pole, 0.5 for poles in succession
            across the sequence, 1.0 for poles co-present in the frame.

What is deliberately NOT in the index
-------------------------------------
M12 (contradiction locus), M09 (off-frame space) and M05 (body-to-space scale)
are all *outcome variables* in hypothesis H2. Folding any of them into VCI would
make H2c — "the magnitude of contradiction does not rise with period, only its
locus moves" — circular: the index would contain the thing it is being compared
against. VCI therefore measures magnitude only, from M10 and the pixels, and
locus is measured separately.

This is the single most important constraint on the index, and it is the reason
VCI_M uses one coded variable rather than a richer combination.

Weighting
---------
w = 0.5 is preregistered: there is no principled basis for privileging the
measured over the coded component, and equal weight is the honest default. The
analysis reports VCI across the whole range w in [0, 1] (`vci_sensitivity.png`),
so a reader can see exactly how much any conclusion depends on the choice.
"""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab
from skimage.measure import block_reduce
from sklearn.cluster import KMeans

from frame_descriptors import (CHROMA_MIN, RANDOM_STATE, _morans_i, _work,
                               palette_descriptors)

W_DEFAULT = 0.5
SEG_GRID = 64


def palette_segregation(lab: np.ndarray, grid: int = SEG_GRID,
                        max_pixels: int = 200_000) -> float:
    """Moran's I of the 2-means label field. +1 = two contiguous regions, <0 = interleaved.

    The label field is built at full resolution and then block-reduced with
    disjoint block means (never an anti-aliased resize, which shares pixels
    between output cells and inflates I - see validation_report.md §2).
    """
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3)
    if flat.shape[0] > max_pixels:
        rs = np.random.default_rng(RANDOM_STATE)
        idx = rs.choice(flat.shape[0], max_pixels, replace=False)
        km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(flat[idx])
        labels = km.predict(flat)
    else:
        if np.allclose(flat.std(axis=0), 0):
            return np.nan
        labels = KMeans(n_clusters=2, n_init=10,
                        random_state=RANDOM_STATE).fit_predict(flat)
    field = np.where(labels.reshape(h, w) == 0, 1.0, -1.0)
    blk = max(1, int(round(max(h, w) / grid)))
    return float(_morans_i(block_reduce(field, (blk, blk), func=np.mean)))


def vci_computed(rgb: np.ndarray, work_height: int = 360) -> dict:
    """VCI_C and its two factors for one RGB frame in [0, 1]."""
    lab = rgb2lab(_work(rgb, work_height))
    bim = palette_descriptors(lab)["palette_bimodality"]
    seg = palette_segregation(lab)
    seg_pos = 0.0 if not np.isfinite(seg) else max(0.0, seg)
    return {"vci_bimodality": float(bim), "vci_segregation": float(seg),
            "vci_segregation_clipped": float(seg_pos),
            "vci_computed": float(bim * seg_pos)}


def vci_coded(pole_copresence) -> float:
    """VCI_M from M10. Returns NaN when M10 is missing (never silently 0)."""
    try:
        m = float(pole_copresence)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(m) or m < 0 or m > 2:
        return np.nan
    return m / 2.0


def vci(rgb_or_vci_c, pole_copresence, w: float = W_DEFAULT) -> float:
    """Combine the two components. Accepts a frame array or a precomputed VCI_C."""
    if isinstance(rgb_or_vci_c, np.ndarray) and rgb_or_vci_c.ndim == 3:
        c = vci_computed(rgb_or_vci_c)["vci_computed"]
    else:
        c = float(rgb_or_vci_c)
    m = vci_coded(pole_copresence)
    if not np.isfinite(m):
        return np.nan
    return float(w * c + (1.0 - w) * m)

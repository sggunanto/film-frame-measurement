"""Helpers for film-frame-measurement. See SKILL.md stages 4-6."""
import numpy as np
import pandas as pd

CHROMA_MIN = 8.0
WARM_AXIS_DEG = 60.0


def block_grid(field, grid=64):
    """Downsample a 2-D field by DISJOINT block means.

    Use this, never resize(..., anti_aliasing=True), before any spatial
    statistic: an anti-aliased kernel is wider than one output cell, so
    neighbouring cells share input pixels and autocorrelation is inflated
    (white noise measured Moran's I = +0.30 before this was fixed).
    """
    a = np.asarray(field, dtype=float)
    h, w = a.shape
    blk = max(1, int(round(max(h, w) / grid)))
    hh, ww = (h // blk) * blk, (w // blk) * blk
    if hh < blk or ww < blk:
        return a
    return a[:hh, :ww].reshape(hh // blk, blk, ww // blk, blk).mean(axis=(1, 3))


def morans_i_grid(z):
    """Moran's I with rook adjacency on a regular grid; NaN if z has no variance.

    Near +1 = like next to like (two contiguous regions). Negative = interleaved
    (a checkerboard or dither). Near 0 = no spatial structure.
    """
    x = np.asarray(z, dtype=float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom < 1e-12:
        return float("nan")
    num = 0.0
    nw = 0
    for axis in (0, 1):
        a = np.take(x, range(0, x.shape[axis] - 1), axis=axis)
        b = np.take(x, range(1, x.shape[axis]), axis=axis)
        num += 2.0 * (a * b).sum()
        nw += 2 * a.size
    return float((x.size / nw) * (num / denom))


def warm_cold(rgb, grid=64):
    """Warm mass ratio and warm-cold spatial segregation from an RGB frame in [0,1].

    warm_ratio separates how much; warm_cold_moran separates where. A left-right
    split and a fine checkerboard of the same two colours share a warm_ratio of
    0.5 and have opposite segregation - that contrast is the point.
    """
    from skimage.color import rgb2lab
    lab = rgb2lab(np.asarray(rgb, dtype=float))
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a)
    chromatic = chroma > CHROMA_MIN
    axis_rad = np.deg2rad(WARM_AXIS_DEG)
    warm = np.cos(hue - axis_rad) > 0
    warmth = np.where(chromatic, chroma * np.cos(hue - axis_rad), 0.0)
    return {"warm_ratio": float(warm[chromatic].mean()) if chromatic.any() else float("nan"),
            "achromatic_fraction": float(1.0 - chromatic.mean()),
            "warm_cold_moran": morans_i_grid(block_grid(warmth, grid))}


def palette_split(rgb, seed=0):
    """Two-population colour split: bimodality in [0,1] and spatial segregation.

    bimodality = dE/(dE + 2s) * 2*min(p1,p2) over a 2-means CIELAB partition.
    The regularised denominator matters: dividing by s alone returns 0 for two
    flat colours (s = 0), i.e. the maximum case reported as the minimum.
    """
    from skimage.color import rgb2lab
    from sklearn.cluster import KMeans
    arr = np.asarray(rgb, dtype=float)
    lab = rgb2lab(arr)
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3)
    if np.allclose(flat.std(axis=0), 0):
        return {"bimodality": 0.0, "segregation": float("nan"), "degenerate": True}
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(flat)
    c1, c2 = km.cluster_centers_
    m = km.labels_ == 0
    p1 = float(m.mean())
    within = np.concatenate([np.linalg.norm(flat[m] - c1, axis=1),
                             np.linalg.norm(flat[~m] - c2, axis=1)]).mean()
    de = float(np.linalg.norm(c1 - c2))
    sep = de / (de + 2.0 * within) if (de + 2.0 * within) > 1e-12 else 0.0
    field = np.where(km.labels_.reshape(h, w) == 0, 1.0, -1.0)
    return {"bimodality": float(sep * 2 * min(p1, 1 - p1)),
            "segregation": morans_i_grid(block_grid(field)), "degenerate": False}


def tension_index(rgb, seed=0):
    """Two-factor visual-tension index: bimodality x max(0, segregation), in [0,1].

    Both factors are necessary, neither sufficient: stippling two opposed colours
    is texture, not tension, so negative segregation is clipped to 0. Note the
    known weakness - a small region of a very distinct colour captures the
    partition and collapses the balance term, so a single flame in a dark room
    scores LOW. Pair it with a human code rather than using it alone.
    """
    p = palette_split(rgb, seed=seed)
    seg = p["segregation"]
    seg_pos = 0.0 if not np.isfinite(seg) else max(0.0, seg)
    return {"tension": float(p["bimodality"] * seg_pos), **p}


def alpha_with_ci(data, level="nominal", n_boot=2000, seed=0):
    """Krippendorff's alpha with a bootstrap CI over UNITS (columns), not ratings.

    data: (n_coders, n_units) float array, np.nan for missing. level is
    nominal / ordinal / interval / ratio - match it to the variable's scale, or
    an ordinal code whose coders agree within one level scores 0.
    """
    import krippendorff
    d = np.asarray(data, dtype=float)
    usable = np.isfinite(d).sum(axis=0) >= 2
    if usable.sum() < 2:
        return {"alpha": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_units": int(usable.sum()), "n_boot_valid": 0}
    d = d[:, usable]
    point = float(krippendorff.alpha(reliability_data=d, level_of_measurement=level))
    rng = np.random.default_rng(seed)
    boots = []
    for i in range(n_boot):
        idx = rng.integers(0, d.shape[1], d.shape[1])
        try:
            a = krippendorff.alpha(reliability_data=d[:, idx], level_of_measurement=level)
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(a):
            boots.append(float(a))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) >= 100 else (float("nan"), float("nan"))
    return {"alpha": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_units": int(d.shape[1]), "n_boot_valid": len(boots)}


def permanova(X, groups, strata=None, n_perm=9999, seed=0):
    """PERMANOVA (Anderson 2001) on Euclidean distances, with restricted permutation.

    strata restricts shuffling to within each stratum. Frames inside one film
    share a palette and a transfer, so they are not exchangeable with another
    film's - unrestricted permutation tests a null nobody believes.
    """
    from scipy.spatial.distance import pdist, squareform
    Xa = np.asarray(X, dtype=float)
    g = np.asarray(groups)
    D = squareform(pdist(Xa, metric="euclidean")) ** 2
    n = len(g)

    def stat(labels):
        ss_total = D.sum() / (2 * n)
        ss_within = 0.0
        for lev in np.unique(labels):
            idx = np.where(labels == lev)[0]
            if len(idx) > 1:
                ss_within += D[np.ix_(idx, idx)].sum() / (2 * len(idx))
        a = len(np.unique(labels))
        if ss_within <= 0 or a < 2:
            return float("nan"), float("nan")
        return ((ss_total - ss_within) / (a - 1)) / (ss_within / (n - a)), (ss_total - ss_within) / ss_total

    f_obs, r2 = stat(g)
    rng = np.random.default_rng(seed)
    st = None if strata is None else np.asarray(strata)
    hits = 0
    for i in range(n_perm):
        gp = g.copy()
        if st is None:
            gp = rng.permutation(gp)
        else:
            for lev in np.unique(st):
                idx = np.where(st == lev)[0]
                gp[idx] = rng.permutation(gp[idx])
        f_perm = stat(gp)[0]
        if np.isfinite(f_perm) and f_perm >= f_obs:
            hits += 1
    return {"pseudo_F": float(f_obs), "R2": float(r2), "p_perm": (hits + 1) / (n_perm + 1),
            "n_perm": n_perm, "restricted": st is not None}


def within_block_permutation(df, outcome, block_col, group_col, unit_col,
                             n_perm=10000, seed=0):
    """Permutation test of a two-level contrast computed WITHIN blocks.

    Units are exchanged between the two groups inside each block, so the test
    never has to estimate a variance from a handful of top-level clusters.
    Blocks whose group has only one level are dropped and counted.
    """
    frame = df.dropna(subset=[outcome]).copy()
    unit = frame.groupby([block_col, group_col, unit_col], as_index=False)[outcome].mean()
    blocks = [b for b in unit[block_col].unique()
              if unit[unit[block_col] == b][group_col].nunique() == 2]
    if not blocks:
        return {"stat": float("nan"), "p_perm": float("nan"), "blocks_used": 0}
    levels = sorted(unit[unit[block_col].isin(blocks)][group_col].unique())

    def stat(tab):
        diffs = []
        for b in blocks:
            s = tab[tab[block_col] == b]
            diffs.append(s[s[group_col] == levels[1]][outcome].mean()
                         - s[s[group_col] == levels[0]][outcome].mean())
        return float(np.mean(diffs))

    obs = stat(unit)
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(n_perm):
        sp = unit.copy()
        for b in blocks:
            m = sp[block_col] == b
            sp.loc[m, group_col] = rng.permutation(sp.loc[m, group_col].values)
        if abs(stat(sp)) >= abs(obs):
            hits += 1
    return {"stat": obs, "p_perm": (hits + 1) / (n_perm + 1),
            "blocks_used": len(blocks), "contrast": f"{levels[1]} - {levels[0]}"}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, input order preserved."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return list(adj)

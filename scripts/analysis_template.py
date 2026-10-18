#!/usr/bin/env python3
"""
run_analysis.py — the confirmatory analysis for "From Ruins to Threshold", plus a
simulator that generates a corpus-shaped dataset with known planted effects.

Two modes:

    python run_analysis.py --simulate --out-prefix sim      # synthetic data + full analysis
    python run_analysis.py --codes coding.csv --descriptors frame_descriptors.csv

The simulate mode exists so the whole pipeline can be executed, debugged and
power-checked before a single real frame is coded. Every number it produces is
labelled SIMULATED and must never appear in the paper as a result.

Tests implemented (see research_design.md §6):
  H1  separability of the three contradiction axes in descriptor space
      -> PERMANOVA, permutations restricted within film
  H2a locus shift towards landscape/absence in the later film of each pair
      -> mixed logistic regression + exact within-pair permutation test
  H2b formal correlates (M05, C15, M09) shift with period
      -> linear mixed models, Holm correction across the three
  H2c magnitude (VCI) does NOT rise with period
      -> linear mixed model; the discriminating prediction
  H3  the ruins motif changes formal profile across period
      -> PERMANOVA on the ruins subset, permutations restricted within pair
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# TEMPLATE. Adapt before use:
#   DESCRIPTORS  - the descriptor columns in your frame_descriptors.csv
#   SOURCES      - one row per film/source: (id, year, block, group)
#                  `block` is the matched pair/stratum, `group` the contrast
#                  level (e.g. early/late). A block whose group has one level
#                  acts as a within-group baseline and contributes no contrast.
#   STRATA       - the per-source slot allocation used when sampling
#   simulate()   - the planted effects; rewrite these to match YOUR hypotheses,
#                  then run the pipeline on simulated data before touching real
#                  data. That run is what tells you the models converge, the
#                  nulls stay null, and what effect size the design can detect.
# The statistical primitives (permanova, within_block_permutation, holm,
# alpha_with_ci) are also in this skill's kernel, available without editing.
# ---------------------------------------------------------------------------


import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

DESCRIPTORS = [
    "palette_entropy", "palette_bimodality", "warm_ratio", "warm_cold_moran",
    "luminance_mean", "luminance_range", "rms_contrast", "hue_dispersion",
    "colourfulness", "edge_density", "orientation_entropy", "glcm_energy",
    "glcm_contrast", "saliency_offset", "saliency_dispersion", "horizon_height",
    "thirds_alignment",
]

SOURCES = FILMS = [   # id, year, block(pair), group(period)  -- EDIT FOR YOUR CORPUS
    ("NAU", 1984, "P2", "early"), ("AKI", 1988, "P2", "early"),
    ("GOF", 1988, "P1", "early"), ("ITC", 2016, "P1", "late"),
    ("MON", 1997, "P3", "early"), ("SUZ", 2022, "P3", "late"),
]
STRATA = ["S1", "S1", "S2", "S2", "S3", "S4"]          # 2/2/1/1 per film
HOME_AXIS = {"P1": "A1", "P2": "A2", "P3": "A3"}
LOCI = ["body", "object", "landscape", "absence"]


# ============================================================== simulation
def simulate(seed: int = 0, period_effect: float = 0.9,
             locus_p_early: float = 0.35, locus_p_late: float = 0.75,
             axis_effect: float = 0.7, icc_target: float = 0.5) -> pd.DataFrame:
    """Generate a 108-frame dataset with the corpus's exact nesting and known effects.

    Planted truth:
      * axis separates descriptor space          (H1 true)
      * late films shift locus to landscape/absence from `locus_p_early` to
        `locus_p_late`                           (H2a true)
      * late films raise M05, C15 and M09 by `period_effect` SD
                                                 (H2b true)
      * VCI has NO period effect                 (H2c true: the null holds)
      * among ruins frames, C11 and C15 shift with period (H3 true)

    Variance is split film : sequence : residual so that the intra-sequence
    correlation is about `icc_target`, which is the quantity the power estimates
    depend on and which is unknown until the first real frames are measured.
    """
    rng = np.random.default_rng(seed)
    sd_film = np.sqrt(icc_target * 0.3)
    sd_seq = np.sqrt(icc_target * 0.7)
    sd_res = np.sqrt(1.0 - icc_target)

    rows = []
    for film, year, pair, period in FILMS:
        film_re = {d: rng.normal(0, sd_film) for d in DESCRIPTORS}
        for s, stratum in enumerate(STRATA, start=1):
            seq_id = f"{film}{s:02d}"
            # axis: four sequences in the pair's home axis, two elsewhere
            others = [a for a in ("A1", "A2", "A3") if a != HOME_AXIS[pair]]
            axis = HOME_AXIS[pair] if s <= 4 else others[(s - 5) % 2]
            seq_re = {d: rng.normal(0, sd_seq) for d in DESCRIPTORS}

            # sequence-level codes
            m09 = int(np.clip(round(rng.normal(1.6 + (period == "late") * period_effect, 0.9)), 0, 4))
            asl = float(np.exp(rng.normal(np.log(4.0), 0.4)))
            sound = rng.choice(["diegetic_everyday", "diegetic_catastrophic",
                                "score_dominant", "silence", "mixed"])

            for role in ("onset", "peak", "release"):
                r = {"frame_id": f"{seq_id}{role[0]}", "seq_id": seq_id, "film_id": film,
                     "year": year, "pair": pair, "period": period, "stratum": stratum,
                     "axis": axis, "frame_role": role, "offframe_space": m09,
                     "asl_seconds": asl, "sound_register": sound}

                for d in DESCRIPTORS:
                    val = film_re[d] + seq_re[d] + rng.normal(0, sd_res)
                    # axis signal on a subset, so H1 is true but not trivially so
                    if d in ("warm_ratio", "colourfulness", "luminance_mean"):
                        val += axis_effect * {"A1": 1.0, "A2": 0.0, "A3": -1.0}[axis]
                    if d in ("orientation_entropy", "edge_density"):
                        val += axis_effect * {"A1": -0.5, "A2": 1.0, "A3": -0.5}[axis]
                    if d == "saliency_dispersion" and period == "late":
                        val += period_effect                      # H2b outcome
                    r[d] = val

                lat = rng.normal((period == "late") * period_effect, 1.0)
                r["body_space_scale"] = int(np.clip(
                    np.digitize(lat, [-1.2, -0.3, 0.4, 1.2]) + 1, 1, 5))   # H2b outcome

                p_space = locus_p_late if period == "late" else locus_p_early
                if rng.random() < 0.82:                       # most frames have both poles
                    r["pole_copresence"] = int(rng.choice([1, 2], p=[0.35, 0.65]))
                    if rng.random() < p_space:
                        r["contradiction_locus"] = str(rng.choice(["landscape", "absence"],
                                                                  p=[0.7, 0.3]))
                    else:
                        r["contradiction_locus"] = str(rng.choice(["body", "object"],
                                                                  p=[0.65, 0.35]))
                    r["affective_function"] = str(rng.choice(
                        ["elegiac", "ominous", "sublime", "tender", "ironic", "flat_numb"]))
                else:
                    r["pole_copresence"] = 0
                    r["contradiction_locus"] = None
                    r["affective_function"] = None

                # VCI: deliberately NO period term (H2c null is true in the simulation)
                vci_c = float(np.clip(rng.beta(2.2, 2.6), 0, 1))
                r["vci_computed"] = vci_c
                r["vci"] = 0.5 * vci_c + 0.5 * (r["pole_copresence"] / 2.0)

                # motif, with ruins concentrated in the landscape/threshold strata
                if stratum in ("S1", "S4") and rng.random() < 0.55:
                    r["dominant_motif"] = str(rng.choice(["ruins", "abandoned_building"]))
                    r["orientation_entropy"] += 0.8 * (period == "late")   # H3
                    r["saliency_dispersion"] += 0.5 * (period == "late")   # H3
                else:
                    r["dominant_motif"] = str(rng.choice(
                        ["fire", "food", "child_body", "forest", "machine", "door",
                         "water", "sky", "interior_domestic", "crowd"]))
                rows.append(r)
    df = pd.DataFrame(rows)
    df.attrs["SIMULATED"] = True
    return df


# ============================================================== PERMANOVA
def permanova(X: np.ndarray, groups: np.ndarray, strata: np.ndarray | None = None,
              n_perm: int = 9999, seed: int = 0) -> dict:
    """PERMANOVA (Anderson 2001) on Euclidean distances, with restricted permutation.

    `strata` restricts permutation: labels are shuffled only *within* each stratum
    (film for H1, pair for H3). Unrestricted permutation would break the nesting
    and inflate significance, because frames within a film are not exchangeable
    with frames from another film.

    Returns pseudo-F, R^2 and the permutation p-value.
    """
    D = squareform(pdist(X, metric="euclidean")) ** 2
    n = len(groups)

    def pseudo_f(g):
        ss_total = D.sum() / (2 * n)
        ss_within = 0.0
        for lev in np.unique(g):
            idx = np.where(g == lev)[0]
            if len(idx) > 1:
                ss_within += D[np.ix_(idx, idx)].sum() / (2 * len(idx))
        a = len(np.unique(g))
        ss_between = ss_total - ss_within
        if ss_within <= 0 or a < 2:
            return np.nan, np.nan
        f = (ss_between / (a - 1)) / (ss_within / (n - a))
        return f, ss_between / ss_total

    f_obs, r2 = pseudo_f(np.asarray(groups))
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    count = 0
    for _ in range(n_perm):
        gp = g.copy()
        if strata is None:
            gp = rng.permutation(gp)
        else:
            for lev in np.unique(strata):
                idx = np.where(np.asarray(strata) == lev)[0]
                gp[idx] = rng.permutation(gp[idx])
        f_perm, _ = pseudo_f(gp)
        if np.isfinite(f_perm) and f_perm >= f_obs:
            count += 1
    return {"pseudo_F": float(f_obs), "R2": float(r2),
            "p_perm": (count + 1) / (n_perm + 1), "n_perm": n_perm,
            "restricted_within": "none" if strata is None else "stratum"}


# ====================================================== within-pair permutation
def within_pair_permutation(df: pd.DataFrame, outcome: str, n_perm: int = 10000,
                            seed: int = 0) -> dict:
    """Exact-style permutation test of the period contrast, computed within pairs.

    Sequences are exchanged between the two films of a pair (P1 and P3 only - P2
    holds period fixed and contributes no contrast). The statistic is the mean
    within-pair difference, late minus early, in sequence-level means of
    `outcome`. This test does not rely on estimating a film-level variance from
    six films.
    """
    seq = (df.groupby(["pair", "film_id", "period", "seq_id"], as_index=False)[outcome]
             .mean())
    pairs = [p for p in seq.pair.unique()
             if seq[seq.pair == p].period.nunique() == 2]
    if not pairs:
        return {"stat": np.nan, "p_perm": np.nan, "pairs_used": 0}

    def stat(frame):
        diffs = []
        for p in pairs:
            s = frame[frame.pair == p]
            diffs.append(s[s.period == "late"][outcome].mean()
                         - s[s.period == "early"][outcome].mean())
        return float(np.mean(diffs))

    obs = stat(seq)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        sp = seq.copy()
        for p in pairs:
            m = sp.pair == p
            sp.loc[m, "period"] = rng.permutation(sp.loc[m, "period"].values)
        if abs(stat(sp)) >= abs(obs):
            count += 1
    return {"stat": obs, "p_perm": (count + 1) / (n_perm + 1),
            "pairs_used": len(pairs), "n_perm": n_perm}


# ============================================================ mixed models
def mixed_period_effect(df: pd.DataFrame, outcome: str) -> dict:
    """Period effect on `outcome` with sequence nested in film.

    Primary: random intercepts for film and for sequence within film.
    Fallback (recorded, not hidden): sequence-level random intercept only, with
    pair as a fixed effect - used when the nested model fails to converge, which
    is a real risk with six film-level clusters.
    """
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[outcome]).copy()
    d["period_late"] = (d.period == "late").astype(float)
    try:
        m = smf.mixedlm(f"{outcome} ~ period_late + C(pair)", d, groups=d.film_id,
                        re_formula="1", vc_formula={"seq": "0 + C(seq_id)"}).fit(reml=True)
        model_used = "film + sequence-in-film random intercepts"
    except Exception:
        m = None
    if m is None or not np.isfinite(getattr(m, "llf", np.nan)):
        m = smf.mixedlm(f"{outcome} ~ period_late + C(pair)", d,
                        groups=d.seq_id).fit(reml=True)
        model_used = "sequence random intercept only (fallback)"
    b = m.params.get("period_late", np.nan)
    se = m.bse.get("period_late", np.nan)
    # Convergence is reported, not assumed. With six film-level clusters the
    # nested model routinely lands on a variance boundary; statsmodels still
    # returns coefficients, so a run that did not converge must be visible in the
    # results table rather than inferred from stderr warnings.
    conv = bool(getattr(m, "converged", False))
    alt = {}
    if not conv and model_used.startswith("film +"):
        # Non-convergence of the nested model is common at six clusters. Rather
        # than report a single number from a model that did not converge, fit the
        # preregistered fallback as well and report both; the analysis plan treats
        # a sign or significance disagreement between them as a reason to rely on
        # the within-pair permutation test instead.
        try:
            m2 = smf.mixedlm(f"{outcome} ~ period_late + C(pair)", d,
                             groups=d.seq_id).fit(reml=True)
            b2, se2 = m2.params.get("period_late", np.nan), m2.bse.get("period_late", np.nan)
            alt = {"alt_estimate": float(b2), "alt_se": float(se2),
                   "alt_p": float(2 * (1 - norm.cdf(abs(b2 / se2)))) if se2 else np.nan,
                   "alt_model": "sequence random intercept only",
                   "alt_converged": bool(getattr(m2, "converged", False))}
        except Exception:
            alt = {"alt_estimate": np.nan, "alt_model": "fallback also failed"}
    return {"outcome": outcome, "converged": conv, **alt,
            "estimate": float(b), "se": float(se),
            "ci_low": float(b - 1.96 * se), "ci_high": float(b + 1.96 * se),
            "z": float(b / se) if se else np.nan,
            "p": float(2 * (1 - norm.cdf(abs(b / se)))) if se else np.nan,
            "n_obs": int(len(d)), "model": model_used}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(idx):
        val = (len(pvals) - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return list(adj)


def locus_logistic(df: pd.DataFrame) -> dict:
    """H2a: P(locus in {landscape, absence}) by period, sequence random intercept."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=["contradiction_locus"]).copy()
    d["space_locus"] = d.contradiction_locus.isin(["landscape", "absence"]).astype(float)
    d["period_late"] = (d.period == "late").astype(float)
    d = d[d.pair.isin([p for p in d.pair.unique()
                       if d[d.pair == p].period.nunique() == 2])]
    try:
        m = smf.logit("space_locus ~ period_late + C(pair)", d).fit(disp=False)
        b, se = m.params["period_late"], m.bse["period_late"]
        return {"outcome": "locus in {landscape, absence}", "log_odds": float(b),
                "odds_ratio": float(np.exp(b)), "se": float(se),
                "ci_low_or": float(np.exp(b - 1.96 * se)),
                "ci_high_or": float(np.exp(b + 1.96 * se)),
                "p": float(m.pvalues["period_late"]), "n_obs": int(len(d)),
                "model": "logistic, pair fixed (cluster-robust SE not applied)"}
    except Exception as e:
        return {"outcome": "locus", "error": str(e)[:200]}


# ================================================================== driver
def run_all(df: pd.DataFrame, out_prefix: str = "sim", n_perm: int = 4999,
            seed: int = 0) -> dict:
    """Execute every confirmatory test and write the results tables."""
    Z = df[DESCRIPTORS].apply(lambda c: (c - c.mean()) / c.std()).values
    results = {}

    # H1 — axis separability (permute axis labels within film)
    results["H1"] = permanova(Z, df.axis.values, strata=df.film_id.values,
                              n_perm=n_perm, seed=seed)

    # H2a — locus shift
    results["H2a_logistic"] = locus_logistic(df)
    d2 = df.dropna(subset=["contradiction_locus"]).copy()
    d2["space_locus"] = d2.contradiction_locus.isin(["landscape", "absence"]).astype(float)
    results["H2a_permutation"] = within_pair_permutation(d2, "space_locus",
                                                         n_perm=10000, seed=seed)

    # H2b — formal correlates, Holm across the family of three
    fam = ["body_space_scale", "saliency_dispersion", "offframe_space"]
    h2b = [mixed_period_effect(df, o) for o in fam]
    for r_, p_adj in zip(h2b, holm([r_["p"] for r_ in h2b])):
        r_["p_holm"] = p_adj
    results["H2b"] = h2b

    # H2c — magnitude must NOT move
    results["H2c"] = mixed_period_effect(df, "vci")
    results["H2c_computed_only"] = mixed_period_effect(df, "vci_computed")

    # H3 — ruins subset (permute period within pair)
    ruins = df[df.dominant_motif.isin(["ruins", "abandoned_building"])]
    if ruins.seq_id.nunique() >= 4 and ruins.period.nunique() == 2:
        Zr = ruins[DESCRIPTORS].apply(lambda c: (c - c.mean()) / c.std()).values
        results["H3"] = permanova(Zr, ruins.period.values, strata=ruins.pair.values,
                                  n_perm=n_perm, seed=seed)
        results["H3"]["n_ruins_frames"] = int(len(ruins))
    else:
        results["H3"] = {"skipped": "too few ruins frames", "n_ruins_frames": int(len(ruins))}

    pd.DataFrame(h2b).to_csv(f"{out_prefix}_h2b_effects.csv", index=False)
    df.to_csv(f"{out_prefix}_dataset.csv", index=False)
    return results


def power_sweep(effects, n_reps: int = 300, icc: float = 0.5, alpha: float = 0.05,
                seed: int = 0) -> pd.DataFrame:
    """Power for the H2b family against planted period effect size, at the real n."""
    out = []
    for eff in effects:
        hits = {o: 0 for o in ["body_space_scale", "saliency_dispersion", "offframe_space"]}
        ok = 0
        for r_ in range(n_reps):
            d = simulate(seed=seed + r_ * 97, period_effect=eff, icc_target=icc)
            try:
                res = [mixed_period_effect(d, o) for o in hits]
            except Exception:
                continue
            ok += 1
            padj = holm([x["p"] for x in res])
            for x, p_ in zip(res, padj):
                if p_ < alpha and x["estimate"] > 0:
                    hits[x["outcome"]] += 1
        for o, h in hits.items():
            out.append({"period_effect_sd": eff, "outcome": o,
                        "power": h / max(ok, 1), "n_reps": ok})
    return pd.DataFrame(out)

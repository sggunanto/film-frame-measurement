#!/usr/bin/env python3
"""
alpha_calculator.py — inter-coder reliability for the visual-contradiction codebook.

Computes Krippendorff's alpha per variable with the scale-appropriate distance
metric, bootstrap confidence intervals, and the disagreement listing used for
adjudication.

The alpha point estimate comes from the `krippendorff` package rather than a
hand-rolled coincidence matrix: the formula is easy to get subtly wrong on
missing data and unequal unit-wise coder counts, and a reference implementation
is the right dependency here. What this module adds is everything around it that
the protocol needs - per-variable scale handling, CIs, thresholds, and the
adjudication list.

Usage
-----
    python alpha_calculator.py --coder1 coding_c1.csv --coder2 coding_c2.csv \
        --fields coding_sheet_fields.csv --out reliability_results.csv

    from alpha_calculator import alpha_with_ci, reliability_table
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff

# Scale -> level of measurement passed to krippendorff.alpha
SCALE_MAP = {"nominal": "nominal", "ordinal": "ordinal", "ratio": "ratio",
             "interval": "interval", "boolean": "nominal", "multi-select": "nominal"}

# ---------------------------------------------------------------------------
# CONFIGURE THESE TWO DICTS FOR YOUR PROJECT.
# THRESHOLDS maps a variable class to its minimum acceptable alpha;
# VARIABLE_CLASS maps your variable names to those classes. Anything not listed
# falls back to "descriptive". The classes below are an example: the stricter
# bar applies to variables that carry hypotheses, the looser one to interpretive
# judgements whose conclusions will be stated as tentative.
# ---------------------------------------------------------------------------

# Minimum acceptable alpha by variable class. Krippendorff's own convention is
# 0.800 for conclusions and 0.667 for tentative ones; this project applies the
# stricter bar to the variables that carry hypotheses and the looser bar to the
# interpretive variables, and says which is which rather than reporting one
# number for everything.
THRESHOLDS = {
    "hypothesis_bearing": 0.800,   # M10, M12 - H2a's outcome and the construct's core
    "descriptive": 0.800,          # M01-M06, M08 - should be easy; failure means the codebook is unclear
    "interpretive": 0.667,         # M13 affective_function - tentative conclusions only
    "sequence_level": 0.800,       # M07, M09 - few units, so a high bar is affordable
}
VARIABLE_CLASS = {
    "pole_copresence": "hypothesis_bearing", "contradiction_locus": "hypothesis_bearing",
    "affective_function": "interpretive",
    "offframe_space": "sequence_level", "sound_register": "sequence_level",
    "asl_seconds": "sequence_level",
}


def _encode(series_pair: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Map a pair of coder columns to a (2, n_units) numeric matrix, NaN for missing."""
    vals = pd.unique(pd.concat([series_pair.iloc[:, 0], series_pair.iloc[:, 1]]).dropna())
    try:                                     # already numeric: keep the real values,
        order = sorted(vals, key=float)      # so ordinal distances mean something
        code = {v: float(v) for v in order}
    except (TypeError, ValueError):
        order = sorted(map(str, vals))
        code = {v: float(i) for i, v in enumerate(order)}
    out = np.full((2, len(series_pair)), np.nan)
    for j, col in enumerate(series_pair.columns):
        for i, v in enumerate(series_pair[col].values):
            if pd.notna(v):
                out[j, i] = code.get(v, code.get(str(v), np.nan))
    return out, order


def alpha_with_ci(data: np.ndarray, level: str = "nominal", n_boot: int = 2000,
                  seed: int = 0) -> dict:
    """Krippendorff's alpha with a bootstrap CI over units.

    Units (columns), not ratings, are resampled: the unit is the independent
    sampling entity. Bootstrap replicates that contain no usable pair, or whose
    values are all identical (expected disagreement zero, alpha undefined), are
    discarded and counted in `n_boot_valid` rather than silently treated as 1.0.
    """
    usable = np.isfinite(data).sum(axis=0) >= 2
    if usable.sum() < 2:
        return {"alpha": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_units": int(usable.sum()), "n_boot_valid": 0}
    d = data[:, usable]
    point = float(krippendorff.alpha(reliability_data=d, level_of_measurement=level))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, d.shape[1], d.shape[1])
        try:
            a = krippendorff.alpha(reliability_data=d[:, idx], level_of_measurement=level)
            if np.isfinite(a):
                boots.append(float(a))
        except (ValueError, ZeroDivisionError):
            continue
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) >= 100 else (np.nan, np.nan))
    return {"alpha": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_units": int(usable.sum()), "n_boot_valid": len(boots)}


def reliability_table(c1: pd.DataFrame, c2: pd.DataFrame, fields: pd.DataFrame,
                      key: str = "frame_id", n_boot: int = 2000) -> pd.DataFrame:
    """Per-variable alpha for every coded variable present in both coders' sheets."""
    merged = c1.merge(c2, on=key, suffixes=("_c1", "_c2"))
    rows = []
    for _, f in fields.iterrows():
        var, scale = f["field"], str(f["scale_type"])
        if str(f["round"]) not in {"1", "2", "1|2"} or scale == "text":
            continue
        a, b = f"{var}_c1", f"{var}_c2"
        if a not in merged or b not in merged:
            continue
        data, domain = _encode(merged[[a, b]])
        level = SCALE_MAP.get(scale, "nominal")
        res = alpha_with_ci(data, level, n_boot=n_boot)
        vclass = VARIABLE_CLASS.get(var, "descriptive")
        thr = THRESHOLDS[vclass]
        rows.append({"variable": var, "scale": scale, "level_used": level,
                     "variable_class": vclass, "threshold": thr,
                     "n_categories": len(domain), **res,
                     "meets_threshold": (np.isfinite(res["alpha"]) and res["alpha"] >= thr),
                     "ci_excludes_threshold": (np.isfinite(res["ci_low"])
                                               and res["ci_low"] >= thr)})
    return pd.DataFrame(rows)


def disagreements(c1: pd.DataFrame, c2: pd.DataFrame, variables: list,
                  key: str = "frame_id") -> pd.DataFrame:
    """Every unit where the two coders differ, long-form, for the adjudication session."""
    merged = c1.merge(c2, on=key, suffixes=("_c1", "_c2"))
    out = []
    for var in variables:
        a, b = f"{var}_c1", f"{var}_c2"
        if a not in merged or b not in merged:
            continue
        d = merged[merged[a].astype(str) != merged[b].astype(str)]
        for _, r in d.iterrows():
            out.append({key: r[key], "variable": var,
                        "coder_1": r[a], "coder_2": r[b],
                        "note_c1": r.get("coder_note_c1", ""),
                        "note_c2": r.get("coder_note_c2", ""),
                        "adjudicated_value": "", "codebook_change_needed": ""})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coder1", type=Path, required=True)
    ap.add_argument("--coder2", type=Path, required=True)
    ap.add_argument("--fields", type=Path, default=Path("coding_sheet_fields.csv"))
    ap.add_argument("--out", type=Path, default=Path("reliability_results.csv"))
    ap.add_argument("--key", default="frame_id")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    c1 = pd.read_csv(args.coder1)
    c2 = pd.read_csv(args.coder2)
    fields = pd.read_csv(args.fields)
    tab = reliability_table(c1, c2, fields, key=args.key, n_boot=args.n_boot)
    tab.to_csv(args.out, index=False)

    failing = tab[~tab.meets_threshold]
    print(tab[["variable", "level_used", "n_units", "alpha", "ci_low", "ci_high",
               "threshold", "meets_threshold"]].to_string(index=False))
    print(f"\n{len(tab) - len(failing)}/{len(tab)} variables meet their threshold")
    if len(failing):
        print("below threshold:", ", ".join(failing.variable))
        dis = disagreements(c1, c2, list(failing.variable), key=args.key)
        dis_path = args.out.with_name("disagreements_for_adjudication.csv")
        dis.to_csv(dis_path, index=False)
        print(f"wrote {dis_path} ({len(dis)} disagreements to adjudicate)")


if __name__ == "__main__":
    main()

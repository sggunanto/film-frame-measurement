# Reliability — sizing, decision rule, metric

Use `alpha_with_ci` (kernel) or `scripts/alpha_calculator.py` (per-variable table, thresholds,
adjudication list). The point estimate comes from the `krippendorff` package rather than a hand-rolled
coincidence matrix: the formula is easy to get subtly wrong under missing data and unequal coder counts.

## Size the subsample by simulation

Simulate two coders at a known agreement level, at *your* n, and look at the spread. Generator: coder 2
copies coder 1 with probability p, otherwise draws uniformly from k categories — for this generator the
true alpha equals p, so recovery is checkable (max bias 0.004 at n = 200).

With a **true** alpha of 0.80 on a 6-category nominal variable:

| units double-coded | mean observed | 95% of replicates | P(observed < 0.667) |
|---|---|---|---|
| 27 | 0.799 | 0.603 – 0.955 | **6.3%** |
| 36 | 0.801 | 0.636 – 0.933 | 4.7% |
| 45 | 0.800 | 0.654 – 0.920 | 3.5% |
| **54** | 0.800 | **0.667** – 0.911 | 2.5% |
| 72 | 0.800 | 0.694 – 0.900 | 0.5% |
| 108 | 0.802 | 0.720 – 0.889 | 0.0% |

The conventional 10–25 percent subsample is frequently too small to certify the threshold it exists to
certify: at 27 units a working instrument reports anywhere from 0.60 to 0.96. Pick the n where the 2.5th
percentile clears your lower threshold — here 54, i.e. 50 percent. The extra coding is the price of the
claim meaning anything.

## Judge the interval, not the point estimate

At every n above, P(observed < 0.800) is close to 0.50 when the true alpha is exactly 0.800 — arithmetic,
not a defect. A rule of "accept if the estimate reaches 0.80" rejects an adequate instrument half the
time.

**Accept when the lower bound of the 95 percent bootstrap interval clears the threshold for the
variable's class.** Report the estimate and the interval for every variable, passing or not. Bootstrap
**units**, not ratings — the unit is the independent sampling entity. Discard and count replicates where
expected disagreement is zero (alpha undefined); never score them as 1.0.

## Match the metric to the scale

A 5-level ordinal variable where every unit is off by exactly one level:

| disagreement rate | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| ordinal metric | 1.000 | 0.959 | 0.918 | 0.877 | 0.836 | **0.796** |
| nominal metric | 1.000 | 0.798 | 0.597 | 0.397 | 0.198 | **0.000** |

Treating an ordinal code as nominal rejects a codebook whose coders agree to within one level everywhere
— which, on a perceptual scale, is agreement. Print the metric used in the results table so the choice
is visible.

## Thresholds by class, not one number

| class | typical members | bar |
|---|---|---|
| hypothesis-bearing | the construct's core code, and any variable that is an outcome | 0.800 |
| descriptive | time, motif, viewpoint, movement, sound | 0.800 — failure here means the codebook is unclear, not the coder unreliable |
| interpretive | affective or evaluative judgements | 0.667, with conclusions stated as tentative |

Do not report a single alpha averaged across variables: it hides exactly the contrast a reader needs
between an easy descriptive variable and a hard interpretive one.

## Decide the failure policy in advance

Written before the alpha is known, so a caveat cannot quietly become a result:

- **A hypothesis-bearing variable fails after one revision** — the test depending on it is *not run*. The
  variable is reported with its alpha, the hypothesis is reported as untestable with this instrument, and
  the corresponding chapter is argued from close reading, labelled as interpretation.
- **A descriptive variable fails** — dropped from the matrix and reported as a codebook limitation.
- **An interpretive variable fails** — illustrative only; no claim rests on it.
- **Everything fails** — a finding about the construct (it cannot be coded consistently by two people)
  and publishable as one.

## Adjudication

Produce one row per disagreeing unit with both codes, both coders' notes, and empty columns for the
adjudicated value and whether a codebook change is needed. Work through it together. The output is a
codebook revision, not a corrected dataset.

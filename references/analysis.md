# Analysis — clustered data, restricted permutation, honest power

Primitives in the kernel: `permanova`, `within_block_permutation`, `holm`, `alpha_with_ci`.
Worked pipeline with a simulator: `scripts/analysis_template.py`.

## Checks that must pass before any model

Fatal, all of them — a model fitted on a broken join is worse than no model:

1. Row counts match the design exactly (units, sequences, sources).
2. Every unit id present in both the codes table and the descriptors table; no unmatched rows.
3. Frame checksums match the extraction manifest.
4. One codebook version across all rows.
5. Unit-above-frame variables constant within their group.
6. Every code requiring evidence has it — those rows are **missing**, not zero.

## Restrict permutations to the nesting

Frames within a source share a palette, a transfer and a designer; they are not exchangeable with another
source's. Permute **within source** for a factor that varies inside sources, **within block** for a
factor that varies between the members of a matched pair. Unrestricted permutation tests a null nobody
believes and inflates significance.

`permanova(X, groups, strata=...)` and `within_block_permutation(...)` both take the restriction
explicitly; passing `strata=None` should be a deliberate choice.

## Prefer the permutation test when clusters are few

A within-block permutation test does not estimate a variance from a handful of top-level clusters, which
is its advantage over the mixed model. With 12 units per block its minimum attainable two-sided p is
about 0.002 — small enough to be useful, honest about its resolution. Make it the primary inference and
report the model for its effect size.

Say in advance what happens if they disagree: the permutation result stands and the discrepancy is
reported as a model limitation, not resolved by preferring the friendlier number.

## Mixed models at few clusters

With a handful of source-level clusters, a nested random-effects model routinely lands on a variance
boundary **and still returns coefficients**. Three requirements:

- carry a `converged` column in the results table;
- when it is False, fit the simpler preregistered fallback (unit-level random intercept, block as a fixed
  effect) and report **both**;
- if the two disagree in sign or significance, use neither — fall back to the permutation test.

In the reference project, one of three outcomes failed to converge and the fallback returned an identical
estimate (0.972, p = 0.0003). Benign — but only visible because the column was there.

## Power from simulation

Textbook formulas ignore the clustering and the multiplicity correction. Simulate the exact design — real
n, real nesting, full pipeline including correction — and sweep the effect size, 300–400 replicates each.

From the reference project (108 frames, 36 sequences, 6 sources, correction across three outcomes):

| planted effect (SD) | unit-level ordinal code | frame-level ordinal code | frame-level continuous descriptor |
|---|---|---|---|
| 0.4 | 0.42 | 0.13 | 0.21 |
| 0.8 | **0.80** | 0.61 | 0.49 |
| 1.0 | 0.91 | **0.80** | 0.61 |
| 1.4 | 0.99 | 0.89 | **0.82** |

80 percent power needed 0.8, 1.0 and 1.4 SD respectively — power differs across outcomes by nearly a
factor of two, which a single number hides. The ordering is structural: a unit-level code has as many
independent values as there are units, while a frame-level descriptor's larger n carries only about as
much independent information, because frames within a unit are strongly correlated.

**The intra-unit correlation that drives this is unknown until real frames are measured.** Code and
measure a two-unit pilot first, estimate it, recompute the power, and report the recomputed figures.
Sensitivity in the same project: at an effect of 0.8 SD, the continuous descriptor's power fell from 0.59
to 0.41 as the intra-unit correlation rose from 0.3 to 0.7, while the unit-level code barely moved.

## Nulls need equivalence bounds

A prediction that something does *not* change cannot be supported by a non-significant p. In one
simulated draw with a planted effect of exactly zero, the test returned **p = 0.071**. State the bound in
advance: an effect whose 95 percent interval lies entirely within ±delta on the outcome's scale counts as
no material change; an interval wider than that is inconclusive and says so.

## Multiplicity

Correct within a family of outcomes testing one hypothesis (Holm). Separate preregistered hypotheses each
get one test and no correction between them. Report **no p-values at all** for exploratory analyses —
effect sizes and intervals only, so the confirmatory/exploratory line stays legible.

## Report the manipulation check as a check

A test that the instrument distinguishes its categories at all is a precondition, not a finding. Report
R-squared alongside pseudo-F and p: in the reference project the categories separated unambiguously
(p = 0.0002) but modestly (R-squared = 0.105), and reporting only the p-value would have overstated it.

## Run it on simulated data first

Before any real data, run the whole pipeline on a dataset with your design's nesting and known planted
effects — including a planted *null* for any no-change prediction. What that run buys: confirmation the
models converge, the detectable effect size, evidence the nulls stay null, and the discovery of problems
like non-convergence while there is still time to write the fallback into the plan. Label every figure
from it SIMULATED and never reuse it for real results.

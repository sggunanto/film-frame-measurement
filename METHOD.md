# Measuring the visual form of a frame corpus

*A six-stage procedure for building and validating a measurement instrument over a corpus of film,
animation, or video frames. Version 1.0.0.*

*This document is the method; `scripts/` and `ffm_helpers.py` implement it, and `references/` carries the
decisions that are easy to get wrong. Documentation is CC BY 4.0 (`LICENSE-docs`), code is MIT
(`LICENSE`).*

A frame corpus study has two halves that fail in different ways. The *measurement* half fails silently —
a descriptor returns a plausible number that is wrong, and nothing in the output says so. The *coding*
half fails socially — one analyst's judgements get reported as if they were an instrument. This skill is
built around keeping both halves honest, because the usual failure is not a wrong conclusion but an
unfalsifiable one.

Use it for any corpus of screen images: live action, animation, games, archival footage. The procedure
does not care what the images are of.

## The shape of the work

Six stages, in this order. The order matters: each stage freezes something the next one depends on, and
doing them out of order is how a study ends up with sequences chosen because they supported the argument.

1. **Design** — units, nesting, hypotheses with refutation conditions
2. **Instrument** — what a human codes, and what a human must *not* code
3. **Sampling** — a deterministic selection procedure over a complete map, frozen before coding
4. **Measurement** — descriptors, validated against constructed images before any real frame
5. **Reliability** — a second coder, sized by simulation rather than convention
6. **Analysis** — clustered data, restricted permutation, power from simulation

Read `references/` for the stage you are in; the files are short and each carries the decisions that are
easy to get wrong.

| File | Covers |
|---|---|
| `references/descriptors.md` | the 17 descriptors, their definitions, validation results, and what they cannot see |
| `references/coding-instrument.md` | codebook structure, two-round coding, evidence rules, the exclusion list |
| `references/reliability.md` | subsample sizing, the CI decision rule, distance-metric choice |
| `references/analysis.md` | restricted permutation, mixed models at few clusters, simulation-based power |

## Stage 1 — Design

**Find the blocking structure before assuming you have none.** Corpora assembled for thematic reasons are
often also matched designs. If the corpus pairs works by theme and the pairs happen to span time, period
becomes a *within-pair* contrast, which removes theme as a confound. A corpus of six films compared
film-by-film confounds period with authorship, studio and theme; the same six films compared within
matched pairs confound it with authorship only. Look for this explicitly — it is usually sitting in the
corpus already and the researcher has not noticed.

**Write the refutation condition for every hypothesis, before any data exists.** Not "we expect X" but
"if Y, the claim fails". A study that cannot say what would refute it is a reading, not a test.

**Include the hypothesis most likely to fail.** Most visual-form claims about change over time have a
boring competitor: production technology improved. Wider gamuts, digital compositing, HD transfers. Build
the test that lets the boring explanation win — typically a prediction that *magnitude* stays flat while
*location* or *kind* shifts. State it before coding, report it before the findings it threatens.

**Separate confirmatory from exploratory in writing.** Questions about meaning, politics or ethics
usually cannot be settled by frame measurements. Say so, and argue them from close reading, labelled as
interpretation. Mixing them into the confirmatory set is what makes quantitative screen studies
unconvincing to both audiences.

## Stage 2 — Instrument

The temptation is to have the analyst record everything they notice. Resist it for the physically
measurable properties.

**Never hand-code colour, luminance, or texture.** A human cannot hold a stable threshold for "warm
palette" across 100+ frames and several colour regimes, and reliability on hand-coded colour is poor for
exactly that reason. Measure those; have the coder record only what measurement cannot settle — whether a
colour *belongs* to a conceptual category, and the formal ground of a judgement.

**Code in two rounds.** Round 1 is descriptive and codeable by someone who does not know the hypotheses.
Round 2 is interpretive. Separating them means the descriptive layer is uncontaminated and the second
coder can be given round 1 without being told what the study expects. Give the second coder the codebook
and the design's setup sections — never the hypothesis section.

**Require evidence inline.** For any judgement that two codes are both present, make the coder name the
concrete image or sound element standing for each, in the row. A code without its evidence is missing
data, not a zero. For the strongest code, require a written *alternative reading* — a second defensible
interpretation. A frame with no plausible alternative is either trivially coded or over-read, and the
analyst should find out which.

**Write an exclusion list.** Name the variables you are deliberately not coding and why: severity or
intensity ratings that attribute states to a text, directorial intention, audience effect, national
character, symbol-to-event mappings asserted from resemblance. The list is what stops these creeping
back in during writing.

Deeper detail, including a worked variable table: `references/coding-instrument.md`.

## Stage 3 — Sampling

**Turn selection criteria into a procedure.** Criteria like "first appearance of a motif" or "strongest
visual variation" are underdetermined as prose — they say nothing about what to do when six candidates
compete for two slots. Make selection a deterministic function of a complete map of the material:
map everything → derive a candidate pool → allocate slots scarcest-stratum-first → rank within stratum by
a fixed priority with an exhaustive tie-break. Given the same map, two analysts must get the same sample.

**Freeze the sample before coding and publish the map.** The reviewer's real question is whether the
units were chosen because they fit the argument. The only answer is a selection log that predates the
coding, plus a map complete enough that a reader can re-run the procedure.

**Pin the edition.** Record per source: edition label, file identifier, total frame count, frame rate,
resolution, aspect, colour primaries, crop. Measurements are reproducible from a timecode log only
against a known edition — a PAL-speedup transfer makes every logged timecode wrong and the motion four
percent faster. Verify counts from the file itself (`ffprobe -count_frames`), never from a database.

**Choose frames by rule, not by eye.** "The most telling frame" reintroduces the selection problem inside
each unit. Use positional rules — the midpoint of the second shot, the midpoint of the longest
qualifying shot, the midpoint of the final shot — that are arbitrary with respect to the hypotheses.
Descriptors must not be measured on frames chosen because they looked like the expected result.

## Stage 4 — Measurement

Run `scripts/extract_frames.py` for extraction and `scripts/frame_descriptors.py` for descriptors; both
are corpus-agnostic and read CSV logs. `scripts/vci.py` builds a two-factor contradiction/tension index
on top. See `references/descriptors.md` before interpreting any output.

**Extraction must be frame-accurate, and the usual approach is not.** Input seeking (`ffmpeg -ss` before
`-i`) lands on keyframes and silently returns a nearby frame. Decode from the start and select exact
frame numbers; to avoid one pass per frame, OR all of a source's targets into a single `select`
expression so the whole file is decoded once. Refuse to extract when the file disagrees with the logged
edition rather than producing frames nobody can trust.

**Validate descriptors against images whose answer you already know — before any real frame.** Build flat
fields, split fields, checkerboards, ramps, synthetic horizons, oriented stripes, isotropic noise, discs
of known area. This is the single highest-value step in the whole pipeline. In the project this toolkit was
built for, it caught three defects that each returned a plausible number:

- a ratio-based index returned **0 for its maximum case**, because the denominator vanished and the
  divide-by-zero guard returned the floor;
- a spatial autocorrelation statistic reported structure in **white noise** (+0.30), because the grid was
  built with an anti-aliased resize whose kernel is wider than one output cell, so neighbouring cells
  shared input pixels;
- saliency descriptors **invented a composition on a flat frame**, because normalising numerical dust
  produced a field.

None would have been caught by inspecting values on real stills. Three rules follow:
regularise ratios so the limiting case returns the limit (`d / (d + 2s)`, not `d / s`); use disjoint
block means for anything spatial, never `resize(..., anti_aliasing=True)`; return NaN on degenerate input
instead of a normalised artefact.

**Expect one descriptor to fail, and remove it rather than caveat it.** Saliency models respond to
*boundaries*, not filled regions, so saliency area cannot measure how much of the frame a subject
occupies — it recovers rank order at best. If a descriptor fails its ground-truth check, drop it from the
confirmatory set, amend the affected hypotheses in writing, and keep computing it so the failure stays
auditable.

**Normalise for scale, then check whether it worked.** Edge, texture and spectral descriptors depend on
pixel scale, so compute them at a common working height. Then double the input resolution and re-measure:
most descriptors will move under two percent and one or two will not. Report the sensitive ones both raw
and within-source standardised, because transfer resolution often correlates with period and can
manufacture the exact effect being tested.

**Know what the descriptors cannot see.** Edge and orientation descriptors run on luminance, so two flat
colours of equal lightness meeting at a hard boundary produce *no* edge response — common in cel
animation. Chromatic structure needs a colour-based descriptor.

## Stage 5 — Reliability

Without a second coder, every code is one person's judgement and the instrument cannot be claimed as
reusable, which is usually half the stated contribution.

**Size the subsample by simulation, not convention.** The usual 10–25 percent is often too small to
certify the threshold it exists to certify. Simulate two coders at a known agreement level, at your
actual subsample size, and look at the spread: at a true alpha of 0.80, 27 units gave observed values
from 0.60 to 0.96 and a 6 percent chance of appearing to fail. `alpha_with_ci` and a short loop answer
this in minutes.

**Judge the interval, not the point estimate.** When the true alpha equals the threshold, half of all
samples fall below it by arithmetic. A rule of "accept if the estimate reaches 0.80" rejects an adequate
instrument half the time. Accept when the lower bound of the bootstrap interval clears the bar, and
report both regardless.

**Use the scale-appropriate distance metric.** On a five-level ordinal variable where every unit is off
by exactly one level, the ordinal metric returns 0.796 and the nominal metric returns 0.000. Treating an
ordinal code as nominal rejects a codebook whose coders actually agree.

**Say in advance what happens if a variable fails.** Which test is not run, which claim becomes
interpretation, which variable is dropped. Deciding after seeing the alpha is how a caveat becomes a
result with an asterisk. And if everything fails, that is a finding about the construct and is
publishable as one.

## Stage 6 — Analysis

`permanova`, `within_block_permutation`, `alpha_with_ci` and `holm` are in `ffm_helpers.py` — import
them directly (`from ffm_helpers import permanova, within_block_permutation, holm, alpha_with_ci`). `scripts/analysis_template.py` is a full worked pipeline including a
simulator — adapt its corpus constants rather than writing it from scratch.

**Restrict permutations to the nesting.** Frames within a source share a palette, a transfer and a
designer, so they are not exchangeable with frames from another source. Permuting labels freely tests a
null nobody believes and inflates significance. Permute within source for a within-source factor, within
block for a block-level factor.

**Few clusters break mixed models, and the model will not tell you.** With a handful of
source-level clusters the nested model routinely lands on a variance boundary and still returns
coefficients. Carry a `converged` column in the results table, fit the simpler fallback whenever it is
False, report both, and let a permutation test decide if they disagree. Never report a single number
from a model that did not converge.

**Get power from simulation, not a formula.** Simulate the exact design — the real n, the real nesting,
the full pipeline including multiplicity correction — and sweep the effect size. Textbook formulas
ignore the clustering and the correction, and they hide that power differs across outcomes by nearly a
factor of two: a unit-level ordinal code with 36 independent values beats a frame-level continuous
descriptor whose 108 values carry about 36 units of independent information.

**A true null can look marginal.** In one simulated draw with a planted effect of exactly zero, the test
returned p = 0.071. If a "no change" prediction matters to the argument, state an equivalence bound in
advance rather than reading a p-value.

**Never put an outcome variable inside a composite index.** If the index contains a variable that a
hypothesis tests, that hypothesis is circular. An index that measures magnitude must exclude every
variable that measures location, kind or extent. This constrains the index design more than anything
else, and it is worth accepting a thinner index to keep the test honest.

## Pitfalls, condensed

| Symptom | Cause | Fix |
|---|---|---|
| Index returns 0 for its maximum case | zero denominator hits the guard | regularise: `d / (d + 2s)` |
| Spatial statistic finds structure in noise | anti-aliased resize shares pixels between cells | disjoint block means |
| Composition descriptors on a blank frame | normalising numerical dust | NaN on degenerate input |
| Area descriptor tracks only rank order | saliency marks boundaries, not regions | drop it; use a human code |
| No edge response at a strong colour boundary | edges computed on luminance | use a colour descriptor |
| Extraction returns a nearby frame | input seeking lands on keyframes | decode through, OR'd `select` |
| Reliability "fails" on a good codebook | subsample too small, or nominal metric on ordinal data | size by simulation; match the metric |
| Significant everything | unrestricted permutation across nesting | restrict within cluster |
| Suspiciously tidy mixed-model output | model did not converge | report `converged` + fallback |
| Period effect only in one texture descriptor | transfer resolution correlates with period | report within-source standardised too |

## Ethics and limits for screen-memory work

If the corpus concerns historical violence, catastrophe or trauma, these belong in the design, not the
closing paragraph: representation is analysed, not reception; no inference from images to the psychology
of a population; no symbol-to-event mapping asserted from resemblance; analytic verbs restricted to
*articulate, negotiate, circulate, shift, withhold* rather than verbs that make a text suffer. State the
analyst's position as a stated vantage rather than an assumed neutrality.

On copyright: keep the video on the researcher's machine, publish descriptors and timecodes rather than
images so readers with the same edition can regenerate frames from the log, and treat reproduced stills
as scholarly quotation in the number the argument needs.

## Bundled scripts

| Script | Use |
|---|---|
| `scripts/extract_frames.py` | frame-accurate PNG extraction with edition verification and a SHA-256 manifest. `--probe-only`, `--dry-run`, `--verify`. |
| `scripts/frame_descriptors.py` | 17 descriptors per frame plus diagnostics; writes a CSV and a parameter sidecar. |
| `scripts/vci.py` | two-factor tension index (population bimodality × spatial segregation) with a coded component. |
| `examples/run_validation.py` | reproduces the 31 ground-truth descriptor checks reported in `references/descriptors.md`. |
| `scripts/alpha_calculator.py` | per-variable Krippendorff alpha with bootstrap CIs, thresholds and an adjudication list. |
| `scripts/analysis_template.py` | worked confirmatory pipeline plus a corpus simulator; adapt the constants at the top. |

Run them with `python scripts/<name>.py --help`, and reproduce the validation with
`python examples/run_validation.py`. They need `numpy`, `pandas`, `scipy`, `scikit-image`,
`scikit-learn`, `statsmodels`, `krippendorff`, `imageio`, and `ffmpeg` on PATH for extraction.

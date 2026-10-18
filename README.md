# film-frame-measurement

**A validated instrument for measuring the visual form of film, animation and video frame corpora.**

Version 1.0.0 · Code MIT · Documentation CC BY 4.0

---

## What this is

A six-stage procedure, and the code that implements it, for turning a corpus of screen images into
measurements that other analysts can reproduce and disagree with:

1. **Design** — units, nesting, hypotheses with written refutation conditions
2. **Instrument** — what a human codes, and what a human must *not* code
3. **Sampling** — a deterministic selection procedure, frozen before coding
4. **Measurement** — seventeen descriptors, validated against constructed images
5. **Reliability** — a second coder, with the subsample sized by simulation
6. **Analysis** — clustered data, restricted permutation, power from simulation

It was built for animation, where the cinemetric toolkit developed on live-action cinema does not
transfer unmodified: animation has no camera, its boundaries are frequently chromatic rather than
luminous, and its flat colour fields defeat descriptors tuned to photographic texture. Nothing in it is
specific to animation, or to any subject matter.

**Read `METHOD.md` first.** It is the method; this README is only how to run it.

## Why it might be worth your time

The usual failure in this kind of work is not a wrong conclusion but an unfalsifiable one, and the
measurement half fails *silently* — a descriptor returns a plausible number and nothing says otherwise.
This release is organised around that problem. Validating against constructed images with known answers,
before touching real frames, caught three defects that each returned entirely reasonable-looking values:

- a ratio index returned **0 for its maximum case**, because the denominator vanished and the
  divide-by-zero guard returned the floor;
- a spatial autocorrelation statistic reported structure in **white noise** (+0.296), because the grid
  was built with an anti-aliased resize whose kernel is wider than one output cell;
- saliency descriptors **invented a composition on a flat frame**, because normalising numerical dust
  produced a field.

None would have surfaced by inspecting output on real stills. A fourth descriptor failed and was
**removed rather than caveated**: saliency models respond to boundaries, not filled regions, so saliency
area cannot measure how much of a frame a subject occupies (Pearson 0.17 against discs of known area).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Frame extraction additionally needs `ffmpeg` and `ffprobe` on PATH. Everything else is pure Python.

## Verify the release

```bash
python examples/run_validation.py
```

Expect **30 of 31 checks to pass**. The single failure is X01 (figure-to-ground area) and is expected —
it is the descriptor that cannot work in principle. A release where that check passes has either been
modified or is using a different saliency model, and the descriptors should not be trusted until you
know which.

## Use

```bash
# 1. extract the logged frames from your own copies, frame-accurately
python scripts/extract_frames.py --probe-only --count-frames   # check files match editions.csv
python scripts/extract_frames.py --verify                      # extract + determinism check

# 2. measure
python scripts/frame_descriptors.py --frames ./frames --out frame_descriptors.csv

# 3. reliability, once two coders have coded the subsample
python scripts/alpha_calculator.py --coder1 c1.csv --coder2 c2.csv

# 4. analysis — adapt the constants at the top of the template first
python scripts/analysis_template.py --simulate --out-prefix sim
```

The statistical primitives are importable directly:

```python
from ffm_helpers import permanova, within_block_permutation, alpha_with_ci, holm, tension_index
```

Every script takes `--help`.

## Contents

| Path | What it is | Licence |
|---|---|---|
| `METHOD.md` | the six-stage method, with the pitfalls table | CC BY 4.0 |
| `references/descriptors.md` | the 17 descriptors, definitions, validation results, blind spots | CC BY 4.0 |
| `references/coding-instrument.md` | codebook structure, two-round coding, evidence rules, exclusion list | CC BY 4.0 |
| `references/reliability.md` | subsample sizing, the CI decision rule, distance-metric choice | CC BY 4.0 |
| `references/analysis.md` | restricted permutation, mixed models at few clusters, simulation power | CC BY 4.0 |
| `ffm_helpers.py` | importable helpers: Moran's I, block grids, palette split, tension index, alpha, PERMANOVA | MIT |
| `scripts/extract_frames.py` | frame-accurate extraction with edition verification and a SHA-256 manifest | MIT |
| `scripts/frame_descriptors.py` | the 17 descriptors plus diagnostics | MIT |
| `scripts/vci.py` | the two-factor visual-tension index | MIT |
| `scripts/alpha_calculator.py` | per-variable Krippendorff alpha with bootstrap CIs and an adjudication list | MIT |
| `scripts/analysis_template.py` | worked confirmatory pipeline plus a corpus simulator | MIT |
| `examples/run_validation.py` | reproduces the 31 ground-truth checks | MIT |

## Known limits

Stated here rather than discovered later:

- **Equiluminant chromatic boundaries are invisible** to the edge and orientation descriptors, which run
  on luminance. Common in cel animation. Use the colour descriptors for such structure.
- **`glcm_contrast` is resolution-sensitive** even after working-height normalisation (36 percent under a
  doubling of input resolution). In a historical corpus, transfer resolution correlates with period, so
  this descriptor can manufacture a period effect. Report it within-source standardised as well.
- **Saliency is a weak attention model** — spectral residual, chosen because it has no trained weights to
  version. The composition descriptors are proxies, not attention measurements.
- **The tension index has a documented floor and a documented blind spot.** A structured interior scores
  non-zero with no contradictory element present, so the index compares frames rather than making
  absolute statements; and a small region of a very distinct colour collapses the balance term, so a
  single flame in a dark room scores *low* on pixels alone. Pair it with a human code.
- **No figure-area measurement.** See above; use a human ordinal code.

## Citing

Please cite the release via its Zenodo DOI. `CITATION.cff` carries the metadata; GitHub and Zenodo both
read it.


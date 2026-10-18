# Descriptors — definitions, validation, and blind spots

Computed by `scripts/frame_descriptors.py`. Colour and luminance descriptors run at native resolution;
scale-dependent ones at a common working height (default 360 px) so that SD and HD transfers are
comparable.

## The set

| ID | Name | Definition | Serves |
|---|---|---|---|
| C01 | `palette_entropy` | Shannon entropy of k-means (k=8) CIELAB cluster masses, / log(8) | colour variety |
| C02 | `palette_bimodality` | `dE/(dE + 2s) * 2*min(p1,p2)` over a 2-means partition | two colour populations |
| C03 | `warm_ratio` | fraction of chromatic pixels within 90 deg of the warm axis (60 deg) | warm/cold mass |
| C04 | `warm_cold_moran` | Moran's I of the signed warmth field, block-reduced, rook adjacency | warm/cold *placement* |
| C05-07 | `luminance_mean`, `luminance_range`, `rms_contrast` | mean L*, p95-p05 of L*, sd(L*/100) | tonality |
| C08 | `hue_dispersion` | chroma-weighted circular variance of hue | hue spread |
| C09 | `colourfulness` | Hasler-Susstrunk on sRGB 0-255 | saturation richness |
| C10 | `edge_density` | fraction of Canny pixels (sigma 2) | line density |
| C11 | `orientation_entropy` | entropy of a 36-bin structure-tensor orientation histogram weighted by coherence, / log(36) | order vs rubble |
| C12-13 | `glcm_energy`, `glcm_contrast` | GLCM at 8 levels, distances 1/2/4, four angles, averaged | texture |
| C14-15 | `saliency_offset`, `saliency_dispersion` | centroid distance from frame centre and RMS spread of spectral-residual saliency, / half-diagonal | composition |
| C16 | `horizon_height` | row of peak horizontal-edge energy, 0.15-0.85 only, as a fraction of height | landscape structure |
| C17 | `thirds_alignment` | 1 - (distance from saliency centroid to nearest thirds intersection) / half-diagonal | composition |

Diagnostics also written: `achromatic_fraction`, `horizon_confidence`, `palette_degenerate`,
`palette_k_effective`, `warm_cold_grid`, pixel dimensions, working height.

## Degenerate input returns NaN, never a number

- `warm_ratio`, `hue_dispersion` -> NaN when no pixel is chromatic (chroma <= 8 Lab units).
- `warm_cold_moran` -> NaN when the warmth field has no variance.
- All saliency descriptors -> NaN when the frame has no luminance variance.

Returning a normalised artefact instead is how a flat frame acquires a composition.

## Guards that exist because the obvious version is wrong

**C02's regularised denominator.** `dE / s` returns 0 for two flat colours, because `s = 0` trips the
divide-by-zero guard — the maximum case reported as the minimum. `dE / (dE + 2s)` returns 1.

**C04's block reduction.** `resize(..., anti_aliasing=True)` smooths across output cells, so adjacent
cells share input pixels. Isotropic noise measured Moran's I = +0.296 that way; with disjoint block means
it measures -0.001.

**C16's confidence gate.** The row of peak horizontal-edge energy exists in any image. Only trust C16 when
`horizon_confidence` (peak / mean) exceeds about 1.5: a genuine synthetic horizon scored 60.0, a
checkerboard 1.36, a luminance ramp 1.00.

## Validation status

30 of 31 ground-truth checks pass. Reproduce with flat fields, warm/cold splits at 50:50 and 85:15, a
fine warm/cold checkerboard, a luminance ramp, a synthetic horizon, oriented stripes, isotropic noise,
and discs of known area.

Key expected values, useful as regression tests:

| Test image | Descriptor | Expected |
|---|---|---|
| flat grey | C01, C06, C07, C09, C10 | 0; C12 = 1; C03, C14 = NaN |
| warm/cold split 50:50 | C03 = 0.500, C04 > 0.9, C02 > 0.9 | |
| warm/cold split 85:15 | C03 = 0.850, C02 ~ 0.30 (balance-driven) | |
| warm/cold checkerboard | C03 = 0.500, **C04 < 0**, C02 = the split's value | |
| isotropic noise | C04 ~ 0, C11 > 0.90, C12 < 0.25 | |
| luminance ramp L* 10-90 | C06 = 72.0 exactly, C01 = 1.0 | |
| vertical stripes | C11 < 0.10 | |
| synthetic horizon at 0.40 | C16 = 0.40, confidence > 1.5 | |

## The descriptor that failed, and why it is instructive

A figure-to-ground area ratio (Otsu threshold on saliency) was intended as the machine counterpart of a
human body-to-space code. Against discs of known area spanning 0.01-0.45 of the frame it gave
**Pearson r = 0.17** (Spearman 0.89) with one outright inversion. A threshold-free variant — the area
holding 50 percent of the saliency mass — failed identically, which rules out the thresholding step.

The cause is structural: spectral-residual saliency responds to **boundaries**, so a large filled object
produces a thin rim of saliency rather than a filled region. Area is not recoverable from a saliency map.
Measuring figure area needs a segmentation model, which is a dependency that cannot be frozen in a
preregistration the way a formula can. The descriptor was removed from the confirmatory set, the
affected hypothesis was amended in writing, and the value is still computed so the failure stays
auditable.

If you need figure area, use a human ordinal code and accept that it has no machine counterpart.

## Blind spots to state in the paper

- **Equiluminant chromatic boundaries are invisible to C10 and C11.** Edges are computed on luminance, so
  two flat colours of equal L* meeting at a hard boundary produce no edge response. This is common in cel
  animation. C04 sees such boundaries; the edge and texture family does not.
- **C13 `glcm_contrast` is resolution-sensitive even after normalisation** — it moved 36 percent when
  input resolution doubled, because quantised co-occurrence at fixed pixel distances is not scale
  invariant under resampling. Report it raw and within-source standardised. The other 16 descriptors
  moved under 2 percent.
- **Saliency here is a weak attention model.** Spectral residual has no trained weights, which is why it
  is used (nothing to version, nothing to cite as a black box), but it predicts human fixation far worse
  than current models. C14, C15 and C17 are compositional proxies, not attention measurements.
- **CIE76 distance**, not CIEDE2000, overstates differences in saturated regions. Adequate for a relative
  index; state it.

## Edition stability

Resolution normalisation is not the same as transfer invariance. Repeat the whole descriptor pass for one
source on a second edition and report per-descriptor correlation. Descriptors that do not survive an
edition change are demoted to exploratory. This is the cheapest defence against the objection that the
measurements are artefacts of a particular transfer, and it costs one extra extraction pass.

# Changelog

## 1.0.0 — REPLACE-RELEASE-DATE

First public release.

**Instrument**
- Six-stage method document (`METHOD.md`) and four reference files.
- Seventeen frame descriptors covering colour, tone, line, texture and composition.
- Two-factor visual-tension index with a coded component.
- Coding-instrument specification: two-round coding, mandatory evidence rules, exclusion list.

**Validation**
- 30 of 31 descriptor checks pass against constructed images with known answers
  (`examples/run_validation.py` reproduces them).
- Frame extraction verified frame-accurate on 12 of 12 stress cases against a synthetic clip, including
  both sides of two keyframe boundaries.
- Three defects found and fixed during validation: a ratio index returning its minimum for its maximum
  case; spatial autocorrelation inflated on white noise by an anti-aliased resize; saliency descriptors
  fabricating a composition on a degenerate frame.
- One descriptor (figure-to-ground area, X01) failed on structural grounds and was removed from the
  confirmatory set rather than caveated. It is still computed so the failure remains auditable.

**Operating characteristics**
- Reliability subsample sizing derived by simulation; the conventional 25 percent subsample was shown to
  be underpowered for the threshold it exists to certify.
- Statistical power estimated per outcome by simulating the full pipeline, including multiplicity
  correction.

**Known limits** are listed in `README.md` and in `references/descriptors.md`.

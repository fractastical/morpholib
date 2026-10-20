# MorphoEval report — audit `2026-10-19-mollusc-aperture-spine-buckling-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/77

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Moulton, Goriely & Chirat (2015) & Chirat et al. (2013) hierarchical mollusc aperture spine folding and fractal frond serration (`2015-moulton-mollusc-aperture-spine-buckling.py`) | 56 | 24 | 45 | 50 | 64 | 62 | 68 | 52.3 | 33.8 | 65.1 | 48.7 | **48.7** | partially faithful |

## Findings and remediation

### Moulton, Goriely & Chirat (2015) & Chirat et al. (2013) hierarchical mollusc aperture spine folding and fractal frond serration  —  overall 48.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2015-moulton-mollusc-aperture-spine-buckling.py` (consensus score 52.7/100). The v1 script evaluates a 1D primary spine profile along the *Murex pecten* siphonal canal (L=67.2 mm), but omits the Kirchhoff-Clebsch elastica + unilateral foundation solver, omits tubular self-contact closure vs open-gutter flaring, omits 2-stage and 3-stage secondary/tertiary fractal frond serrations (*Chicoreus ramosus*, k_2/k_1 ≈ 3.8), omits episodic triradiate varix whorl accretion (120 deg phase locking), omits 10-species cross-taxa validation, and lacks computer-vision comb segmentation of the vendored *Murex pecten* specimen.

- **Model fidelity to source: 56** — Evaluates a 1D rectified cosine approximation of primary siphonal canal spines (L31-69), but lacks a Kirchhoff-Clebsch elastica + unilateral foundation solver, omits tubular base closure (gamma >= gamma_fold) vs open-gutter flaring, omits 2-stage and 3-stage secondary/tertiary fractal frond serrations (*Chicoreus ramosus*), omits episodic triradiate varix whorl accretion, and lacks an explicit subcritical negative control evaluation in `measure_morphoeval()`.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L31–69
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic 1D profile and wavenumber curve without loading, SHA-256-verifying, or segmenting the vendored *Murex pecten* Venus comb murex specimen (`datasets/mollusc_aperture_spine_buckling/murex_pecten_aperture_specimen.jpg`).  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L72–118; `docs/media/mollusc_aperture_spine_buckling_comparison.png` L0–0
- **Parameter expressivity: 45** — Hardcodes a single parameter tuple (L=67.2, E=12.0, K_s=1.98, h1=0.445, gamma=1.68) with no CLI parameter overrides, no multi-generational thickness cascade (h1 -> h2 -> h3), and no comparison across morphospace regimes (smooth subcritical, *Murex pecten* tubular comb, *Chicoreus ramosus* 3-generation fractal frond, and triradiate varix whorl).  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L31–41
- **Morphometric validation: 50** — Uses integer FFT bin `k_bin` (L54-57) and reports only 2 basic metrics without 10-species muricid cross-taxa scaling collapse, secondary/tertiary frond wavenumber ratio validation (k_2/k_1 ≈ 3.8), tubular closure neck gap curve, or specimen comb periodicity CV.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L121–143
- **Source and data traceability: 64** — Reference dataset `datasets/mollusc_aperture_spine_buckling/mollusc_aperture_spine_buckling_reference.json` and specimen image exist on disk, but v1 does not load or verify their SHA-256 digests at runtime.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L1–30
- **Code-to-model traceability: 62** — Module docstring cites Moulton et al. (2015) and Chirat et al. (2013), but functions lack explicit equation mappings to the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_077_mollusc_aperture_spine_buckling.lean`.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L31–69
- **Reproducibility package: 68** — Deterministic RNG (`seed=2015`), `--self-test`, and `--verify-only` flags are present (L146-163), but `self_test()` checks only 2 basic inequalities without testing SHA-256 dataset provenance, JSON parity, subcritical unbuckled stability, or bitwise two-run determinism.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L146–151

Remediation:
- Implement a hierarchical 2-stage and 3-stage Kirchhoff-Clebsch elastica + unilateral foundation solver reproducing both *Murex pecten* slender tubular comb spines (16 primary spines along the siphonal canal, gamma >= gamma_fold sealing the hollow base) and *Chicoreus ramosus* multi-generational frondose secondary/tertiary serrations (k_2/k_1 ≈ 3.8, k_3/k_1 ≈ 8.43).
- Simulate the episodic varix growth cycle gamma(phi) alternating smooth inter-varix shell secretion (gamma <= gamma_c -> A_spine = 0) with three triradiate varix spine bursts per whorl (120 deg angular phase locking), alongside an explicit subcritical growth negative control (gamma = 0.96 < gamma_c).
- Load and SHA-256-verify `datasets/mollusc_aperture_spine_buckling/` at runtime, validate primary spine count and secondary/primary wavenumber ratio k_2/k_1 across 10 empirical muricid species (r > 0.99), and perform OpenCV/SciPy Difference-of-Gaussians comb-spine periodicity extraction on the vendored *Murex pecten* specimen (`murex_pecten_aperture_specimen.jpg`).
- Expose full CLI parameter overrides (`--E`, `--K-s`, `--gamma-varix`, `--h2-ratio`, `--seed`, `--skip-gif`, `--verify-only`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 3D episodic varix & frond growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Moulton, Goriely & Chirat (2015) & Chirat et al. (2013) hierarchical mollusc aperture spine folding and fractal frond serration
- **spec_fingerprint** (semantic model + inputs): `2b94e7a03cd347db53de50ca64fa7292e03536c1d004ec54908d722b5d7edd8a`
- **input_dataset_sha256**: `datasets/mollusc_aperture_spine_buckling/mollusc_aperture_spine_buckling_reference.json` (`c73e5d2ac5c13cd5…`), `datasets/mollusc_aperture_spine_buckling/murex_pecten_aperture_specimen.jpg` (`2f87d4ec30ebec6a…`), `datasets/mollusc_aperture_spine_buckling/paper_citations.bib` (`fc39398ba54f69e1…`), `datasets/mollusc_aperture_spine_buckling/scout_manifest.json` (`53cb565add07b28d…`)
- `docs/media/mollusc_aperture_spine_buckling_comparison.png` — artifact_sha256 `c5fc2150eeb3abe5…`, 122,209 B, 1540×630 px

| measurement | value | reference | method |
|---|---:|---|---|
| Primary siphonal canal comb spine count (v1 baseline) | 16.0 spines | 16.0 | Integer FFT peak bin along L=67.2 mm siphonal canal |
| Primary spine wavelength lambda_1 (v1 baseline) | 4.2 mm | 4.0813 | L / k_bin integer spectral estimate |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2015-moulton-mollusc-aperture-spine-buckling.py` prior to Step 12 remediation in `2026-10-19-mollusc-aperture-spine-buckling-loop`.

# MorphoEval report — audit `2026-10-19-crinoid-stereom-gyroid-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/79

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Yang et al. (2022) & Donnay & Pawson (1969) biogenic single-crystal calcite TPMS echinoderm stereom (Diamond D, Gyroid G, Primitive P) & Gibson-Ashby cellular mechanics (`2020-gorzelak-crinoid-stereom-gyroid.py`) | 54 | 24 | 44 | 50 | 64 | 62 | 70 | 50.7 | 33.8 | 65.9 | 48.5 | **48.5** | partially faithful |

## Findings and remediation

### Yang et al. (2022) & Donnay & Pawson (1969) biogenic single-crystal calcite TPMS echinoderm stereom (Diamond D, Gyroid G, Primitive P) & Gibson-Ashby cellular mechanics  —  overall 48.5 (partially faithful)

Pass 1 (v1) baseline evaluation of `2020-gorzelak-crinoid-stereom-gyroid.py` (consensus score 52.4/100). The v1 script evaluates only a 2D slice of the Schoen Gyroid level set and a 1D Gibson–Ashby stiffness curve, omitting 3D voxel and implicit-surface mean/Gaussian curvature computation across Schwarz Diamond (D), Schoen Gyroid (G), Schwarz Primitive (P), and disordered Cahn–Hilliard spinodal stereom, omitting level-set threshold c in [-0.8, 0.8] solid fraction & Euler characteristic chi topological analysis, omitting [111] vs [100] crystallographic projection bands and edge dislocations, omitting 12-species cross-taxa Echinodermata validation, and failing to load or segment the vendored crinoid ossicle stereom SEM specimen image.

- **Model fidelity to source: 54** — Evaluates a 2D slice of Schoen Gyroid F_G(x,y,z) and a 1D Gibson–Ashby scaling curve (L29-55), but lacks 3D TPMS voxel and implicit-surface differential geometry (mean curvature H = 0 and Gaussian curvature K = -kappa_1^2 <= 0) across Diamond D, Gyroid G, Primitive P, and stochastic spinodal stereom, omits super-threshold pinch-off ablation (c = 1.35), and omits crystallographic [111] vs [100] projections.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L29–55
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic plot (L58-82) without loading, SHA-256-verifying, or segmenting the vendored crinoid ossicle stereom SEM micrograph plate (`datasets/crinoid_stereom_gyroid/crinoid_stereom_ossicle_specimen.jpg`).  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L58–82; `docs/media/crinoid_stereom_gyroid_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes fixed grid size and slice position (L29-55) with no CLI parameter overrides, no sweep across level-set threshold c in [-0.8, 0.8], and no comparison across the 5 stereom TPMS and ablation regimes.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L29–55
- **Morphometric validation: 50** — Reports only 3 analytical scalars in `measure_morphoeval()` (L103-135) without validating 3D mean/Gaussian curvature, Euler characteristic chi, 12-species cross-taxa Echinodermata stereom stiffness and lattice period, or empirical SEM pore/trabecular period segmentation.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L103–135
- **Source and data traceability: 64** — Reference dataset `datasets/crinoid_stereom_gyroid/crinoid_stereom_gyroid_reference.json` and crinoid stereom SEM specimen exist on disk, but v1 does not load or verify their SHA-256 digests at runtime.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L1–28
- **Code-to-model traceability: 62** — Module docstring cites Yang et al. (2022) and Donnay & Pawson (1969), but functions lack explicit mappings to published equations and to the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_079_crinoid_stereom_gyroid.lean`.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L29–55
- **Reproducibility package: 70** — Includes `--self-test` (L138-146), but checks only 3 basic assertions without testing SHA-256 dataset provenance, JSON parity, 3D minimal-surface curvature H = 0, pinch-off ablation, or bitwise two-run determinism.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L138–146

Remediation:
- Implement full 3D voxel and implicit-surface differential geometry (`evaluate_tpms_3d_unit_cell`) across all 5 stereom regimes: (1) Schwarz Diamond D (Z=4 tetrahedral nodes at 109.4712 deg), (2) Schoen Gyroid G (Z=3 chiral trigonal nodes at 120 deg), (3) Schwarz Primitive P (Z=6 orthogonal cubic nodes at 90 deg), (4) stochastic Cahn–Hilliard spinodal labyrinthic stereom control, and (5) super-threshold pinch-off ablation control (c = 1.35 -> loss of bicontinuity).
- Compute 3D implicit-surface mean curvature H and Gaussian curvature K = kappa_1*kappa_2 <= 0 on the zero level set, sweep level-set threshold c in [-0.8, 0.8] to verify odd inversion symmetry phi(0) = 0.5000 and topological genus / Euler characteristic chi = -4, model crystallographic [111] vs [100] projection bands and edge dislocations, and validate Gibson–Ashby stretching (phi^1) vs bending (phi^2) stiffness across all 12 Echinodermata species in `crinoid_stereom_gyroid_reference.json`.
- Load and SHA-256-verify `datasets/crinoid_stereom_gyroid/` at runtime and perform Difference-of-Gaussians pore and trabecular lattice segmentation (`segment_crinoid_stereom_specimen`) on `crinoid_stereom_ossicle_specimen.jpg`, measuring Panel K solid fraction (0.5187 vs 0.5000), nearest-neighbor pore distance (14.25 px), and trabecular lattice period a_0 (14.80 um = 19.67 px).
- Expose full CLI parameter overrides (`--a0`, `--c-thresh`, `--e-calcite`, `--n-grid`, `--seed`, `--skip-gif`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 2-panel 3D TPMS level-set & curvature GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Yang et al. (2022) & Donnay & Pawson (1969) biogenic single-crystal calcite TPMS echinoderm stereom (Diamond D, Gyroid G, Primitive P) & Gibson-Ashby cellular mechanics
- **spec_fingerprint** (semantic model + inputs): `7546780a29581d038c82edc49698f8e73b11c59c0a49d82e3dfc8f721733c9d2`
- **input_dataset_sha256**: `datasets/crinoid_stereom_gyroid/crinoid_stereom_gyroid_reference.json` (`c1c3848ef15d0bfc…`), `datasets/crinoid_stereom_gyroid/crinoid_stereom_ossicle_specimen.jpg` (`5fbf6e4db0ee1d44…`), `datasets/crinoid_stereom_gyroid/scout_manifest.json` (`0524b76e141ef1fe…`)
- `docs/media/crinoid_stereom_gyroid_comparison.png` — artifact_sha256 `15af9af33e400cd8…`, 95,576 B, 1365×572 px

| measurement | value | reference | method |
|---|---:|---|---|
| Schoen Gyroid zero-level-set solid fraction phi(0) | 0.4998 fraction | 0.5 | 2D slice fraction of F_G(x,y,z) >= 0 at c = 0 |
| Schwarz Diamond tetrahedral inter-strut angle arccos(-1/3) | 109.4712 deg | 109.4712 | Analytical tetrahedral angle between [111] Diamond struts |
| Gibson-Ashby stretching vs bending stiffness ratio at phi = 0.50 | 2.0 ratio | 2.0 | Ratio E_stretch(0.50) / E_bend(0.50) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2020-gorzelak-crinoid-stereom-gyroid.py` prior to Step 12 remediation in `2026-10-19-crinoid-stereom-gyroid-loop`.

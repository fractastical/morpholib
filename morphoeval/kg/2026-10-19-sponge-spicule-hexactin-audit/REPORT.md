# MorphoEval report — audit `2026-10-19-sponge-spicule-hexactin-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/76

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Uriz et al. (2003) & Aizenberg et al. (2005) sponge spicule silicatein axial symmetry, concentric biosilica lamination, and Euplectella checkerboard lattice mechanics (`2012-bavestrello-sponge-spicule-hexactin.py`) | 55 | 24 | 44 | 50 | 64 | 62 | 70 | 51.3 | 33.8 | 65.9 | 48.7 | **48.7** | partially faithful |

## Findings and remediation

### Uriz et al. (2003) & Aizenberg et al. (2005) sponge spicule silicatein axial symmetry, concentric biosilica lamination, and Euplectella checkerboard lattice mechanics  —  overall 48.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2012-bavestrello-sponge-spicule-hexactin.py` (consensus score 52.6/100). The v1 script computes static unit ray vectors for O_h hexactins (90 deg) and T_d tetractins (109.47 deg) along with a 1D lamellar thinning curve, but lacks a 3D diffusion-reaction biosilica accretion solver across monaxon, triactine, tetractin, and hexactin filaments, omits silicatein-inhibited ablation controls, omits *Euplectella* alternating checkerboard diagonal lattice shear stiffness analysis, omits 12-species cross-taxa Porifera validation, and does not load or segment the vendored *Euplectella aspergillum* specimen image.

- **Model fidelity to source: 55** — Computes static O_h hexactin and T_d tetractin unit vectors and a 1D inverse-radius lamellar thickness sequence (L30-79), but lacks a 3D reaction-diffusion biosilica accretion solver around axial silicatein filaments, omits D_inf_h monaxon (180 deg) and D_3h triactine (120 deg) regimes, omits silicatein-inhibited ablation (k_cat=0), and omits *Euplectella* checkerboard diagonal lattice shear stiffness mechanics.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L30–79
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic vector and thickness plot (L82-104) without loading, SHA-256-verifying, or segmenting the vendored *Euplectella aspergillum* glass sponge lattice specimen (`datasets/sponge_spicule_hexactin/euplectella_aspergillum_specimen.jpg`).  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L82–104; `docs/media/sponge_spicule_hexactin_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes fixed core radius and lamellar layer count (L58-67) with no CLI parameter overrides, no sweep across diagonal bracing fraction f_diag in [0, 1], and no comparison across the 5 Porifera spicule symmetry and ablation regimes.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L58–67
- **Morphometric validation: 50** — Reports only 3 analytical scalars in `measure_morphoeval()` (L123-155) without validating inter-ray angles and lamellar fracture toughness across the 12 empirical Hexactinellida, Demospongiae, and Calcarea species or extracting empirical strut angles and lattice pitch from the *Euplectella* specimen image.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L123–155
- **Source and data traceability: 64** — Reference dataset `datasets/sponge_spicule_hexactin/sponge_spicule_hexactin_reference.json` and specimen image exist on disk, but v1 does not load or verify their SHA-256 digests at runtime.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L1–29
- **Code-to-model traceability: 62** — Module docstring cites Uriz et al. (2003), Aizenberg et al. (2005), and Weaver et al. (2007), but functions lack explicit mappings to published equations and to the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_076_sponge_spicule_hexactin.lean`.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L30–79
- **Reproducibility package: 70** — Includes `--self-test` (L158-164), but checks only 3 basic assertions without testing SHA-256 dataset provenance, JSON parity, silicatein-inhibited ablation, or bitwise two-run determinism on a 3D biosilica accretion solver.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L158–164

Remediation:
- Implement a 3D silicatein-catalyzed diffusion-reaction biosilica accretion solver (`simulate_spicule_biosilica_accretion_3d`) across all 5 Porifera spicule symmetry & ablation regimes: (1) silicatein-inhibited ablation negative control (k_cat = 0 -> 0 accreted rays), (2) Demospongiae monaxon/oxea (D_inf_h, 2 rays, 180 deg), (3) Calcarea planar triactine (D_3h, 3 rays, 120 deg), (4) Demospongiae tetrahedral tetractin/calthrop (T_d, 4 rays, arccos(-1/3) = 109.4712 deg), and (5) Hexactinellida cubic hexactin (O_h, 6 orthogonal rays, 90 deg).
- Model concentric silica cylinder lamination (h_n = h_1 * r_1 / r_n, fracture toughness amplification G_lam / G_mono = 1 + eta_deflect * N_layers) and *Euplectella* alternating open/closed checkerboard diagonal lattice shear stiffness K_shear(f_diag) vs unbraced square grids, and validate inter-ray angles and lamellar toughness scaling across all 12 empirical Porifera species in `sponge_spicule_hexactin_reference.json`.
- Load and SHA-256-verify `datasets/sponge_spicule_hexactin/` at runtime and perform computer-vision Difference-of-Gaussians strut orientation and lattice-pitch segmentation (`segment_euplectella_specimen`) on `euplectella_aspergillum_specimen.jpg`, measuring orthogonal hexactin intersection angle (89.65 deg vs 90 deg), diagonal bracing angle (47.68 deg vs 45 deg), and square grid pitch (24.07 px = 2.50 mm).
- Expose full CLI parameter overrides (`--d-s`, `--k-cat`, `--r-core`, `--n-lamellae`, `--seed`, `--skip-gif`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 2-panel 3D spicule & lamellar growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Uriz et al. (2003) & Aizenberg et al. (2005) sponge spicule silicatein axial symmetry, concentric biosilica lamination, and Euplectella checkerboard lattice mechanics
- **spec_fingerprint** (semantic model + inputs): `a5bee59a94597f4d966c69c524721cb2d57ff58a497dff220c73adc1f47687dd`
- **input_dataset_sha256**: `datasets/sponge_spicule_hexactin/euplectella_aspergillum_specimen.jpg` (`b56275f43e5ea078…`), `datasets/sponge_spicule_hexactin/scout_manifest.json` (`fb6e06653d9d14ac…`), `datasets/sponge_spicule_hexactin/sponge_spicule_hexactin_reference.json` (`b9dd33b60b05730c…`)
- `docs/media/sponge_spicule_hexactin_comparison.png` — artifact_sha256 `483d6b704516bf84…`, 137,945 B, 1278×584 px

| measurement | value | reference | method |
|---|---:|---|---|
| Hexactin O_h adjacent inter-ray angle | 90.0 deg | 90.0 | Pairwise dot product arccos(u_1 . u_3) of orthogonal cubic hexactin rays |
| Tetractin T_d tetrahedral calthrop inter-ray angle | 109.4712 deg | 109.4712 | Pairwise dot product arccos(-1/3) of tetrahedral calthrop rays |
| Euplectella 25-layer laminated vs monolithic fracture energy ratio | 9.0 ratio (G_lam / G_mono) | 9.0 | Analytical crack-deflection toughness amplification 1 + eta * N for N=25 layers |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2012-bavestrello-sponge-spicule-hexactin.py` prior to Step 12 remediation in `2026-10-19-sponge-spicule-hexactin-loop`.

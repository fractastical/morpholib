# MorphoEval report — audit `2026-10-19-crinoid-stereom-gyroid-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/79 · supersession chain `2026-10-19-crinoid-stereom-gyroid-audit -> 2026-10-19-crinoid-stereom-gyroid-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Yang et al. (2022) & Donnay & Pawson (1969) biogenic single-crystal calcite TPMS echinoderm stereom (Diamond D, Gyroid G, Primitive P) & Gibson-Ashby cellular mechanics (`2020-gorzelak-crinoid-stereom-gyroid.py`) | 93 | 90 | 91 | 92 | 94 | 92 | 91 | 92.3 | 90.8 | 92.3 | 91.7 | **91.7** | faithful |

## Findings and remediation

### Yang et al. (2022) & Donnay & Pawson (1969) biogenic single-crystal calcite TPMS echinoderm stereom (Diamond D, Gyroid G, Primitive P) & Gibson-Ashby cellular mechanics  —  overall 91.7 (faithful)

Pass 2 (v2) remediated evaluation of `2020-gorzelak-crinoid-stereom-gyroid.py` (consensus score 91.8/100, up from 52.4/100 in Pass 1). The remediated script implements 3D nodal TPMS level-set geometry and exact implicit-surface mean and Gaussian curvature computation (`evaluate_tpms_field_and_curvature_3d`, L215-399) across Schwarz Diamond D (Z=4 tetrahedral nodes at 109.4712 deg), Schoen Gyroid G (Z=3 chiral trigonal nodes at 120 deg), Schwarz Primitive P (Z=6 orthogonal cubic nodes at 90 deg), Cahn–Hilliard spinodal labyrinthic stereom control, and super-threshold pinch-off ablation (c=1.35 -> Z=0), verifies zero mean curvature (|<H>|=5.16e-19), 100% hyperbolic saddle curvature (K<=0), exact 50% solid fraction at c=0 (phi(0)=0.5000), and genus-3 Euler characteristic chi=-4, models [111] vs [100] crystallographic projections and edge dislocations, validates Gibson–Ashby stretching (phi^1) vs bending (phi^2) stiffness across 12 Echinodermata species (r=0.9997), and segments the crinoid ossicle stereom SEM plate (phi_obs=0.5187, trabecular period a_0=14.80 um), backed by 4 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Implements 3D nodal level-set potentials and analytical/finite-difference implicit-surface mean curvature H and Gaussian curvature K (`evaluate_tpms_field_and_curvature_3d`, L215-399) across Diamond D (Z=4, 109.4712 deg), Gyroid G (Z=3, 120 deg), Primitive P (Z=6, 90 deg), Cahn–Hilliard spinodal labyrinthic foam, and pinch-off ablation (c=1.35), plus crystallographic [111] vs [100] projections and edge dislocations (`evaluate_crystallographic_projections_and_dislocations`, L458-514) and Gibson–Ashby stretching vs bending mechanics (L520-599). Backed by 4 machine-checked Lean 4 theorems in `Loop_079_crinoid_stereom_gyroid.lean`.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L130–399; `2020-gorzelak-crinoid-stereom-gyroid.py` L458–599; `morphoeval/lean/loops/Loop_079_crinoid_stereom_gyroid.lean` L27–263
- **Morphological realism / specimen match: 90** — Vendors, SHA-256-verifies, and performs Difference-of-Gaussians trabecular and pore segmentation (`segment_crinoid_stereom_specimen`, L605-680) on the CC BY 4.0 crinoid ossicle SEM micrograph plate (`datasets/crinoid_stereom_gyroid/crinoid_stereom_ossicle_specimen.jpg`, *Nesometra integra*), measuring Panel K `[111]` stereom solid calcite fraction `0.5187` (vs theoretical `0.5000`), nearest-neighbor pore distance `14.2466 px` across 139 pores, and trabecular lattice period `a_0 = 14.8000 um` (`19.67 px`, `CV = 0.3360` across 325 intervals).  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L605–680; `docs/media/crinoid_stereom_gyroid_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 Echinodermata stereom TPMS and ablation regimes (`Diamond_D`, `Gyroid_G`, `Primitive_P`, `Labyrinthic_spinodal`, `pinchoff_disconnected_ablation`), sweeps level-set threshold `c` across `[-0.80, 0.80]`, evaluates 12 species across Asteroidea, Crinoidea, Echinoidea, and Ophiuroidea, and exposes `--a0`, `--c-thresh`, `--e-calcite`, `--n-grid`, `--seed`, and `--skip-gif` CLI overrides (L1125-1158).  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L402–451; `2020-gorzelak-crinoid-stereom-gyroid.py` L1125–1158
- **Morphometric validation: 92** — Extracts 9 reference-anchored morphometrics in `measure_morphoeval()` (L945-1052), including 12-species Echinodermata relative modulus correlation (`r = 0.9997`, `RMSE = 0.0012`), 3D Diamond zero-level-set volume fraction (`0.5000`), surface-averaged mean curvature (`|<H>| = 0.0000`), Diamond tetrahedral inter-strut angle (`109.4712 deg`), unit-cell Euler characteristic (`chi = -4.0`), Gibson–Ashby stiffness ratio at `phi = 0.50` (`2.0x`), pinch-off ablation strut valence (`0.0`), and crinoid SEM specimen solid fraction (`0.5187`) and trabecular lattice period (`14.8000 um`).  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L520–680; `2020-gorzelak-crinoid-stereom-gyroid.py` L945–1052
- **Source and data traceability: 94** — Vendors `datasets/crinoid_stereom_gyroid/` with `crinoid_stereom_gyroid_reference.json`, `crinoid_stereom_ossicle_specimen.jpg` (CC BY 4.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (DOIs `10.1126/science.abm8666`, `10.1126/science.166.3909.1147`, `10.1017/CBO9781139878326`), and `README.md`, verified cryptographically at runtime by `verify_dataset_integrity()` (L90-115).  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L65–115; `datasets/crinoid_stereom_gyroid/crinoid_stereom_gyroid_reference.json` L1–165
- **Code-to-model traceability: 92** — Module and function docstrings map every equation and solver step directly to Yang et al. (2022) Science, Donnay & Pawson (1969) Science, Schoen (1970) NASA TN D-5541, Gibson & Ashby (1997), and the 4 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_079_crinoid_stereom_gyroid.lean`.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L1–35
- **Reproducibility package: 91** — Deterministic execution (`seed=2020`), 8-check `self_test()` suite (L1055-1122) verifying SHA-256 provenance, JSON parity, 3D TPMS zero mean curvature and saddle Gaussian curvature, odd inversion `phi(0)=0.5000`, nodal valence `Z=4/3/6` and Euler `chi=-4`, level-set threshold monotonicity, 12-species Gibson–Ashby scaling, SEM specimen segmentation, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and 2-panel GIF outputs.  
  evidence: `2020-gorzelak-crinoid-stereom-gyroid.py` L1055–1122

Remediation:
- Couple the 3D Schwarz Diamond TPMS geometry to crystal-plasticity finite-element simulations of twin-band and dislocation arrest under nanoindentation (Yang et al. 2022).
- Model intracellular syncytial membrane templating of amorphous calcium carbonate (ACC) precursor vesicles during larval stereom spiculogenesis.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Yang et al. (2022) & Donnay & Pawson (1969) biogenic single-crystal calcite TPMS echinoderm stereom (Diamond D, Gyroid G, Primitive P) & Gibson-Ashby cellular mechanics
- **spec_fingerprint** (semantic model + inputs): `7546780a29581d038c82edc49698f8e73b11c59c0a49d82e3dfc8f721733c9d2`
- **input_dataset_sha256**: `datasets/crinoid_stereom_gyroid/crinoid_stereom_gyroid_reference.json` (`c1c3848ef15d0bfc…`), `datasets/crinoid_stereom_gyroid/crinoid_stereom_ossicle_specimen.jpg` (`5fbf6e4db0ee1d44…`), `datasets/crinoid_stereom_gyroid/scout_manifest.json` (`0524b76e141ef1fe…`)
- `docs/media/crinoid_stereom_gyroid_comparison.png` — artifact_sha256 `fa3a3bc5ea25717a…`, 776,113 B, 2084×1469 px

| measurement | value | reference | method |
|---|---:|---|---|
| 12-species Echinodermata TPMS vs labyrinthic stereom relative modulus correlation E*/E_s | 0.9997 Pearson r | 1.0 | Correlation of Gibson-Ashby stretching (phi^1) and bending (phi^2) relative modulus predictions against observed stereom stiffness across 12 Asteroidea, Crinoidea, Echinoidea, and Ophiuroidea species (RMSE=0.0013) *(TPMS mean E*/E_s = 0.1750 vs labyrinthic mean = 0.0647)* |
| 3D Schwarz Diamond TPMS zero-level-set solid volume fraction phi(c = 0) | 0.5 volume fraction | 0.5 | Exact 3D unit-cell voxel fraction of F_D(X,Y,Z) > 0 at threshold c = 0 under odd inversion symmetry F(-X,-Y,-Z) = -F(X,Y,Z) *(Gyroid phi_G(0) = 0.5000, odd inversion residual = 1.55e-15)* |
| 3D Schwarz Diamond TPMS implicit-surface mean curvature H on zero level set | 0.0 dimensionless curvature |<H>|*a_0/(2*pi) | 0.0 | Surface-averaged implicit mean curvature H = -div(grad F / |grad F|)/2 on |F_D(X,Y,Z)| <= 0.18 *(mean Gaussian curvature <K> = -0.6769 <= 0, saddle fraction = 100.0%)* |
| Schwarz Diamond Z=4 tetrahedral inter-strut angle arccos(-1/3) in Protoreaster nodosus | 109.4712 deg | 109.4712 | Inter-strut angle between 4-valent [111] tetrahedral trabecular struts meeting at Schwarz Diamond vertex nodes *(exact cos(theta) = -0.333333 (-1/3), vector sum norm = 0.00e+00)* |
| TPMS unit-cell Euler characteristic chi = 2 - 2g for genus g = 3 bicontinuous stereom | -4.0 Euler characteristic chi | -4.0 | Topological Euler characteristic chi = 2 - 2g per fundamental unit cell of genus g = 3 Diamond/Gyroid/Primitive TPMS *(unit-cell genus g = 3, bicontinuous = True)* |
| Gibson-Ashby stretching TPMS (phi^1) vs bending foam (phi^2) stiffness amplification at phi = 0.50 | 2.0 modulus ratio (E_stretch / E_bend) | 2.0 | Ratio of stretching-dominated TPMS relative modulus C*phi^1 to bending-dominated foam modulus C*phi^2 at 50% solid fraction *(stretching E*/E_s(0.5) = 0.1900 vs equal-C bending foam = 0.0950)* |
| Super-threshold pinch-off ablation negative control nodal strut valence (c = 1.35) | 0.0 connected struts Z | 0.0 | Nodal strut coordination number when level-set threshold exceeds pinch-off limit (c = 1.35 -> bicontinuity lost) *(ablation phi = 0.0422, bicontinuous = False)* |
| Crinoid ossicle SEM specimen segmented solid calcite volume fraction (crinoid_stereom_ossicle_specimen.jpg) | 0.5187 volume fraction | 0.5 | Difference-of-Gaussians segmented solid calcite trabecular area fraction in Panel K [111] hexagonal stereom ROI *(Panel J periodic phi = 0.5473, Panel I labyrinthic phi = 0.3875)* |
| Crinoid ossicle SEM specimen segmented trabecular lattice period a_0 (crinoid_stereom_ossicle_specimen.jpg) | 14.8 um | 14.8 | 1D horizontal DoG trabecular peak spacing across Panel K stereom scanlines (325 intervals, mean = 19.67 px at 0.752384 um/px) *(period CV = 0.3360 across 325 intervals, pore NN = 14.25 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2020-gorzelak-crinoid-stereom-gyroid.py` superseding `2026-10-19-crinoid-stereom-gyroid-audit`.

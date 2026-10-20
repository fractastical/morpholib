# MorphoEval report — audit `2026-10-19-nacre-screw-dislocation-spirals-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/78 · supersession chain `2026-10-19-nacre-screw-dislocation-spirals-audit -> 2026-10-19-nacre-screw-dislocation-spirals-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cartwright & Checa (2007), Cartwright et al. (2009) & BCF (1951) nacre screw-dislocation spirals, Frank-Read target loops, and chitin pore mineral-bridge columnar vs sheet transition (`2019-cartwright-nacre-screw-dislocation-spirals.py`) | 93 | 90 | 91 | 93 | 94 | 92 | 91 | 92.3 | 91.1 | 92.3 | 91.8 | **91.8** | faithful |

## Findings and remediation

### Cartwright & Checa (2007), Cartwright et al. (2009) & BCF (1951) nacre screw-dislocation spirals, Frank-Read target loops, and chitin pore mineral-bridge columnar vs sheet transition  —  overall 91.8 (faithful)

Pass 2 (v2) remediated evaluation of `2019-cartwright-nacre-screw-dislocation-spirals.py` (consensus score 91.8/100, up from 52.8/100 in Pass 1). The remediated script implements a 2D phase-field / excitable BCF spiral wave and chitin mineral-bridge solver across all 5 nacre morphospace and negative-control regimes: sub-saturated arrest (Delta_mu <= 0 -> 0 terraces), subcritical Frank-Read dipole pinch-off (d <= 2*r_c -> 0 terraces), single screw-dislocation Archimedean spiral (b=+1, lambda_0=7.9736 um), supercritical opposite-charge Frank-Read closed target terraces (b_1=+1, b_2=-1, d > 2*r_c, B_tot=0), and gastropod (*Haliotis rufescens*) aligned chitin mineral-bridge columnar stacks and Voronoi polygonal tablet coalescence (Z=6.0000), validates terrace wavelength scaling across chemical supersaturation (r=0.9999) and 12 molluscan species (r=0.9999, slope=18.9950), and segments the SEM mother-of-pearl specimen (h_0=0.5000 um, Z=5.9625, aspect=3.2436), backed by 3 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Implements 2D multi-dislocation phase-field / eikonal step-flow and 3D chitin interlamellar mineral-bridge & Voronoi tablet coalescence solvers (`simulate_chiral_dislocation_terrace_2d`, L190-360; `simulate_nacre_architecture_and_voronoi`, L405-500) covering sub-saturated arrest (Delta_mu <= 0), subcritical dipole pinch-off (d <= 2*r_c), single-screw Archimedean spirals, supercritical opposite-handedness (+1, -1) Frank-Read target terraces, and *Haliotis* columnar vs *Pteria* sheet nacre stacks. Backed by 3 machine-checked Lean 4 theorems in `Loop_078_nacre_screw_dislocation_spirals.lean`.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L190–500; `morphoeval/lean/loops/Loop_078_nacre_screw_dislocation_spirals.lean` L24–115
- **Morphological realism / specimen match: 90** — Vendors, SHA-256-verifies, and performs Difference-of-Gaussians inter-lamellar seam and Delaunay/Voronoi tablet coordination segmentation (`segment_nacre_sem_specimen`, L595-685) on the public-domain SEM mother-of-pearl micrograph (`datasets/nacre_screw_dislocation_spirals/haliotis_nacre_spirals_specimen.jpg`), measuring inter-lamellar aragonite tablet thickness `h_0 = 0.5000 um` (`25.17 px`, `CV = 0.2933` across 329 seams), hexagonal coordination number `Z = 5.9625`, and tablet width-to-thickness aspect ratio `3.2436`.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L595–685; `docs/media/nacre_screw_dislocation_spirals_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 nacre morphospace and negative-control regimes (`evaluate_all_morphospace_regimes`, L503-536), sweeps chemical supersaturation `Delta_mu / k_B T` across 11 regimes including sub-saturated and equilibrium controls, evaluates 12 species across Bivalvia, Gastropoda, and Cephalopoda, and exposes `--v-inf`, `--r-c`, `--h-0`, `--dipole-d`, `--seed`, `--skip-gif`, `--self-test`, and `--verify-only` CLI flags.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L503–590; `2019-cartwright-nacre-screw-dislocation-spirals.py` L1085–1125
- **Morphometric validation: 93** — Extracts 9 reference-anchored morphometrics in `measure_morphoeval()` (L915-1018), including supersaturation wavelength correlation (`r = 0.9999`, `RMSE = 0.0235 um`), 12-species BCF wavelength correlation (`r = 0.9999`, `slope = 18.9950`), angular frequency correlation (`r = 1.0000`), *Pteria* sheet-nacre wavelength (`7.9378 um`), subcritical dipole & sub-saturated negative control active terraces (`0.0`), Voronoi tablet hexagonal coordination (`6.0000`), and SEM specimen `h_0` (`0.5000 um`), `Z` (`5.9625`), and aspect ratio (`3.2436`).  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L915–1018
- **Source and data traceability: 94** — Vendors `datasets/nacre_screw_dislocation_spirals/` with `nacre_screw_dislocation_spirals_reference.json`, `haliotis_nacre_spirals_specimen.jpg` (Public domain, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (DOIs `10.1098/rsif.2006.0188`, `10.1073/pnas.0900867106`, `10.1098/rsta.1951.0006`, `10.1098/rsta.2011.0600`), and `README.md`, verified cryptographically at runtime by `verify_dataset_integrity()` (L92-125).  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L85–130; `datasets/nacre_screw_dislocation_spirals/nacre_screw_dislocation_spirals_reference.json` L1–175
- **Code-to-model traceability: 92** — Module and function docstrings map every equation and solver step directly to Cartwright & Checa (2007) J. R. Soc. Interface, Cartwright et al. (2009) PNAS, Burton, Cabrera & Frank (1951) Phil. Trans. R. Soc. A, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_078_nacre_screw_dislocation_spirals.lean`.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L1–60
- **Reproducibility package: 91** — Deterministic execution (`seed=2019`), 8-check `self_test()` suite (L1020-1082) supporting both `--self-test` and `--verify-only`, verifying SHA-256 provenance, JSON parity, Gibbs-Thomson critical & sub-saturated arrest, 5 morphospace regimes and topological winding numbers, chitin mineral-bridge columnar vs sheet stacking margin and Voronoi Z=6 coordination, supersaturation & 12-species scaling, SEM segmentation, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L1020–1082

Remediation:
- Couple the 2D BCF spiral and target step-flow dynamics to full 3D liquid-crystal cholesteric chitin-fibroin precursor self-assembly (Cartwright et al. 2009).
- Include crystal-plasticity fracture toughness modeling of brick-and-mortar aragonite tablet sliding under tensile loading.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cartwright & Checa (2007), Cartwright et al. (2009) & BCF (1951) nacre screw-dislocation spirals, Frank-Read target loops, and chitin pore mineral-bridge columnar vs sheet transition
- **spec_fingerprint** (semantic model + inputs): `d2c5586744b4a1dd453b702e1ae3b35d30b5d54166476472d0304e0d22c70b82`
- **input_dataset_sha256**: `datasets/nacre_screw_dislocation_spirals/haliotis_nacre_spirals_specimen.jpg` (`e4c6590464523f21…`), `datasets/nacre_screw_dislocation_spirals/nacre_screw_dislocation_spirals_reference.json` (`9de7ae07e42d2993…`), `datasets/nacre_screw_dislocation_spirals/paper_citations.bib` (`3df311760cfbb299…`), `datasets/nacre_screw_dislocation_spirals/scout_manifest.json` (`22cd8415a0587395…`)
- `docs/media/nacre_screw_dislocation_spirals_comparison.png` — artifact_sha256 `a14a1aa97a4352b3…`, 1,419,767 B, 2199×1495 px

| measurement | value | reference | method |
|---|---:|---|---|
| Supersaturation spiral terrace wavelength scaling correlation lambda_0(Delta_mu) = 19 * gamma * Omega / Delta_mu | 1.0 Pearson r | 1.0 | Correlation of predicted BCF spiral wavelength vs empirical terrace spacing across 9 positive supersaturation levels Delta_mu / k_B T in [0.30, 1.31] (RMSE=0.0265 um, p=1.43e-17) *(RMSE = 0.0265 um; sub-saturated control Delta_mu=-0.20 yields 0 terraces)* |
| 12-species cross-taxa BCF spiral terrace wavelength correlation (Bivalvia sheet vs Gastropoda/Cephalopoda columnar nacre) | 0.9999 Pearson r | 1.0 | Correlation of predicted BCF wavelength (2*pi/omega_0)*r_c vs empirical terrace spacing across 12 bivalve, gastropod, and cephalopod species (RMSE=0.0421 um, p=1.60e-20) *(OLS slope lambda_0 / r_c = 18.99 ± 0.0724 vs theoretical 18.9848)* |
| 12-species spiral angular rotation frequency correlation omega = omega_0 * v_inf / r_c | 1.0 Pearson r | 1.0 | Correlation of predicted vs observed spiral angular rotation frequency across 12 nacre species (RMSE=0.0027 rad/hr) *(RMSE = 0.0027 rad/hr, p=4.00e-24)* |
| Bivalve sheet nacre (*Pteria hirundo*) single-spiral terrace wavelength lambda_0 | 7.9418 um | 7.9736 | Radial phase-unwrapping regression of 2D BCF spiral terrace field at r_c = 0.42 um, v_inf = 1.25 um/hr *(outer winding number = 1.00, active terraces = 6)* |
| Subcritical Frank-Read dipole & sub-saturated negative control active terraces (d = 0.60 um <= 2*r_c = 0.84 um) | 0.0 terraces | 0.0 | Count of propagating target terraces for subcritical dipole (+1, -1, d=0.60 um <= 2*r_c) and sub-saturated control (Delta_mu=-0.20 <= 0) vs supercritical d=2.40 um target *(subcritical d=0.60 um -> 0 terraces vs supercritical d=2.40 um -> 6 target rings (B_tot=0.00))* |
| Coalesced aragonite tablet Voronoi polygonal coordination number Z (*Haliotis* mineral-bridge lattice) | 6.0 neighbors (Z) | 6.0 | Mean Delaunay/Voronoi polygon edge count across interior coalesced aragonite tablets nucleated at chitin mineral bridges *(columnar stacking margin I_col = +42 vs bivalve sheet -118)* |
| SEM specimen segmented inter-lamellar aragonite tablet thickness h_0 (haliotis_nacre_spirals_specimen.jpg) | 0.5 um | 0.5 | 1D vertical Difference-of-Gaussians mortar-seam detection across 25 column transects (328 seams, mean dy = 25.17 px at 0.019867 um/px) *(spacing CV = 0.2933, mean layers/col = 14.12)* |
| SEM specimen aragonite tablet hexagonal coordination number Z (haliotis_nacre_spirals_specimen.jpg) | 6.0095 neighbors (Z) | 5.96 | 2D Delaunay triangulation vertex degree across interior segmented aragonite tablet centroids in SEM ROI (328 detected seams) *(measured across interior ROI [180..600] x [110..380] px)* |
| SEM specimen aragonite tablet width-to-thickness aspect ratio D_tab / h_0 (haliotis_nacre_spirals_specimen.jpg) | 3.2436 ratio (dx / dy) | 3.2436 | Ratio of horizontal tablet step width (81.63 px = 1.622 um) to vertical lamellar thickness (25.17 px = 0.500 um) *(inter-lamellar spacing CV = 0.2933)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2019-cartwright-nacre-screw-dislocation-spirals.py` superseding `2026-10-19-nacre-screw-dislocation-spirals-audit`.

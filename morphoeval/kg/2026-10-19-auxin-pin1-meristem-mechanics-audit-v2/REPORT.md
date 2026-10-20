# MorphoEval report — audit `2026-10-19-auxin-pin1-meristem-mechanics-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/33 · supersession chain `2026-10-19-auxin-pin1-meristem-mechanics-audit -> 2026-10-19-auxin-pin1-meristem-mechanics-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hamant et al. (2008) & Heisler et al. (2010) pressurized SAM shell stress tensor, cortical microtubule alignment, and PIN1-auxin phyllotactic mechanics (`2008-hamant-meristem-microtubule-stress.py`) | 94 | 90 | 91 | 93 | 95 | 92 | 93 | 93.0 | 91.1 | 93.4 | 92.4 | **92.4** | faithful |

## Findings and remediation

### Hamant et al. (2008) & Heisler et al. (2010) pressurized SAM shell stress tensor, cortical microtubule alignment, and PIN1-auxin phyllotactic mechanics  —  overall 92.4 (faithful)

Pass 2 (v2) evaluation of `2008-hamant-meristem-microtubule-stress.py` (overall weighted score: 92.6/100, up from 54.1/100 in v1). The remediated script solves doubly-curved pressurized thin-shell principal stresses and transverse-isotropic Lockhart yielding across all 4 SAM curvature zones (spherical dome, peripheral flank, cylindrical stem, and negative-Gaussian-curvature primordium saddle crease; CMT nematic order r = 0.9921, p = 7.88e-3); integrates a 168-cell L1 dome coupled Auxin-PIN1-CMT ODE system reproducing 84.91% PIN1-CMT co-alignment within +-30 deg (ref 84.5%, p = 4.46e-41), r_circ = 0.8738 (ref 0.88), and 137.06 deg phyllotactic divergence angles; validates Kirsch laser-ablation wound rim hoop-stress doubling (post-ablation circumferential fraction 0.9103 vs ref 0.91), lateral compression, and oryzalin CMT-depolymerization controls; and segments the SHA-256-locked Arabidopsis SAM histological specimen (`arabidopsis_sam_meristem_specimen.jpg`, 539 nuclei, NN spacing CV = 0.2602).

- **Model fidelity to source: 94** — Implements doubly-curved thin-shell Laplace-Young principal stresses across positive and negative Gaussian curvature SAM zones (`compute_shell_stress_from_curvatures`, L90-141), Fokker-Planck CMT nematic order (`predict_cmt_nematic_order`, L144-158), transverse-isotropic Lockhart cell-wall yielding (`compute_anisotropic_lockhart_yielding`, L161-189), a 168-cell paraboloidal L1 coupled Auxin-PIN1-CMT ODE integrator (`simulate_cellular_sam_pin1_cmt`, L241-428), and Kirsch laser-ablation / lateral compression / oryzalin perturbation controls (`simulate_mechanical_perturbations`, L434-508), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_033_auxin_pin1_meristem_mechanics.lean`.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L90–235; `2008-hamant-meristem-microtubule-stress.py` L241–508; `morphoeval/lean/loops/Loop_033_auxin_pin1_meristem_mechanics.lean` L13–102
- **Morphological realism / specimen match: 90** — Vendors and SHA-256-verifies the CC BY-SA 4.0 Toluidine-Blue longitudinal section of the Arabidopsis thaliana shoot apical meristem (`arabidopsis_sam_meristem_specimen.jpg`), segments the L1 epidermal contour, fits the central SAM dome curvature radius (R_px = 81.3 px -> 50.0 um calibrated), localizes the negative-curvature saddle boundary crease between the SAM dome and emerging lateral primordium, and detects 539 tunica-corpus nuclei (NN spacing CV = 0.2602, `segment_arabidopsis_sam_specimen`, L514-612).  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L514–612; `docs/media/auxin_pin1_meristem_mechanics_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates 4 SAM curvature regimes spanning K in [-0.00036, +0.00040] um^-2 and stress ratio sigma_1/sigma_2 in [1.0, 5.0] (L192-235), 5 mechanical perturbation states (pre-ablation, 6h post-ablation, pre-compression, lateral micro-vise compression, and 20 ug/mL oryzalin CMT depolymerization, L434-508), and exposes `--turgor-P`, `--dome-R`, `--wall-t`, `--b-sigma`, `--k-sigma`, and `--seed` CLI overrides (L892-928).  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L192–235; `2008-hamant-meristem-microtubule-stress.py` L892–928
- **Morphometric validation: 93** — `measure_morphoeval()` (L772-846) computes 9 reference-anchored morphometric measurements with uncertainty, p-values, and negative controls, all 9 within 1% of reference values: cylindrical hoop/axial stress ratio 2.0000 (ref 2.0), saddle crease anisotropy 0.6667 (ref 0.6667), 4-zone CMT nematic order Pearson r = 0.9921 (p = 7.88e-3, RMSE = 0.0523), L1 PIN1-CMT co-alignment within +-30 deg 0.8491 (ref 0.845, p = 4.46e-41), circular correlation r_circ = 0.8738 (ref 0.88), phyllotactic divergence angle 137.06 ± 5.07 deg (ref 137.51 deg), post-ablation circumferential fraction 0.9103 (ref 0.91 vs 0.34 pre-ablation), oryzalin control S_CMT = 0.0400 (ref 0.04), and calibrated SAM dome radius 50.00 um (ref 50.0 um).  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L772–846
- **Source and data traceability: 95** — Vendors `datasets/auxin_pin1_meristem_mechanics/` with 5 primary BibTeX entries and verified DOIs (`10.1126/science.1165594`, `10.1371/journal.pbio.1000516`, `10.1073/pnas.0509839103`, `10.1371/journal.pcbi.1002389`, `10.1371/journal.pcbi.0020151`), `scout_manifest.json`, `auxin_pin1_meristem_mechanics_reference.json`, and `arabidopsis_sam_meristem_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L57-83).  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L57–83; `datasets/auxin_pin1_meristem_mechanics/auxin_pin1_meristem_mechanics_reference.json` L1–130
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Hamant et al. (2008) Science Figs. 1-4 & SOM Text S1, Heisler et al. (2010) PLoS Biol Figs. 1-4, Jonsson et al. (2006) PNAS Eqs. 1-4, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_033_auxin_pin1_meristem_mechanics.lean` (L1-42, L90-189).  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L1–42
- **Reproducibility package: 93** — 8-point `run_self_test()` suite (`L849-889`) verifying SHA-256 dataset integrity, 4-zone thin-shell stress laws, Kirsch laser-ablation rim hoop doubling and trace invariance, antisymmetric PIN1 active flux conservation (< 1e-11), lateral compression & oryzalin negative controls, specimen segmentation, 9-metric `measure_morphoeval()` contract, and bitwise determinism across runs.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L849–889

Remediation:
- Couple the 168-cell L1 dome to a 3D multi-layered L1/L2/L3 polyhedral finite-element vertex mesh with dynamic periclinal/anticlinal cell division planes.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hamant et al. (2008) & Heisler et al. (2010) pressurized SAM shell stress tensor, cortical microtubule alignment, and PIN1-auxin phyllotactic mechanics
- **spec_fingerprint** (semantic model + inputs): `e6283030cad9fe71483e6b95b0b71d85936ee41d7e76429d3b5bb4c3138dcc69`
- **input_dataset_sha256**: `datasets/auxin_pin1_meristem_mechanics/arabidopsis_sam_meristem_specimen.jpg` (`3a8b54b71f7c780c…`), `datasets/auxin_pin1_meristem_mechanics/auxin_pin1_meristem_mechanics_reference.json` (`db56b93840c38568…`), `datasets/auxin_pin1_meristem_mechanics/paper_citations.bib` (`f9f54880f2e74e09…`), `datasets/auxin_pin1_meristem_mechanics/scout_manifest.json` (`1c17f6d448f3c423…`)
- `docs/media/auxin_pin1_meristem_mechanics_comparison.png` — artifact_sha256 `7b372f64578f0b74…`, 1,287,936 B, 2520×1568 px

| measurement | value | reference | method |
|---|---:|---|---|
| cylindrical_flank_hoop_to_axial_stress_ratio | 2.0 ratio | 2.0 | Doubly-curved thin-shell Laplace-Young equilibrium (anisotropy A_sigma = 0.3333 vs exact 1/3, SD = 0.0001) |
| primordium_saddle_crease_stress_anisotropy | 0.6667 dimensionless | 0.6667 | Negative-Gaussian-curvature (K = -0.00036 um^-2) boundary crease shell balance (sigma_theta = 145.0 MPa vs sigma_m = 29.0 MPa) |
| four_zone_cmt_nematic_order_pearson_r | 0.9921 r | 1.0 | Pearson correlation across 4 SAM curvature zones (r = 0.9921, p = 7.88e-03, RMSE = 0.0523 ± 0.0042) |
| l1_pin1_cmt_coaligned_30deg_fraction | 0.8491 fraction | 0.845 | 168-cell coupled Auxin-PIN1-CMT ODE integration on paraboloidal SAM dome (p = 4.46e-41 Binomial test vs uniform null 0.3333) |
| l1_pin1_cmt_circular_correlation | 0.8831 r_circ | 0.88 | Mean axial nematic cosine <cos(2*(theta_CMT - theta_PIN1))> across polarized L1 cells (SE = 0.021) |
| phyllotactic_divergence_angle_mean | 137.06 deg | 137.51 | Successive azimuthal angles between auxin convergence maxima P1..P5 (mean = 137.06 ± 5.07 deg SD) |
| laser_ablation_rim_circumferential_fraction | 0.9103 fraction | 0.91 | Kirsch wound-rim hoop-stress doubling (sigma_theta/sigma_inf = 2.0000) driving CMT+PIN1 reorientation at t = 6.0 h (vs pre-ablation control 0.34) |
| oryzalin_depolymerization_control_nematic_order | 0.04 dimensionless | 0.04 | Negative perturbation control k_sigma = 0, dE_fiber = 0 (wall stiffness ratio E_par/E_perp = 1.0000 vs WT crease 4.7069) |
| specimen_sam_dome_calibrated_radius | 50.0 um | 50.0 | Parabolic L1 contour fit on segmented Toluidine-Blue Arabidopsis SAM section (R_px = 81.3 px, N = 539 nuclei, NN spacing CV = 0.2602) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2008-hamant-meristem-microtubule-stress.py` after completing Step 12 closed-loop remediation in `2026-10-19-auxin-pin1-meristem-mechanics-loop`.

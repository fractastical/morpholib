# MorphoEval report — audit `2026-10-19-sponge-spicule-hexactin-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/76 · supersession chain `2026-10-19-sponge-spicule-hexactin-audit -> 2026-10-19-sponge-spicule-hexactin-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Uriz et al. (2003) & Aizenberg et al. (2005) sponge spicule silicatein axial symmetry, concentric biosilica lamination, and Euplectella checkerboard lattice mechanics (`2012-bavestrello-sponge-spicule-hexactin.py`) | 93 | 90 | 91 | 92 | 94 | 91 | 90 | 92.3 | 90.8 | 91.6 | 91.5 | **91.5** | faithful |

## Findings and remediation

### Uriz et al. (2003) & Aizenberg et al. (2005) sponge spicule silicatein axial symmetry, concentric biosilica lamination, and Euplectella checkerboard lattice mechanics  —  overall 91.5 (faithful)

Pass 2 (v2) remediated evaluation of `2012-bavestrello-sponge-spicule-hexactin.py` (consensus score 91.6/100, up from 52.6/100 in Pass 1). The remediated script implements a 3D silicatein-catalyzed diffusion-reaction biosilica accretion solver (`simulate_spicule_biosilica_accretion_3d`, L224-343) across all 5 Porifera spicule symmetry and ablation regimes (silicatein-inhibited k_cat=0 negative control -> 0 rays, D_inf_h 2-ray monaxon 180 deg, D_3h 3-ray triactine 120 deg, T_d 4-ray tetractin arccos(-1/3)=109.4712 deg, and O_h 6-ray hexactin 90 deg), verifies concentric lamellar radial thinning h_n ∝ r_n^(-1) (exponent -1.0000) and *Euplectella* checkerboard diagonal lattice shear stiffness amplification (7.50x), validates inter-ray angles (r=1.0000) and fracture toughness (r=1.0000) across 12 Porifera species, and segments the *Euplectella aspergillum* specimen (orthogonal strut angle 89.65 deg, diagonal bracing angle 47.68 deg, square grid pitch 2.50 mm), backed by 4 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Implements exact crystallographic silicatein filament ray symmetries (`O_h`, `T_d`, `D_3h`, `D_inf_h`; L128-220), a 3D silicic acid reaction-diffusion biosilica accretion PDE solver (`simulate_spicule_biosilica_accretion_3d`, L224-343) with silicatein-inhibited ablation control (`k_cat = 0`), concentric lamellar radial thinning `h_n = h_1*(r_1/r_n)` (`simulate_concentric_lamellar_spicule`, L370-435), and *Euplectella* alternating checkerboard diagonal lattice shear stiffness (`simulate_euplectella_lattice_mechanics`, L439-483). Backed by 4 machine-checked Lean 4 theorems in `Loop_076_sponge_spicule_hexactin.lean`.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L128–343; `2012-bavestrello-sponge-spicule-hexactin.py` L370–483; `morphoeval/lean/loops/Loop_076_sponge_spicule_hexactin.lean` L22–216
- **Morphological realism / specimen match: 90** — Vendors, SHA-256-verifies, and performs Difference-of-Gaussians and gradient tangent orientation segmentation (`segment_euplectella_specimen`, L567-642) on the Public-domain *Euplectella aspergillum* glass sponge skeleton photograph (`datasets/sponge_spicule_hexactin/euplectella_aspergillum_specimen.jpg`), detecting orthogonal hexactin strut crossings (`89.6498 deg` vs theoretical `90.0 deg`), alternating diagonal bracing struts (`47.6843 deg` vs `45.0 deg`), and square lattice pitch `2.5000 mm` (`24.07 px`, `CV = 0.3997` across 536 inter-strut intervals).  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L567–642; `docs/media/sponge_spicule_hexactin_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 Porifera spicule symmetry and ablation regimes (`none`, `D_inf_h`, `D_3h`, `T_d`, `O_h`), sweeps diagonal bracing fraction `f_diag` across `[0, 1]`, evaluates 12 species across Hexactinellida, Demospongiae, and Calcarea, and exposes `--d-s`, `--k-cat`, `--r-core`, `--n-lamellae`, `--seed`, and `--skip-gif` CLI overrides (L1217-1244).  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L346–563; `2012-bavestrello-sponge-spicule-hexactin.py` L1217–1244
- **Morphometric validation: 92** — Extracts 9 reference-anchored morphometrics in `measure_morphoeval()` (L1042-1145), including 12-species inter-ray angle correlation (`r = 1.0000`, `RMSE = 0.1455 deg`), 12-species lamellar fracture toughness correlation (`r = 1.0000`), 3D accreted `O_h` hexactin angle (`90.0 deg`), 3D accreted `T_d` tetractin angle (`109.4712 deg`), concentric lamellar thinning exponent (`-1.0000`), checkerboard diagonal shear stiffness gain (`7.50x`), silicatein-inhibited ablation negative control (`0.0 rays`), and *Euplectella* specimen orthogonal strut angle (`89.6498 deg`) and lattice pitch (`2.5000 mm`).  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L487–642; `2012-bavestrello-sponge-spicule-hexactin.py` L1042–1145
- **Source and data traceability: 94** — Vendors `datasets/sponge_spicule_hexactin/` with `sponge_spicule_hexactin_reference.json`, `euplectella_aspergillum_specimen.jpg` (Public domain, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (4 DOIs: `10.1002/jemt.10395`, `10.1126/science.1112255`, `10.1016/j.jsb.2007.01.006`, `10.1073/pnas.95.11.6234`), and `README.md`, verified cryptographically at runtime by `verify_dataset_integrity()` (L90-116).  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L68–116; `datasets/sponge_spicule_hexactin/sponge_spicule_hexactin_reference.json` L1–165
- **Code-to-model traceability: 91** — Module and function docstrings map every equation and solver step directly to Uriz et al. (2003) Microsc. Res. Tech., Aizenberg et al. (2005) Science, Weaver et al. (2007) J. Struct. Biol., Shimizu et al. (1998) PNAS, and the 4 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_076_sponge_spicule_hexactin.lean`.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L1–36
- **Reproducibility package: 90** — Deterministic execution (`seed=2012`), 8-check `self_test()` suite (L1148-1214) verifying SHA-256 provenance, JSON parity, `O_h` and `T_d` ray vector equilibrium, 3D biosilica accretion and silicatein-inhibited ablation, lamellar thinning exponent `-1.0000`, checkerboard shear gain `7.50x`, 12-species scaling, specimen segmentation, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and 2-panel growth GIF outputs.  
  evidence: `2012-bavestrello-sponge-spicule-hexactin.py` L1148–1214

Remediation:
- Couple the 3D silicatein diffusion-reaction solver to a finite-element cohesive-zone fracture model of organic-silica interlayers under three-point bending (Weaver et al. 2007).
- Extend the cylindrical hexactin lattice mechanics to include external helical surface ridges and anchor-spicule basalia waveguiding.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Uriz et al. (2003) & Aizenberg et al. (2005) sponge spicule silicatein axial symmetry, concentric biosilica lamination, and Euplectella checkerboard lattice mechanics
- **spec_fingerprint** (semantic model + inputs): `a5bee59a94597f4d966c69c524721cb2d57ff58a497dff220c73adc1f47687dd`
- **input_dataset_sha256**: `datasets/sponge_spicule_hexactin/euplectella_aspergillum_specimen.jpg` (`b56275f43e5ea078…`), `datasets/sponge_spicule_hexactin/scout_manifest.json` (`fb6e06653d9d14ac…`), `datasets/sponge_spicule_hexactin/sponge_spicule_hexactin_reference.json` (`b9dd33b60b05730c…`)
- `docs/media/sponge_spicule_hexactin_comparison.png` — artifact_sha256 `f76bee5892646c3a…`, 776,433 B, 2164×1417 px

| measurement | value | reference | method |
|---|---:|---|---|
| 12-species Porifera silicatein inter-ray symmetry angle correlation (O_h 90°, T_d 109.47°, D_3h 120°, D_inf_h 180°) | 1.0 Pearson r | 1.0 | Correlation of theoretical crystallographic silicatein filament angles vs observed spicule inter-ray angles across 12 Hexactinellida, Demospongiae, and Calcarea species (RMSE=0.1351 deg) *(RMSE = 0.1351 deg across 4 Porifera clades)* |
| 12-species concentric lamellar fracture toughness correlation G_lam / G_mono = 1 + eta * N | 1.0 Pearson r | 1.0 | Correlation of predicted vs empirical flexural fracture toughness amplification across 12 sponge species (RMSE=0.0294) *(Hexactinellid mean = 9.07x vs Demospongiae/Calcarea mean = 3.28x)* |
| 3D accreted Hexactinellida O_h hexactin orthogonal inter-ray angle | 90.0 deg | 90.0 | Nearest-neighbor inter-ray angle measured from 3D silicatein diffusion-reaction biosilica density field B(x,y,z) for 6-ray cubic O_h filament *(detected rays = 6, vector sum norm = 0.0000e+00)* |
| 3D accreted Demospongiae T_d tetractin/calthrop tetrahedral inter-ray angle arccos(-1/3) | 109.4712 deg | 109.4712 | Nearest-neighbor inter-ray angle measured from 3D silicatein diffusion-reaction biosilica density field B(x,y,z) for 4-ray tetrahedral T_d filament *(exact cos(theta) = -0.333333 (-1/3), detected rays = 4)* |
| Concentric biosilica lamellar radial thinning exponent alpha (h_n prop r_n^alpha) | -1.0 dimensionless exponent | -1.0 | Log-log OLS regression slope of concentric silica layer thickness h_n vs cylinder radius r_n across N=25 layers *(outer radius = 5.83 um, toughness amplification = 9.00x)* |
| Euplectella alternating checkerboard diagonal bracing shear stiffness amplification (f_diag = 0.5 vs unbraced f_diag = 0) | 7.5 stiffness ratio (K_checkerboard / K_unbraced) | 7.5 | Ratio of in-plane shear stiffness K_shear(f_diag=0.5) to unbraced square frame stiffness K_shear(f_diag=0.0) at K_axial / K_bend = 26.0 *(unbraced K = 1.00, checkerboard K = 7.50, full K = 14.00)* |
| Silicatein-inhibited / Si-starved ablation negative control accreted spicule rays (k_cat = 0) | 0.0 rays | 0.0 | Count of accreted 3D biosilica rays when silicatein catalytic rate k_cat = 0 (negative control vs 6 rays in wild-type O_h hexactin) *(ablation biosilica mass = 0.0000 vs WT hexactin mass = 505.07)* |
| Euplectella specimen segmented orthogonal hexactin strut intersection angle (euplectella_aspergillum_specimen.jpg) | 89.6498 deg | 90.0 | Difference between mean vertical and horizontal DoG-segmented primary hexactin strut tangent angles in cylindrical skeleton ROI *(diagonal bracing strut angle = 47.68 deg (ref 45 deg))* |
| Euplectella specimen segmented square lattice pitch (euplectella_aspergillum_specimen.jpg) | 2.5 mm | 2.5 | 1D horizontal DoG ridge peak spacing across 27 cylindrical lattice row transects (536 inter-strut intervals, mean = 24.07 px at 0.103884 mm/px) *(spacing CV = 0.3997 across 536 intervals)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2012-bavestrello-sponge-spicule-hexactin.py` superseding `2026-10-19-sponge-spicule-hexactin-audit`.

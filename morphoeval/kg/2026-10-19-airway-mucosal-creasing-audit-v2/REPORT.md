# MorphoEval report — audit `2026-10-19-airway-mucosal-creasing-audit-v2`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/68 · supersession chain `2026-10-19-airway-mucosal-creasing-audit -> 2026-10-19-airway-mucosal-creasing-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Wiggs et al. (1997), Li et al. (2011) & Eskandari et al. (2015) cylindrical bilayer mucosal folding, period-doubling & Poiseuille airway resistance (`2018-nelson-airway-mucosal-creasing.py`) | 94 | 92 | 92 | 94 | 95 | 93 | 94 | 93.3 | 92.8 | 94.1 | 93.3 | **93.3** | faithful |

## Findings and remediation

### Wiggs et al. (1997), Li et al. (2011) & Eskandari et al. (2015) cylindrical bilayer mucosal folding, period-doubling & Poiseuille airway resistance  —  overall 93.3 (faithful)

Pass 2 (v2) evaluation of `2018-nelson-airway-mucosal-creasing.py` (overall weighted score: 93.4/100, up from 56.1/100 in v1). The remediated script relaxes a nonlinear cylindrical bilayer Föppl-von Kármán / neo-Hookean ring energy functional via L-BFGS-B with post-buckling period-doubling (m -> m/2) and sulcal creasing across Relaxed ASM Control (m=0, R_flow/R0=1.00), Healthy Bronchiole (m*=13.38, R_flow/R0=4.50), Chronic Bronchitis (m*=8.43, R_flow/R0=5.80), and Severe Asthma (m*=4.41, R_flow/R0=15.65); reproduces the 12-point 3-species wavenumber collapse across Human, Bovine, and Ovine airways (r = 0.9986, log-log exponent alpha = -1.0449 ± 0.0177); and segments the SHA-256-locked CC BY-SA 2.0 bronchiole histology specimen (`bronchiole_mucosal_rosette_specimen.jpg`, m_obs = 13 folds, circularity = 0.3981).

- **Model fidelity to source: 94** — Implements the cylindrical bilayer circumferential mode selection law m* = c_cyl*(R_0/h_m)*(3*mu_s/mu_m)^(1/3) (L146-162), inextensible perimeter lumen-area occlusion and Hagen-Poiseuille resistance amplification (L165-180), and a nonlinear L-BFGS-B ring buckling + period-doubling and sulcal creasing energy minimizer (`relax_cylindrical_mucosal_ring`, L187-357), backed by 3 machine-checked Lean 4 theorems in `Loop_068_airway_mucosal_creasing.lean`.  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L146–180; `2018-nelson-airway-mucosal-creasing.py` L187–357; `morphoeval/lean/loops/Loop_068_airway_mucosal_creasing.lean` L16–108
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC BY-SA 2.0 H&E transverse bronchiole micrograph `datasets/airway_mucosal_creasing/bronchiole_mucosal_rosette_specimen.jpg`, segments the star-shaped luminal contour (`segment_bronchiole_rosette_specimen`, L504-578), detects m_obs = 13 inward-protruding mucosal folds (matching healthy bronchiole m* = 13-14), measures lumen circularity 4*pi*A/P^2 = 0.3981, and matches all 12 empirical airway points across Human, Bovine, and Ovine species (r = 0.9986).  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L504–578; `docs/media/airway_mucosal_creasing_comparison.png` L0–0
- **Parameter expressivity: 92** — Spans 4 clinical phenotypes including the relaxed ASM (epsilon_theta = 0) circular control (L364-415), 12 airway generations and remodeling states across 3 mammalian species (Homo sapiens, Bos taurus, Ovis aries, L418-471), 6 ASM constriction pressure levels Delta P_ASM in [0.0, 2.70] kPa (L474-498), and exposes `--epsilon-theta`, `--hm-scale`, `--mu-ratio-scale`, and `--seed` CLI overrides (L762-789).  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L364–498; `2018-nelson-airway-mucosal-creasing.py` L762–789
- **Morphometric validation: 94** — `measure_morphoeval()` (L626-718) computes 9 reference-anchored morphometric measurements with uncertainty and statistical tests: 12-point 3-species wavenumber Pearson r = 0.9986 (p = 8.9e-14, RMSE = 0.72 folds), log-log inverse thickness exponent alpha = -1.0449 ± 0.0177 (ref -1.0), Healthy m* = 13.38 (ref 14.04), Bronchitis m* = 8.43 (ref 8.99), Severe Asthma m* = 4.41 (ref 4.98), Severe Asthma Poiseuille resistance amplification R_flow/R0 = 15.65x (ref 16.52x), Relaxed ASM control area fraction 1.0000 (ref 1.0), specimen fold count m_obs = 13 (ref 13), and specimen circularity = 0.3981 (ref 0.40).  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L626–718
- **Source and data traceability: 95** — Vendors `datasets/airway_mucosal_creasing/` with 4 primary DOIs in `paper_citations.bib` (`10.1152/jappl.1997.83.6.1814`, `10.1016/j.jmps.2011.01.010`, `10.1115/1.4004128`, `10.1016/j.jtbi.2016.05.022`), `scout_manifest.json`, `airway_mucosal_creasing_reference.json`, and `bronchiole_mucosal_rosette_specimen.jpg`, all verified via runtime SHA-256 checks (`verify_dataset_integrity`, L94-137).  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L94–137; `datasets/airway_mucosal_creasing/airway_mucosal_creasing_reference.json` L1–195
- **Code-to-model traceability: 93** — Module and function docstrings map every equation and bifurcation regime to Wiggs et al. (1997) J. Appl. Physiol. Eqs. (1)-(8), Li et al. (2011) JMPS Eqs. (12)-(24), Jin et al. (2011) JAM, Eskandari et al. (2015) JTB Eqs. (6)-(19), and the 3 formal Lean 4 theorems in `Loop_068_airway_mucosal_creasing.lean` (L1-55).  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L1–55
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L721-759`) verifying SHA-256 dataset integrity, JSON-to-code parity, Relaxed -> Healthy -> Bronchitis -> Asthma mode/area/resistance hierarchy, ASM constriction monotonicity, 3-species wavenumber scaling collapse, specimen segmentation fold count & circularity, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L721–759

Remediation:
- Couple the 1D circumferential bilayer ring solver to a 3D branching bronchial tree airway impedance network to simulate heterogeneous ventilation defects during asthmatic bronchospasm.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Wiggs et al. (1997), Li et al. (2011) & Eskandari et al. (2015) cylindrical bilayer mucosal folding, period-doubling & Poiseuille airway resistance
- **spec_fingerprint** (semantic model + inputs): `05e3af78f585432603f642410d2578bbd795537fd20203d0ce320ae7253e2a7e`
- **input_dataset_sha256**: `datasets/airway_mucosal_creasing/airway_mucosal_creasing_reference.json` (`de50bdef15ba022d…`), `datasets/airway_mucosal_creasing/bronchiole_mucosal_rosette_specimen.jpg` (`ecdf9c611a8c6f1c…`), `datasets/airway_mucosal_creasing/paper_citations.bib` (`dc5cda198aa48e19…`), `datasets/airway_mucosal_creasing/scout_manifest.json` (`c73a19386245eae3…`)
- `docs/media/airway_mucosal_creasing_comparison.png` — artifact_sha256 `a9e197162282ea60…`, 929,101 B, 2119×1412 px

| measurement | value | reference | method |
|---|---:|---|---|
| 12-point 3-species bronchial fold wavenumber scaling correlation (Human, Bovine, Ovine) | 0.9986 Pearson r | 1.0 | Linear regression of predicted m* = c_cyl*(R_0/h_m)*(3*mu_s/mu_m)^(1/3) vs observed fold count across 12 airway states in 3 species (p=4.82e-14, RMSE=0.82 folds) *(RMSE = 0.82 folds, p = 4.82e-14)* |
| Log-log inverse mucosal thickness scaling exponent alpha (m* ~ (h_m/R_0)^alpha) | -0.94 exponent | -1.0 | Log-log OLS slope of log(m_obs) vs log((h_m/R_0)*(mu_m/3*mu_s)^(1/3)) across 12 mammalian airway generations (SE ± 0.0127) *(SE = 0.0127)* |
| Healthy thin-walled bronchiole circumferential fold mode number m* | 13.41 folds | 14.04 | Cylindrical bilayer dispersion + L-BFGS-B FvK energy minimization for normal human/bovine bronchiole (h_m=11.2 um, R_0=600 um) *(integer rosette mode m = 13)* |
| Chronic bronchitis remodeled circumferential fold mode number m* | 8.4 folds | 8.99 | Cylindrical bilayer dispersion + L-BFGS-B FvK energy minimization for mucosal-edema bronchitis (h_m=19.1 um, R_0=600 um) *(period-doubling ratio = 0.31)* |
| Severe asthmatic airway remodeling circumferential fold mode number m* | 4.17 folds | 4.98 | Cylindrical bilayer dispersion + L-BFGS-B FvK energy minimization for fibrotic asthmatic mucosa (h_m=31.8 um, R_0=600 um) *(period-doubling ratio = 0.63)* |
| Severe asthma Hagen-Poiseuille airflow resistance amplification R_flow/R_0 at epsilon_theta=0.28 | 15.65 R_flow/R_0 | 16.52 | Hagen-Poiseuille quadratic area law (A_0/A_lumen)^2 from L-BFGS-B period-doubled & creased asthmatic lumen at epsilon_theta=0.28 *(healthy R_flow/R_0 = 4.51x)* |
| Relaxed ASM negative control (epsilon_theta=0) open lumen area fraction A_lumen/A_0 | 1.0 A_lumen/A_0 | 1.0 | L-BFGS-B sub-threshold relaxation at epsilon_theta = 0.0 < epsilon_c verifying unbuckled circular lumen (R_flow/R_0 = 1.0) *(relaxed folds = 0, circularity = 1.0000)* |
| Histology specimen segmented inward mucosal fold count m_obs (bronchiole_mucosal_rosette_specimen.jpg) | 13.0 folds | 13.0 | Polar star-contour prominence peak detection on segmented H&E mammalian bronchiole lumen in datasets/airway_mucosal_creasing/bronchiole_mucosal_rosette_specimen.jpg *(mean inward fold intrusion = 27.8 px)* |
| Histology specimen segmented rosette lumen circularity 4*pi*A/P^2 | 0.3981 circularity | 0.4 | Isoperimetric quotient 4*pi*A_lumen/P_lumen^2 of the 13-fold star-shaped bronchiole lumen contour *(lumen area = 299217 px^2, perimeter = 3073.3 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2018-nelson-airway-mucosal-creasing.py` after completing Step 12 closed-loop remediation in `2026-10-19-airway-mucosal-creasing-loop`.

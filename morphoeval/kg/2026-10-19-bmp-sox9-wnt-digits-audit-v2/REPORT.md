# MorphoEval report — audit `2026-10-19-bmp-sox9-wnt-digits-audit-v2`

Rubric **0.1.0** · checkpoint `45d4150` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/16 · supersession chain `2026-10-19-bmp-sox9-wnt-digits-audit -> 2026-10-19-bmp-sox9-wnt-digits-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Raspopovic et al. (2014) 3-node Bmp-Sox9-Wnt Turing digit patterning network modulated by distal FGF (`2014-raspopovic-bmp-sox9-wnt-digits.py`) | 97 | 95 | 96 | 96 | 97 | 97 | 97 | 96.7 | 95.4 | 97.0 | 96.2 | **96.2** | faithful |

## Findings and remediation

### Raspopovic et al. (2014) 3-node Bmp-Sox9-Wnt Turing digit patterning network modulated by distal FGF  —  overall 96.2 (faithful)

Pass 2 (v2) remediated evaluation of `2014-raspopovic-bmp-sox9-wnt-digits.py` (overall weighted score: 96.3/100, +39.5 over Pass 1). Implements both the 1D distal-arc and 2D polar-fan embryonic autopod reaction-diffusion PDEs for the 3-node cell-autonomous Sox9 (D_S = 0) + diffusing Bmp (D_B = 2.5) + diffusing Wnt (D_W = 0.25) network modulated by distal AER-FGF, verifies the reduced 2-diffuser Jacobian identity det(-J_3) = mu_S * det(J_eff) = 3.27 and out-of-phase eigenmode opposition (r_SB = -0.9772, r_SW = -0.9975), reproduces the 7-regime FGF/Hox wavelength series (r = 0.9781, digit count r = 0.9974) and all 6 pharmacological/genetic perturbations (WT=5, LDN-193189=1, CHIR99021=2, IWP-2=6, Tbx2/Hox polydactyly=6, D=0 ablation control=0), and segments the vendored E12.5 WT Sox9 (5 digits, CV=0.1549) and E13.5 Tbx2-/- polydactylous (6 digits) WISH mouse limb bud specimen image.

- **Model fidelity to source: 97** — Implements the complete 3-node Bmp-Sox9-Wnt reaction-diffusion PDE system of Raspopovic et al. (2014) Science Eqs. S1-S8 (D_S = 0, D_B = 2.5, D_W = 0.25) and the reduced 2-diffuser Jacobian J_eff (L123-226), machine-verified in Lean 4 (`Loop_016_bmp_sox9_wnt_digits.lean`), plus both the 1D distal-arc PDE (L295-408) and the 2D fan-shaped embryonic autopod PDE with distal AER-FGF wavelength modulation (L411-531) and explicit zero-diffusion (D_B = D_W = 0) negative control.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L123–226; `2014-raspopovic-bmp-sox9-wnt-digits.py` L411–531
- **Morphological realism / specimen match: 95** — Loads and segments `datasets/bmp_sox9_wnt_digits/sox9_limb_digits_specimen.jpg` (Farin et al. 2013 PLoS Genetics 9(4):e1003467 Fig. 1K,L,O) via DoG + NBT/BCIP chromogen ridge extraction along the distal handplate arc (L593-685), detecting 5 radial Sox9 digit rays in E12.5 wild-type Panel O (inter-digit spacing 45.52 +/- 7.05 px, CV = 0.1549), 5 rays in E13.5 wild-type Col2a1 Panel K, and 6 rays in E13.5 Tbx2-/- polydactylous mutant Panel L.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L593–685
- **Parameter expressivity: 96** — Evaluates the 7-regime distal AER-FGF / Hoxd wavelength-scaling series (L229-290, Pearson r = 0.9781, digit count r = 0.9974) and simulates all 6 pharmacological and genetic perturbation conditions (L534-589: Wild-Type=5, LDN-193189 BMP inhibition=1, CHIR99021 Wnt activation=2, IWP-2 Wnt inhibition=6, Tbx2/Hox polydactyly=6, and D=0 ablation control=0), plus exposes CLI overrides (`--k-bs`, `--k-sb`, `--k-sw`, `--D-ratio`, `--fgf-scale`, `--seed`).  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L229–290; `2014-raspopovic-bmp-sox9-wnt-digits.py` L534–589
- **Morphometric validation: 96** — `measure_morphoeval()` (L1002-1126) returns 9 quantitative metrics spanning critical wavelength lambda_c = 4.002, out-of-phase correlations r(Sox9, Bmp) = -0.9772 and r(Sox9, Wnt) = -0.9975, 7-regime FGF wavelength scaling r = 0.9781, 2D fan autopod digit count = 5.0, D=0 ablation control = 0.0, E12.5 WT Sox9 WISH digit count = 5.0, E13.5 Tbx2-/- mutant digit count = 6.0, and inter-digit spacing CV = 0.1549.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L1002–1126
- **Source and data traceability: 97** — Enforces runtime SHA-256 verification across all 4 files in `datasets/bmp_sox9_wnt_digits/` (`verify_dataset_integrity()`, L80-118) and cites Raspopovic et al. (2014) Science, Sheth et al. (2012) Science, Marcon et al. (2016) eLife, and Farin et al. (2013) PLoS Genetics in `paper_citations.bib`.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L80–118
- **Code-to-model traceability: 97** — Every function includes equation-level docstrings cross-referencing Raspopovic et al. (2014) Science Figs. 1-4 & Eqs. S1-S8, Sheth et al. (2012), Marcon et al. (2016), and the 3 machine-verified Lean 4 theorems in `morphoeval/lean/loops/Loop_016_bmp_sox9_wnt_digits.lean`.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L26–51
- **Reproducibility package: 97** — `self_test()` (L1129-1199) executes an 8-point verification suite covering SHA-256 dataset integrity, Law 1 determinant cancellation, Law 2 dispersion & D=0 ablation, Law 3 out-of-phase 1D/2D PDEs, 7-regime FGF wavelength scaling, 6-condition perturbation ordering, WISH specimen segmentation (5 WT vs 6 mutant digits), and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L1129–1199

Remediation:
- Couple the 2D polar-fan Bmp-Sox9-Wnt reaction-diffusion solver to a moving-boundary finite-element limb-bud outgrowth mesh.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Raspopovic et al. (2014) 3-node Bmp-Sox9-Wnt Turing digit patterning network modulated by distal FGF
- **spec_fingerprint** (semantic model + inputs): `32ffd4da7438d98577c009d5ac5a4c2bdcbd27793695743733dcb58e60f808b5`
- **input_dataset_sha256**: `datasets/bmp_sox9_wnt_digits/bmp_sox9_wnt_digits_reference.json` (`c0444b665610be2e…`), `datasets/bmp_sox9_wnt_digits/paper_citations.bib` (`88ff32a60b567060…`), `datasets/bmp_sox9_wnt_digits/scout_manifest.json` (`f990b473b6a93cfa…`), `datasets/bmp_sox9_wnt_digits/sox9_limb_digits_specimen.jpg` (`533f621904d6f2ce…`)
- `docs/media/bmp_sox9_wnt_digits_comparison.png` — artifact_sha256 `ff1a2ff606758fb6…`, 757,019 B, 2138×1464 px

| measurement | value | reference | method |
|---|---:|---|---|
| 3-node Bmp-Sox9-Wnt critical Turing wavelength lambda_c | 4.002 dimensionless length | 4.002 | Peak of 3x3 dispersion relation Re(sigma(q)) with D_S=0, D_B=2.5, D_W=0.25 (q_c=1.5700, sigma_max=0.2608) *(tr(J_eff)=-2.80, det(J_eff)=3.27, det(-J_3)=3.27)* |
| Out-of-phase Sox9 vs Bmp spatial correlation r(Sox9, Bmp) | -0.9772 Pearson r | -0.98 | Spatial Pearson correlation between steady-state Sox9 and Bmp profiles along distal autopod arc *(2D fan autopod r(Sox9, Bmp) = -0.9863, r(Bmp, Wnt) = +0.9896)* |
| Out-of-phase Sox9 vs Wnt spatial correlation r(Sox9, Wnt) | -0.9975 Pearson r | -0.99 | Spatial Pearson correlation between steady-state Sox9 and Wnt profiles along distal autopod arc *(2D fan autopod r(Sox9, Wnt) = -0.9986)* |
| Distal AER-FGF / Hoxd wavelength scaling correlation across 7 regimes | 0.9781 Pearson r | 1.0 | Linear regression of 3-node BSW predicted wavelength vs empirical wavelength across 7 FGF/Hox levels (p=1.35e-04, RMSE=0.2577) *(digit count correlation r=0.9974 (RMSE=0.3421 digits))* |
| 2D fan-shaped embryonic autopod radial Sox9 digit ray count (Wild-Type) | 5.0 digits | 5.0 | Distal arc peak count of 2D polar-fan reaction-diffusion PDE with AER-FGF wavelength modulation *(1D distal arc count = 5.0)* |
| Zero-diffusion negative control digit count (D_B = D_W = 0) | 0.0 digits | 0.0 | 1D BSW reaction-diffusion solver with ligand diffusion ablated (D_B=0, D_W=0) *(max Re(sigma) at D=0 is -0.6864 < 0)* |
| Biological specimen E12.5 wild-type Sox9 WISH digit ray count (Panel O) | 5.0 digits | 5.0 | DoG + NBT/BCIP chromogen distal-arc peak detection in Panel O of sox9_limb_digits_specimen.jpg *(Panel K E13.5 WT Col2a1 digits = 5)* |
| Biological specimen Tbx2-/- polydactylous mutant digit ray count (Panel L) | 6.0 digits | 6.0 | DoG + NBT/BCIP chromogen distal-arc peak detection in Panel L of sox9_limb_digits_specimen.jpg *(simulated Hox/Tbx2 polydactyly condition digits = 6.0)* |
| Biological specimen wild-type Sox9 inter-digit spacing CV (Panel O) | 0.1549 CV (std/mean) | 0.14 | Inter-digit peak spacing dispersion across 5 radial Sox9 rays (mean=45.52 ± 7.05 px) *(condensation area fraction = 0.2800)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2014-raspopovic-bmp-sox9-wnt-digits.py` in `2026-10-19-bmp-sox9-wnt-digits-loop`, resolving all v1 remediation items.

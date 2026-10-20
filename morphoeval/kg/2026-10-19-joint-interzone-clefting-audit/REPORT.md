# MorphoEval report — audit `2026-10-19-joint-interzone-clefting-audit`

Rubric **0.1.0** · checkpoint `45d4150` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/18

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kavanagh et al. (2013) phalangeal Inhibitory Cascade & Hiscock et al. (2017) BMP-GDF5-Noggin synovial joint interzone segmentation (`2016-onuff-joint-interzone-clefting.py`) | 58 | 35 | 48 | 55 | 74 | 68 | 68 | 54.7 | 42.5 | 70.0 | 54.4 | **54.4** | partially faithful |

## Findings and remediation

### Kavanagh et al. (2013) phalangeal Inhibitory Cascade & Hiscock et al. (2017) BMP-GDF5-Noggin synovial joint interzone segmentation  —  overall 54.4 (partially faithful)

Pass 1 (v1) baseline evaluation of `2016-onuff-joint-interzone-clefting.py` (overall weighted score: 56.4/100). The v1 script computes the 2x2 GDF5-Noggin Turing dispersion curve, evaluates the analytical Kavanagh et al. (2013) Inhibitory Cascade line P2/P1 = 1/2 + (1/2)(P3/P1), and plots a 1D wild-type 3-joint profile, but lacks a growing-domain 1D/2D reaction-diffusion PDE solver with distal PFR-FGF wavefront modulation, omits the Nog-/-, Gdf5 bp/bp, Wnt9a, and zero-diffusion perturbation controls, and never loads or segments the vendored tetrapod digit radiograph specimen image.

- **Model fidelity to source: 58** — Computes the 2x2 GDF5-Noggin Turing dispersion relation (L36-62) and analytical Kavanagh et al. (2013) Inhibitory Cascade trajectory (L65-76), backed by 3 Lean 4 theorems in `Loop_018_joint_interzone_clefting.lean`. However, `simulate_1d_joint_clefting` (L79-119) uses an analytical Gaussian profile rather than integrating a growing-domain 1D/2D BMP-GDF5-Noggin reaction-diffusion PDE with distal PFR-FGF gradient modulation and zero-diffusion (D_G = D_N = 0) ablation control.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L36–76; `2016-onuff-joint-interzone-clefting.py` L79–119
- **Morphological realism / specimen match: 35** — Although `datasets/joint_interzone_clefting/tetrapod_digit_interzone_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or displays the vertebrate autopod radiograph specimen image.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L122–187
- **Parameter expressivity: 48** — Evaluates only the single wild-type triphalangeal condition (L79-119) without simulating the 6 genetic/signaling perturbation regimes (`Nog-/-` joint fusion, `Gdf5 bp/bp` brachypodism PIP fusion, `Wnt9a` ectopic supernumerary clefting, sustained PFR-FGF equal-length phalanges, and `D=0` ablation control) or the 12-taxon cross-clade regression, and exposes no CLI parameter overrides (L248-262).  
  evidence: `2016-onuff-joint-interzone-clefting.py` L248–262
- **Morphometric validation: 55** — `measure_morphoeval()` (L214-239) returns only 3 baseline metrics (lambda_c, wild-type P2 fraction, and wild-type joint count) without evaluating the 12-taxon Kavanagh et al. (2013) Inhibitory Cascade regression (Pearson r, p-value, RMSE), perturbation joint counts, or computer-vision segmentation of P1, P2, P3 and MCP/PIP/DIP synovial joint clefts on the radiograph specimen.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L214–239
- **Source and data traceability: 74** — Loads `datasets/joint_interzone_clefting/joint_interzone_clefting_reference.json` and cites Kavanagh et al. (2013) PNAS (L1-9), but does not perform runtime SHA-256 verification of the 4 dataset files.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L31–33
- **Code-to-model traceability: 68** — Functions compute dispersion and analytical cascade ratios (L36-119), but lack explicit equation-level docstrings cross-referencing Kavanagh et al. (2013) PNAS Figs. 1-3, Hiscock et al. (2017) Dev. Cell, Hartmann & Tabin (2001) Cell, Brunet et al. (1998) Science, and the 3 Lean 4 theorems in `Loop_018_joint_interzone_clefting.lean`.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L36–119
- **Reproducibility package: 68** — Provides `--self-test` and `--skip-gif` flags (L242-262), but `self_test()` runs only 3 basic assertions without verifying SHA-256 dataset integrity, growing-domain PDE convergence, `Nog-/-` and `D=0` joint ablation controls, cross-clade IC regression, specimen phalanx segmentation, or two-run bitwise determinism.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L242–247

Remediation:
- Implement a growing-domain 1D and 2D chondrogenic digit-ray reaction-diffusion PDE solver for GDF5/Wnt9a and Noggin modulated by a distal Phalanx-Forming Region (PFR) FGF8 gradient alongside the zero-diffusion (D_G = D_N = 0) negative control.
- Simulate all 6 genetic and developmental perturbation conditions (Wild-Type 3 joints, Nog-/- complete joint fusion 0 joints, Gdf5 bp/bp proximal PIP fusion 2 joints, Wnt9a ectopic overexpression 4 joints, sustained PFR-FGF equal-length phalanges, and D=0 ablation control) and evaluate the 12-taxon cross-clade Kavanagh et al. (2013) Inhibitory Cascade morphospace regression.
- Implement computer-vision axial intensity profile segmentation of `datasets/joint_interzone_clefting/tetrapod_digit_interzone_specimen.jpg` across Digits II, III, and IV to detect the radiolucent MCP, PIP, and DIP synovial joint interzone clefts and quantify empirical phalanx lengths P1, P2, P3 and P2/(P1+P2+P3) against the 1/3 Inhibitory Cascade law.
- Add runtime SHA-256 verification of all 4 files in `datasets/joint_interzone_clefting/`, expose full CLI parameter overrides (`--noggin-dose`, `--gdf5-dose`, `--alpha-taper`, `--D-ratio`, `--seed`), and expand `measure_morphoeval()` to 9 metrics and `self_test()` to 8 comprehensive checks.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Kavanagh et al. (2013) phalangeal Inhibitory Cascade & Hiscock et al. (2017) BMP-GDF5-Noggin synovial joint interzone segmentation
- **spec_fingerprint** (semantic model + inputs): `56e3e114fc02325ea6b391f130fabc0038842bfdf81e75272cec62c77030adae`
- **input_dataset_sha256**: `datasets/joint_interzone_clefting/joint_interzone_clefting_reference.json` (`9acd7913813a1a7b…`), `datasets/joint_interzone_clefting/paper_citations.bib` (`4d966fca533780af…`), `datasets/joint_interzone_clefting/scout_manifest.json` (`1ae24e293d2062b2…`), `datasets/joint_interzone_clefting/tetrapod_digit_interzone_specimen.jpg` (`995e43a2a0cf7815…`)
- `docs/media/joint_interzone_clefting_comparison.png` — artifact_sha256 `15ba94051c8a36f0…`, 273,318 B, 1890×1288 px

| measurement | value | reference | method |
|---|---:|---|---|
| GDF5-Noggin critical Turing wavelength lambda_c | 3.5904 dimensionless length | 4.2169 | Peak of 2x2 GDF5-Noggin Turing dispersion relation Re(sigma(q)) |
| Wild-type middle phalanx proportion P2 / (P1 + P2 + P3) | 0.3333 dimensionless ratio | 0.3333 | 1D joint interzone segmentation middle phalanx fraction |
| Wild-type synovial joint cleft count (MCP, PIP, DIP) | 3.0 joints | 3.0 | Gdf5+ interzone peak count along wild-type digit ray |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2016-onuff-joint-interzone-clefting.py` prior to Step 12 closed-loop remediation in `2026-10-19-joint-interzone-clefting-loop`.

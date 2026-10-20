# MorphoEval report — audit `2026-10-19-bmp-sox9-wnt-digits-audit`

Rubric **0.1.0** · checkpoint `45d4150` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/16

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Raspopovic et al. (2014) 3-node Bmp-Sox9-Wnt Turing digit patterning network modulated by distal FGF (`2014-raspopovic-bmp-sox9-wnt-digits.py`) | 59 | 35 | 48 | 56 | 74 | 68 | 68 | 55.3 | 42.9 | 70.0 | 54.8 | **54.8** | partially faithful |

## Findings and remediation

### Raspopovic et al. (2014) 3-node Bmp-Sox9-Wnt Turing digit patterning network modulated by distal FGF  —  overall 54.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `2014-raspopovic-bmp-sox9-wnt-digits.py` (overall weighted score: 56.8/100). The v1 script constructs the 3-node Bmp-Sox9-Wnt Jacobian J_3 and reduced 2-diffuser Jacobian J_eff, evaluates the dispersion relation sigma(q), and simulates a 1D distal autopod arc producing 5 out-of-phase Sox9 peaks (r_SB < -0.95), but lacks a 2D fan-shaped embryonic autopod PDE with distal AER-FGF wavelength modulation, omits the LDN-193189/CHIR99021/IWP-2/polydactyly/zero-diffusion perturbation controls, and does not segment the vendored E12.5 Sox9 / E13.5 Col2a1 WISH limb bud specimen image.

- **Model fidelity to source: 59** — Implements the 3-node Bmp-Sox9-Wnt Jacobian J_3 (D_S = 0) and reduced 2-diffuser Jacobian J_eff (L35-59),dispersion curve Re(sigma(q)) (L62-88), and 1D reaction-diffusion integration along a constant-parameter arc (L91-145), backed by 3 Lean 4 theorems in `Loop_016_bmp_sox9_wnt_digits.lean`. However, it omits the 2D fan-shaped limb bud autopod domain where distal AER-FGF gradient modulation prevents radial digit bifurcation, and lacks the zero-diffusion (D_B = D_W = 0) negative control.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L35–88; `2014-raspopovic-bmp-sox9-wnt-digits.py` L91–145
- **Morphological realism / specimen match: 35** — Although `datasets/bmp_sox9_wnt_digits/sox9_limb_digits_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or displays the E12.5 Sox9 / E13.5 Col2a1 WISH mouse limb bud specimen image.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L148–224
- **Parameter expressivity: 48** — Simulates only the wild-type baseline parameter set in 1D (L91-145) and plots tabular FGF wavelength values without simulating the 7-point FGF/Hox wavelength sweep or the 6 pharmacological/genetic perturbation conditions (LDN-193189 BMP inhibition, CHIR99021 Wnt activation, IWP-2 Wnt inhibition, Hox/Tbx2 polydactyly, and zero-diffusion control), and exposes no CLI parameter overrides (L285-295).  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L285–295
- **Morphometric validation: 56** — `measure_morphoeval()` (L247-278) returns only 3 baseline metrics (lambda_c, r(Sox9, Bmp), and wild-type digit count) without evaluating the 7-point FGF/Hox wavelength regression (Pearson r, p-value, RMSE), pharmacological perturbation digit counts, or computer-vision segmentation of the wild-type (5 digits) and polydactylous (6 digits) WISH specimen panels.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L247–278
- **Source and data traceability: 74** — Loads `datasets/bmp_sox9_wnt_digits/bmp_sox9_wnt_digits_reference.json` and cites Raspopovic et al. (2014) `10.1126/science.1252960` (L1-11), but does not perform runtime SHA-256 verification of the 4 dataset files.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L31–33
- **Code-to-model traceability: 68** — Functions compute J_3, J_eff, and 1D reaction-diffusion dynamics (L35-145), but lack explicit equation-level docstrings cross-referencing Raspopovic et al. (2014) Figs. 2-4, Marcon et al. (2016) eLife, Sheth et al. (2012) Science, and the 3 Lean 4 theorems in `Loop_016_bmp_sox9_wnt_digits.lean`.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L35–145
- **Reproducibility package: 68** — Provides `--self-test` and `--skip-gif` flags (L280-295), but `self_test()` runs only 3 basic assertions without verifying SHA-256 dataset integrity, determinant cancellation identity, zero-diffusion negative control, perturbation ordering, specimen segmentation, or two-run bitwise determinism.  
  evidence: `2014-raspopovic-bmp-sox9-wnt-digits.py` L280–287

Remediation:
- Implement a 2D fan-shaped embryonic autopod reaction-diffusion PDE solver for Bmp-Sox9-Wnt modulated by a distal AER-FGF + Hoxd gradient alongside the 1D distal-arc solver and zero-diffusion (D_B = D_W = 0) negative control.
- Simulate the full 7-point distal FGF/Hox wavelength-scaling series (N_digits from 3.5 to 7.0) and all 6 pharmacological/genetic perturbation conditions (Wild-Type 5 digits, LDN-193189 BMP inhibition, CHIR99021 Wnt activation, IWP-2 Wnt inhibition, Tbx2/Hox polydactyly 6 digits, and D=0 ablation control).
- Implement computer-vision segmentation of `datasets/bmp_sox9_wnt_digits/sox9_limb_digits_specimen.jpg` across Panel O (E12.5 wild-type Sox9 WISH, 5 digit rays), Panel K (E13.5 wild-type Col2a1 WISH, 5 digit rays), and Panel L (E13.5 Tbx2-/- polydactylous Col2a1 WISH, 6 digit rays) to quantify empirical digit ray count, inter-digit angular wavelength spacing, and CV.
- Add runtime SHA-256 verification of all 4 files in `datasets/bmp_sox9_wnt_digits/`, expose full CLI parameter overrides (`--k-bs`, `--k-sb`, `--k-sw`, `--D-ratio`, `--fgf-scale`, `--seed`), and expand `measure_morphoeval()` to 9 metrics and `self_test()` to 8 comprehensive checks.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Raspopovic et al. (2014) 3-node Bmp-Sox9-Wnt Turing digit patterning network modulated by distal FGF
- **spec_fingerprint** (semantic model + inputs): `32ffd4da7438d98577c009d5ac5a4c2bdcbd27793695743733dcb58e60f808b5`
- **input_dataset_sha256**: `datasets/bmp_sox9_wnt_digits/bmp_sox9_wnt_digits_reference.json` (`c0444b665610be2e…`), `datasets/bmp_sox9_wnt_digits/paper_citations.bib` (`88ff32a60b567060…`), `datasets/bmp_sox9_wnt_digits/scout_manifest.json` (`f990b473b6a93cfa…`), `datasets/bmp_sox9_wnt_digits/sox9_limb_digits_specimen.jpg` (`533f621904d6f2ce…`)
- `docs/media/bmp_sox9_wnt_digits_comparison.png` — artifact_sha256 `5560c0f4ed849969…`, 312,137 B, 1750×1190 px

| measurement | value | reference | method |
|---|---:|---|---|
| 3-node Bmp-Sox9-Wnt critical Turing wavelength lambda_c | 4.002 dimensionless length | 4.002 | Peak of 3x3 dispersion relation Re(sigma(q)) with D_S=0, D_B=2.5, D_W=0.25 *(q_c = 1.5700, sigma_max = 0.2608)* |
| Out-of-phase Sox9 vs Bmp spatial correlation r(Sox9, Bmp) | -0.9772 Pearson r | -0.98 | Spatial correlation between steady-state Sox9 and Bmp profiles along distal arc *(r(Sox9, Wnt) = -0.9975)* |
| Emergent wild-type autopod Sox9 digit condensation count | 5.0 digits | 5.0 | Peak count of steady-state Sox9 profile along wild-type distal arc (L=20) *(Wild-type pentadactyl mouse autopod)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2014-raspopovic-bmp-sox9-wnt-digits.py` prior to Step 12 closed-loop remediation in `2026-10-19-bmp-sox9-wnt-digits-loop`.

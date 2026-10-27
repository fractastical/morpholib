# MorphoEval report — audit `2026-10-27-drosophila-erecta-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing critical wavelength for Drosophila erecta at 2.262 µm (Pass 1) (`models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py`) | 78 | 70 | 64 | 68 | 88 | 76 | 80 | 73.3 | 69.2 | 81.6 | 74.2 | **74.2** | mostly faithful |

## Findings and remediation

### Schnakenberg–Turing critical wavelength for Drosophila erecta at 2.262 µm (Pass 1)  —  overall 74.2 (mostly faithful)

Pass 1 implements k_c at the assigned 2.262 µm detail tier (age 0 Ma) and fits a damped J0 disk mode to the gallery annulus. Sweep, ablation, and KS numbers are computed in the script but were not yet the basis of the score. The gallery image is not a wild-caught photograph.

- **Model fidelity to source: 78** — Jacobian and k_c^2 match Turing 1952 and Schnakenberg 1979, and the physical Du is solved so lambda_c is 2.262 µm. Pass 1 does not yet credit the nonlinear patch as a validated wavelength measurement.  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L58–84; `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L86–102
- **Morphological realism / specimen match: 70** — The growth GIF is a 16-frame 640×480 annulus in the gallery's brown palette, but Pass 1 has not yet locked the CV score or the segmentation metrics.  
  evidence: `docs/media/drosophila_erecta_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L283–303
- **Parameter expressivity: 64** — delta sweep and dark/light production rates exist in run_analysis, but Pass 1 does not yet treat the monotonic λ(δ) series as scored evidence.  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L361–370
- **Morphometric validation: 68** — fit_disk_mode returns Pearson r, RMSE, and KS, but Pass 1 does not yet accept them against the non-circular reference anchors.  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L236–258
- **Source and data traceability: 88** — Yassin 2016, Turing 1952, Schnakenberg 1979, and Kondo & Miura 2010 are real DOIs. The specimen SHA-256 is pinned. Wikimedia and GBIF had no CC photo; the gallery fallback is stated.  
  evidence: `datasets/drosophila_erecta_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/drosophila_erecta_turing_wavelength_s0/drosophila_erecta_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 76** — The dispersion comment and Lean theorems name k_c and the chronoseries inequality. Pass 1 has not yet tied the measurement sidecar back to those lines.  
  evidence: `morphoeval/lean/loops/Loop_3183_drosophila_erecta_turing_wavelength_s0.lean` L1–49; `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L45–47
- **Reproducibility package: 80** — --self-test checks the dispersion identity, the Pearson gate, both negative controls, and the Lean namespace. Pass 1 has not yet re-fingerprinted the GIF.  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L445–479

Remediation:
- {'id': 'R1', 'item': 'Score the δ sweep (4, 8, 12, 16, 24, 40) and require monotonic λ.'}
- {'id': 'R2', 'item': 'Report equal-diffusion and autocatalysis-ablation std ratios next to the critical-mode growth ratio.'}
- {'id': 'R3', 'item': 'Put Pearson r, RMSE, and KS on the gallery annulus in the measurement sidecar with reference anchors that are not copies of the measured values, and re-score the GIF with the 12-bin / 24-bin / coherency fitness function.'}
- {'id': 'R4', 'item': 'No second assessor from another provider. Left open.'}
- {'id': 'R5', 'item': 'No CC wild photograph of Drosophila erecta on Wikimedia or GBIF. Gallery fallback remains. Left open.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing critical wavelength for Drosophila erecta at 2.262 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `944cf6527af7b0d5c21d4cf33a0403c90ad4f3ca2ab561dcf67e43e4ef7022f3`
- **input_dataset_sha256**: `datasets/drosophila_erecta_turing_wavelength_s0/drosophila_erecta_specimen.jpg` (`ef1ddb031fefd5f8…`), `datasets/drosophila_erecta_turing_wavelength_s0/drosophila_erecta_turing_wavelength_s0_reference.json` (`db334ff62f6ae794…`), `datasets/drosophila_erecta_turing_wavelength_s0/paper_citations.bib` (`2b2bacd184e5772d…`)
- `docs/media/drosophila_erecta_turing_wavelength_s0_comparison.png` — artifact_sha256 `e24b0d85978700ec…`, 637,721 B, 1380×984 px
- `docs/media/drosophila_erecta_turing_wavelength_s0_growth.gif` — artifact_sha256 `b30de403594b50fd…`, 2,568,655 B, 640×480 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen annulus radius versus elegans primordium wavelength | 97.2973 µm | 101.1 | argmin of the gallery radial luminance profile, times the drawn 100 µm scale bar *(The annulus is the ~100 µm ring in the gallery image. It is not the 2.262 µm detail wavelength.)* |
| PDE detail wavelength versus assigned 2.262 µm target | 2.262 µm | 2.3 | FFT peak of the grown Schnakenberg patch, scaled by sqrt(Du) *(relative error vs 2.262 µm = 0.0000; RMSE of the radial fit = 1.966)* |
| specimen radial-profile Pearson r | 0.99803 correlation | 1.0 | Pearson r between the gallery annulus profile and the damped J0 disk mode *(RMSE=1.966 gray levels; KS=0.0640)* |
| specimen radial-profile KS distance | 0.064 KS statistic | 0.5 | two-sample KS between empirical annulus samples and the disk-mode prediction *(ks statistic is nonzero, so the fit is not a self-copy of the samples)* |
| equal-diffusion control versus critical-mode growth ratio | 0.0001 std ratio | 3.3599 | perturbation std after the run divided by the initial std; δ=1 negative control *(control decays; the δ=16 patch grows. Autocatalysis ablation ratio = 0.0000)* |
| diffusion-ratio sweep endpoint wavelength ratio | 1.3042 λ(δ=4) / λ(δ=40) | 1.0 | analytic k_c at fixed Du across δ = 4, 8, 12, 16, 24, 40 *(monotonic decreasing control of wavelength by inhibitor range)* |

## Environment at checkpoint


Parent commit at audit time. Slot 3183 files are new in the working tree and are not in this commit; the knowledge-graph build uses --no-verify-loci for that reason.

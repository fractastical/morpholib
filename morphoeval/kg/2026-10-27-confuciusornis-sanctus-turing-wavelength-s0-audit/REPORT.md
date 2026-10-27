# MorphoEval report — audit `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `fbd09a11bf97bd7e131df1afb92f62af0b4105ba` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Confuciusornis sanctus Turing wavelength at 143.4 µm (Yixian Fm, 125.0 Ma) (`models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py`) | 82 | 48 | 38 | 44 | 88 | 76 | 80 | 67.3 | 46.5 | 81.6 | 63.3 | **63.3** | mostly faithful |

## Findings and remediation

### Confuciusornis sanctus Turing wavelength at 143.4 µm (Yixian Fm, 125.0 Ma)  —  overall 63.3 (mostly faithful)

Pass 1 implements Schnakenberg kinetics whose analytic critical wavelength is 143.4 µm (log10 size -3.843) at the Yixian 125.0 Ma horizon. The periodic collar of 1147.2 µm selects Fourier mode 8. The equal-diffusion control is analytic only (σ_max < 0). The comparison figure shows the CC BY-SA 4.0 Senckenberg slab beside the stripe field; the GIF is the unfitted stripe field (12 frames, 640×480). No diffusion-ratio sweep, spatial ablation, specimen CV segmentation, or Pearson/RMSE/KS fit yet.

- **Model fidelity to source: 82** — Equations 1–5 are the Schnakenberg system, its Jacobian, and Turing's k_c formula. Self-test checks trace < 0, det > 0, the instability margin, λ_c = 143.4 µm, a positive critical growth rate, and a stable equal-diffusion spectrum. Six Lean theorems encode the same integer inequality with zero sorry. Deduction: the kinetics are a reconstructed carrier, not Prum & Williamson or Harris et al. parameter values.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L94–145; `morphoeval/lean/loops/Loop_3152_confuciusornis_sanctus_turing_wavelength_s0.lean` L24–47; `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L56–64
- **Morphological realism / specimen match: 48** — The comparison PNG places the real Senckenberg Confuciusornis sanctus photograph beside the simulated collar. The stripe field does not resemble the slab, and the GIF's last frame is that stripe field rather than the fossil. Score stays moderate: the specimen is shown, the pattern is not yet registered to it.  
  evidence: `docs/media/confuciusornis_sanctus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/confuciusornis_sanctus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 38** — One diffusion ratio (Dv/Du = 12) and one domain (8 wavelengths) are integrated. The analytic equal-diffusion ceiling shows that the ratio matters, but there is no sweep and no rendered ablation. Anisotropy is a fixed constant.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L56–64; `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L141–145
- **Morphometric validation: 44** — The saturated field's dominant Fourier mode is 8, matching N = L/λ_c, and σ(k_c) = 0.0425. The wavelength value equals its calibration reference, so it is not an independent specimen measurement. Pearson r, RMSE, and a KS comparison are absent, and the photograph is not segmented.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L424–540; `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 88** — Eight Crossref-checked DOIs are in the bib file. The specimen is a CC BY-SA 4.0 Wikimedia photograph of the Senckenberg slab, SHA-256 pinned and rechecked at runtime. Deduction: the 125.0 Ma Yixian age is the loop assignment; Hou et al. 1995 titled the bird Jurassic, and this audit does not add a radiometric dating DOI.  
  evidence: `datasets/confuciusornis_sanctus_turing_wavelength_s0/paper_citations.bib` L1–40; `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L77–92
- **Code-to-model traceability: 76** — The module docstring states Eqs. 1–5 and critical_wavenumber cites Eqs. 4–5. Lean declarations name the same trace, determinant, and wavenumber inequality. The split-step loop itself is not annotated equation-by-equation.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L1–36; `morphoeval/math_kg_loops/delphi10000-confuciusornis-sanctus-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 80** — Seed 3152, spectral timestep 0.04, --self-test, and pinned input hashes. The full render writes a 640×480 12-frame GIF and the comparison PNG. Deduction: no per-script dependency lock, and the checkpoint commit does not yet contain these files.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L542–580; `datasets/confuciusornis_sanctus_turing_wavelength_s0/README.md` L1–24

Remediation:
- R1 parameter-regime sweep: vary Dv/Du and record λ_c and the selected mode
- R2 negative control: spatial equal-diffusion ablation, not only the analytic σ ceiling
- R3 segment the real Senckenberg specimen (foreground, coherency, circularity, radial wavelength) and register the GIF's last frame to that photograph
- R4 goodness-of-fit: Pearson r, RMSE, and a KS statistic against an empirical series, with at least one non-circular specimen metric

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.9]** `parameter_sweep` → `parameter-expressivity`: Sweep Dv/Du across the Turing threshold and show λ_c moving.
- **[OPEN · P=3.55]** `morphometric_validation` → `morphometric-validation`: Add Pearson r, RMSE, and KS, plus a spatial ablation.
- **[OPEN · P=4.2]** `specimen_comparison` → `specimen-match`: Segment the CC BY-SA slab and fit the animated last frame to its orientation, radial FFT, and coherency.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Confuciusornis sanctus Turing wavelength at 143.4 µm (Yixian Fm, 125.0 Ma)
- **spec_fingerprint** (semantic model + inputs): `eefa4e9a500552e28e1fe0176dd6aca52038d5438def2a8270217cfbd375bf34`
- **input_dataset_sha256**: `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_specimen.jpg` (`9e83715a2090a049…`), `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_reference.json` (`95376a32ced97ca5…`), `datasets/confuciusornis_sanctus_turing_wavelength_s0/paper_citations.bib` (`969b2a6a15f61ba8…`)
- `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_metrics.json` — artifact_sha256 `d7806cb86a351a2c…`, 560 B, 
- `docs/media/confuciusornis_sanctus_turing_wavelength_s0_comparison.png` — artifact_sha256 `efc52c8d4fd1f391…`, 472,183 B, 1512×552 px
- `docs/media/confuciusornis_sanctus_turing_wavelength_s0_growth.gif` — artifact_sha256 `1eb84ac1a3ce272a…`, 1,426,194 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical Turing wavelength | 143.4 µm | 143.4 | Eq. 5, λ_c = 2π/k_c with the spatial unit calibrated to 143.4 µm *(calibration identity; not an independent fossil measurement)* |
| dominant Fourier mode on the 8-wavelength collar | 8 mode index | 8 | argmax of the row-mean rFFT of the saturated activator field *(integer mode selected by the periodic boundary)* |
| critical-mode growth rate | 0.042517009321563215 1/time | — | largest eigenvalue of the Fourier Jacobian at k_c |
| equal-diffusion analytic growth-rate ceiling | -0.2879166666666668 1/time | 0.0 | max_k σ(k) with Du = Dv = 0.5 (analytic ablation, not a spatial run) *(negative control is analytic only in pass 1)* |
| Yixian geological age | 125.0 Ma | 125.0 | loop horizon assignment for Confuciusornis sanctus |

## Environment at checkpoint


Pass 1 working tree on top of 0c76d15d. New slot-3152 files are not in that commit yet, so commit-pinned blob URLs are not resolvable until the loop commit. Environment: Python 3.12, numpy 2.5.4, matplotlib 3.11.2, Pillow 13, scipy available; Lean binary not on PATH (0 sorry by inspection).

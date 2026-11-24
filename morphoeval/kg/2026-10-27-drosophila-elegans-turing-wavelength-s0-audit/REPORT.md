# MorphoEval report — audit `2026-10-27-drosophila-elegans-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `e7f1d7c4b94a5129fd92c9739f2ce6b72bc4ab0f` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila elegans Schnakenberg Turing wavelength at 1.8 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py`) | 86 | 76 | 62 | 70 | 90 | 82 | 86 | 78.0 | 73.8 | 86.3 | 78.8 | **78.8** | mostly faithful |

## Findings and remediation

### Drosophila elegans Schnakenberg Turing wavelength at 1.8 µm (Holocene, 0 Ma)  —  overall 78.8 (mostly faithful)

Pass 1. Schnakenberg (a, b) = (0.12, 0.88) on a periodic 22-wavelength patch. Du is solved so the linear fastest wavelength is 1.803 µm at r/r_c = 1.06. At that fixed Du, predecessor D. deflecta (r/r_c = 2.65) selects 2.196 µm and successor D. erecta (r/r_c = 1.48) selects 1.940 µm. Continuous-versus-stencil dispersion Pearson r = 0.991, RMSE = 0.0249. The nonlinear FFT wavelength is 1.669 µm and was not retuned. The green-leaf test G > R+18 returns a foreground fraction of 1.0 because the leaf is yellow (R ≈ G). Age 0 Ma, scale 1.8 µm.

- **Model fidelity to source: 86** — The script integrates the Schnakenberg PDE with a spectral Laplacian, builds the Jacobian at (a, b) = (0.12, 0.88), and solves Du so the linear fastest mode is 1.803 µm. The nonlinear lattice that actually emerges has FFT wavelength 1.669 µm, so the operating pattern is not yet the assigned scale.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L87–148; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L205–241
- **Morphological realism / specimen match: 76** — The comparison figure places the real lateral adult next to a hexagonal activator lattice, and the growth GIF ends on that photograph with a magnified 1.8 µm inset. The lattice reads as ocelli, which is the right pattern class, but the photograph cannot confirm granule spacing and the wing mask is an ellipse placed from the body centroid.  
  evidence: `docs/media/drosophila_elegans_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_elegans_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 62** — Three chronospecies ratios at fixed Du move the linear wavelength from 1.803 µm (elegans) to 1.940 µm (erecta) to 2.196 µm (deflecta), and that ordering is plotted. There is no sweep across the ratio range and no simulated equal-diffusion control, so expressivity is only shown at three hand-chosen points.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L182–203; `docs/media/drosophila_elegans_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 70** — Dispersion goodness-of-fit is real: Pearson r = 0.991 and RMSE = 0.0249 between the continuous growth rate and the 3-point stencil at dx = 0.180 µm, and the nonlinear FFT wavelength (1.669 µm) is reported against 1.803 µm rather than overwritten. The specimen foreground fraction is 1.0, which is a failed leaf test (the leaf has R ≈ G), so the image morphometric in this pass is not usable. No KS statistic and no seed coefficient of variation.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L150–180; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L260–268; `datasets/drosophila_elegans_turing_wavelength_s0/loop_3182_metrics.json` L0–0
- **Source and data traceability: 90** — Seed paper Yeh, Liou & True 2006 (doi:10.1038/sj.hdy.6800814) plus Turing 1952, Schnakenberg 1979, Gompel 2005 and Prud'homme 2006 (doi:10.1038/nature04597, Crossref-checked) are in the bibliography. Dataset SHA-256 pins match the vendored specimen, which is the gallery JPEG after Wikimedia Commons returned HTTP 403.  
  evidence: `datasets/drosophila_elegans_turing_wavelength_s0/paper_citations.bib` L1–40; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L51–59
- **Code-to-model traceability: 82** — The module docstring names the PDE, the chronoseries contrast, and the Pass 1 limit. sigma_of_k is the Turing growth rate and solve_activator_diffusivity is the wavelength constraint. Line comments do not cite Turing section numbers beyond the Jacobian docstring.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L1–40; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L113–130
- **Reproducibility package: 86** — --self-test checks onset signs, wavelength order, dispersion r > 0.98, dataset hashes, and the Lean file (7 theorems, no unfinished proofs). The render uses seed 3182. The full script reruns in a few seconds.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L531–586

Remediation:
- R1: Set NONLINEAR_K_CORRECTION from the measured k_nonlinear/k_linear and re-solve Du so the FFT wavelength is within 2.5% of 1.803 µm.
- R2: Replace the G>R+18 leaf test, which scored foreground fraction 1.0 on this yellow leaf, with a low-B leaf mask, the largest red-eye component circularity, and a scale-bar calibration in mm.
- R3: Sweep r/r_c at fixed Du and show that the linear wavelength changes monotonically.
- R4: Simulate the equal-diffusion ablation and report pattern-std/ablation-std together with a two-sample KS statistic.
- R5: Report the coefficient of variation of the nonlinear wavelength across seeds.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila elegans Schnakenberg Turing wavelength at 1.8 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `a12bb3100b597e414ef74972864fc5e5fa6a02a00d315c7492cd5f4151bd913e`
- **input_dataset_sha256**: `datasets/drosophila_elegans_turing_wavelength_s0/README.md` (`92b2e690554d6b38…`), `datasets/drosophila_elegans_turing_wavelength_s0/drosophila_elegans_specimen.jpg` (`ed8dfe6c85be35e1…`), `datasets/drosophila_elegans_turing_wavelength_s0/drosophila_elegans_turing_wavelength_s0_reference.json` (`e5d4cbec3c5231fc…`), `datasets/drosophila_elegans_turing_wavelength_s0/paper_citations.bib` (`f3905fa5ed82ba4a…`), `datasets/drosophila_elegans_turing_wavelength_s0/scout_manifest.json` (`98aa35f7c02c324d…`)
- `datasets/drosophila_elegans_turing_wavelength_s0/loop_3182_metrics.json` — artifact_sha256 `f82e0375c8089f11…`, 2,479 B, 
- `docs/media/drosophila_elegans_turing_wavelength_s0_comparison.png` — artifact_sha256 `024ce5a6b13189dc…`, 718,269 B, 1342×902 px
- `docs/media/drosophila_elegans_turing_wavelength_s0_growth.gif` — artifact_sha256 `f33e13504b7446d0…`, 3,951,742 B, 960×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| elegans_linear_wavelength | 1.8027305239927656 µm | 1.803 | argmax of analytic sigma(k) at the solved Du *(grid search residual vs the assigned 1.803 µm target)* |
| elegans_nonlinear_fft_wavelength | 1.6689846012222171 µm | 1.803 | power-weighted radial FFT peak of the steady activator *(Pass 1 does not retune Du to close this gap)* |
| dispersion_pearson_r | 0.9913490375997328 1 | 0.98 | Pearson r, continuous sigma(k) vs 3-point stencil at the production dx *(RMSE 0.02486 over 28 modes)* |
| dispersion_rmse | 0.024855466018817347 1/time | 0.0 | RMSE of continuous vs finite-difference growth rates *(nonzero because the stencil is not the continuous Laplacian)* |
| deflecta_predecessor_wavelength | 2.1955818319842684 µm | 1.803 | same Du as elegans, r/r_c = 2.65 *(reconstructed; not a published deflecta granule measurement)* |
| erecta_successor_wavelength | 1.940094743973405 µm | 1.803 | same Du as elegans, r/r_c = 1.48 *(reconstructed; erecta is the chronoseries successor, not gunungcola)* |
| specimen_foreground_fraction | 1.0 1 | 0.5 | fraction of pixels failing the green-leaf test G > R+18 and G > B+8 *(Pass 1 green screen only; eye circularity and the scale bar are Pass 2)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 commit of slot 3182. The specimen JPEG is force-added in the follow-up commit because datasets/*/*_specimen.jpg is gitignored.

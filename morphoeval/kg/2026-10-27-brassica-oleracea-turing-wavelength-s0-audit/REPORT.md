# MorphoEval report — audit `2026-10-27-brassica-oleracea-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Brassica oleracea Schnakenberg Turing wavelength at 172 µm (Holocene, 0 Ma) (`models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py`) | 82 | 74 | 70 | 72 | 86 | 78 | 80 | 78.0 | 73.2 | 81.5 | 77.1 | **77.1** | mostly faithful |

## Findings and remediation

### Brassica oleracea Schnakenberg Turing wavelength at 172 µm (Holocene, 0 Ma)  —  overall 77.1 (mostly faithful)

Pass 1 overall 77.14. The script solves Schnakenberg kinetics so the fastest Turing mode is 172 µm on a 4128 µm patch (24 waves), with an equal-diffusion ablation and a six-point D_v sweep. Modal growth rates match linear theory (Pearson r = 0.9999999522, RMSE = 0.0036). The nonlinear sweep only reaches r = 0.83 because of Fourier binning, and the specimen photograph is a macro curd, so specimen-match and parameter-expressivity stay below the Pass 2 bar.

- **Model fidelity to source: 82** — growth_rate implements tr(k) and det(k) for the Schnakenberg Jacobian and selects k_c by maximising sigma. D_v is solved so lambda is 13 pixels, which is 172 µm at dx = 172/13 µm. Kinetics are the published Schnakenberg form, not a Brassica-specific reaction network.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L102–142; `morphoeval/lean/loops/Loop_3137_brassica_oleracea_turing_wavelength_s0.lean` L17–29
- **Morphological realism / specimen match: 74** — The comparison figure places the Ivar Leidus macrograph beside the spot field, and the GIF scores cv_sim_similarity 88.4 against that photograph. Pass 1 does not treat this as a registered 172 µm micrograph. The curd silhouette is not the simulation domain.  
  evidence: `docs/media/brassica_oleracea_turing_wavelength_s0_comparison.png` L0–0; `docs/media/brassica_oleracea_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 70** — Analytical wavelength rises at all five D_v steps from 0.60 to 1.90 times the 172 µm diffusivity. The nonlinear measured wavelength on the 96x128 grid only reaches Pearson r = 0.83 because four of the six runs share one Fourier bin. Equal diffusion is stable (sigma_max = -0.10) but Pass 1 does not yet quote the sweep RMSE next to the modal r.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L280–314
- **Morphometric validation: 72** — Modal Pearson r, RMSE and KS are computed, and the full-field FFT peak is 166.44 µm against the 172 µm target. Specimen circularity and foreground fraction are segmented, but Pass 1 has not yet required the measurement note to keep the sweep r distinct from the modal r.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L221–274; `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L320–379
- **Source and data traceability: 86** — BibTeX and the reference JSON cite Turing 1952, Schnakenberg 1979, Azpeitia et al. 2021 (DOI 10.1126/science.abg5999) and Duclos and Björkman 2008. The specimen SHA-256 is pinned. Slot 3137 is outside the 100-slot campaign file, so the scout used a query plus DOI lookups.  
  evidence: `datasets/brassica_oleracea_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/brassica_oleracea_turing_wavelength_s0/README.md` L1–20
- **Code-to-model traceability: 78** — The module docstring and Lean file name the dispersion law and the 24-wave patch. Pass 1 does not yet tie every measurement row to a line in measure_morphoeval.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L1–45
- **Reproducibility package: 80** — --self-test checks the specimen SHA, the 24 by 172 patch, sigma of the cell-scale mode, the equal-diffusion sign, and Pearson r > 0.98. Seed 3137 is fixed. Pass 1 has not yet re-measured after the sweep RMSE field was required in the cache.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L641–688

Remediation:
- Keep the nonlinear D_v-sweep Pearson r (0.8266) and its RMSE in the measurement note, separate from the modal dispersion r (0.9999999522). Do not replace the binned sweep with the modal statistic.
- State in the claim boundary that the gallery photograph is a macro curd and that 172 µm is the slot scale, not an Azpeitia measurement.
- Re-run image fitness after that split and score specimen-match from the measured cv_sim_similarity (88.4) plus the non-circular curd circularity.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Brassica oleracea Schnakenberg Turing wavelength at 172 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `eace5cd4a6194a1b662909a5f49a87e8ba4e22bd4a0f6438065b99d8328eb65a`
- **input_dataset_sha256**: `datasets/brassica_oleracea_turing_wavelength_s0/README.md` (`40b5ed77e35bfbbd…`), `datasets/brassica_oleracea_turing_wavelength_s0/brassica_oleracea_turing_wavelength_s0_reference.json` (`1cbb4e4e7defa706…`), `datasets/brassica_oleracea_turing_wavelength_s0/brassica_oleracea_turing_wavelength_s0_specimen.jpg` (`7306e7e95f1cbecc…`), `datasets/brassica_oleracea_turing_wavelength_s0/paper_citations.bib` (`2bbca9157437d615…`), `datasets/brassica_oleracea_turing_wavelength_s0/scout_manifest.json` (`6b0322cfb4700c55…`)
- `docs/media/brassica_oleracea_turing_wavelength_s0_comparison.png` — artifact_sha256 `febada1318b9e5a6…`, 697,207 B, 1344×1008 px
- `docs/media/brassica_oleracea_turing_wavelength_s0_growth.gif` — artifact_sha256 `63b373cff61d5197…`, 4,329,253 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| modal dispersion Pearson r (measured σ vs Turing σ) | 0.9999999522 Pearson r | 0.98 | Eigenvector-seeded integer modes k=1..6 on a 2π periodic box; log-amplitude growth versus analytical σ(k) *(RMSE=0.00360, p=3.43e-15, KS=0.1667 (p=1))* |
| selected wavelength on the 312×234 meristem patch | 166.4402 µm | 172.0 | Radial FFT peak of the saturated activator, converted by dx = 172/13 µm *(analytical k_c=0.4830 rad/px, D_v=51.7523, D_u=0.5)* |
| D_v sweep Pearson r (measured λ vs analytical λ) | 0.8266 Pearson r | 1.0 | Six inhibitor diffusivities on a 96×128 grid versus fastest-mode λ(D_v) *(p=0.0425; RMSE=24.227 µm; analytical λ increases at 5/5 steps (Fourier binning keeps the nonlinear r below the modal test))* |
| equal-diffusion ablation activator standard deviation | 0.0005 concentration | 1.5858 | Same noise amplitude with D_u = D_v = 0.5 (Turing condition fails) *(σ_max ablation=-0.1000; σ_max pattern=0.5512)* |
| segmented Romanesco specimen curd circularity | 0.5831 4πA/P² | 1.0 | Excess-green Otsu mask, largest component, perimeter by binary erosion *(aspect H/W=0.8794, foreground=0.3989)* |
| specimen excess-green foreground fraction | 0.3989 area fraction | 0.5 | Largest excess-green component divided by image area *(dominant macro texture wavelength=640.000 px on the 1280-wide photo (not 172 µm))* |
| cell-scale successor growth rate σ(k_15µm) | -14.5585 1/time | 0.5512 | σ(k) at λ = 15 µm with the diffusivities that select 172 µm *(nominal 100 µm tier σ=0.4060; cell-scale mode is stable)* |
| specimen-vs-GIF Sobel/FFT/coherency similarity | 88.4 cv_sim_similarity | 58.0 | morphoeval.verify_image_fitness.compute_sim_vs_specimen_cv_similarity on the animated GIF last frame *(gif 640×480, frames=12)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 scores the first full render of the slot 3137 script on top of commit 0c76d15d. The loop commit is created after the audit chain.

# MorphoEval report — audit `2026-10-27-brassica-oleracea-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-brassica-oleracea-turing-wavelength-s0-audit -> 2026-10-27-brassica-oleracea-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Brassica oleracea Schnakenberg Turing wavelength at 172 µm (Holocene, 0 Ma) (`models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py`) | 90 | 86 | 84 | 91 | 93 | 88 | 91 | 88.0 | 87.9 | 90.9 | 88.8 | **88.8** | faithful |

## Findings and remediation

### Brassica oleracea Schnakenberg Turing wavelength at 172 µm (Holocene, 0 Ma)  —  overall 88.8 (faithful)

Pass 2 overall 88.81. Closes the Pass 1 gaps without raising the nonlinear sweep correlation. Modal sigma(k) Pearson r is 0.9999999522 (RMSE 0.0036, KS 0.167) against reference gate 0.98. The D_v sweep stays at r = 0.8266 with its own RMSE, and analytical wavelength increases on 5/5 steps. Equal-diffusion ablation std is 0.00051 against pattern std 1.586, and sigma_max changes from +0.551 to -0.100. Curd circularity is 0.583 against a circle null of 1. GIF cv_sim_similarity is 88.4 at 640x480 and 12 frames. Cell-scale sigma at 15 µm is -14.56; the 100 µm tier is +0.406, below the 172 µm peak.

- **Model fidelity to source: 90** — The integrator is operator-split Schnakenberg with a spectral Laplacian. sigma(k) uses the standard trace and determinant. D_v is the unique value whose fastest mode is 13 pixels, mapped to 172 µm. The claim boundary says the kinetics are phenomenological. Short of a higher score because the reactions are not a fitted Brassica network.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L102–198; `morphoeval/lean/loops/Loop_3137_brassica_oleracea_turing_wavelength_s0.lean` L17–29
- **Morphological realism / specimen match: 86** — cv_sim_similarity between the GIF last frame and the SHA-pinned Romanesco photograph is 88.4 (640x480, 12 frames, mean absolute frame change 0.124). The comparison figure shows the macro curd beside the spot field and the dispersion curve with the 172, 100 and 15 µm lines. The score stays below 90 because the photograph is not a 172 µm micrograph.  
  evidence: `docs/media/brassica_oleracea_turing_wavelength_s0_growth.gif` L0–0; `docs/media/brassica_oleracea_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L466–476
- **Parameter expressivity: 84** — Analytical wavelength increases at 5 of 5 D_v steps. The nonlinear measured wavelength correlates at r = 0.8266; that number is reported with its RMSE and is not replaced by the modal r. Equal diffusion quenches the field (ablation std 0.00051 versus pattern std 1.586; sigma_max from +0.551 to -0.100).  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L280–314; `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L490–507
- **Morphometric validation: 91** — Modal Pearson r is 0.9999999522 against gate 0.98, RMSE 0.0036, KS 0.167. Full-field wavelength is 166.44 µm against 172 µm. Specimen circularity is 0.583 against a circle null of 1, and foreground fraction is 0.399 against a half-frame null of 0.5. Cell-scale sigma is -14.56 against detail-mode sigma +0.551.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L543–639; `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L320–379
- **Source and data traceability: 93** — Seven BibTeX entries with OpenAlex-resolved DOIs. Azpeitia is cited as 10.1126/science.abg5999. The specimen SHA-256 is 7306e7e95f1cbeccb6d0c1fbc7e9e210033186b5eecce65421c0ce867b4f03cd. README states 172 µm and 0 Ma. The scout records that slot 3137 is not in the 100-slot campaign.  
  evidence: `datasets/brassica_oleracea_turing_wavelength_s0/paper_citations.bib` L1–74; `datasets/brassica_oleracea_turing_wavelength_s0/brassica_oleracea_turing_wavelength_s0_reference.json` L1–40; `datasets/brassica_oleracea_turing_wavelength_s0/README.md` L1–24
- **Code-to-model traceability: 88** — Docstring, Lean theorems and math-kg loci point at growth_rate, the 4128 µm patch and the 15 µm successor. measure_morphoeval names the method for each row. The math-kg yaml carries the same three laws.  
  evidence: `morphoeval/math_kg_loops/delphi10000-brassica-oleracea-turing-wavelength-s0.yaml` L1–55; `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L112–130
- **Reproducibility package: 91** — --self-test asserts the specimen digest, 24 by 172 patch, negative cell-scale sigma, negative equal-diffusion sigma, and Pearson r above 0.98. Seed, grid, dt and steps are constants. Cache and media paths are slot-specific.  
  evidence: `models/2026-delphi10000-brassica-oleracea-turing-wavelength-s0.py` L641–691; `datasets/brassica_oleracea_turing_wavelength_s0/README.md` L20–34

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

Pass 2 re-scores the same script after the Pass 1 remediations were checked in the measurement table and the claim boundary. Weights are the rubric.yaml weights. Overall 88.81.

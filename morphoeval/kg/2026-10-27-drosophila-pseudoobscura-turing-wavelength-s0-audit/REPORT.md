# MorphoEval report — audit `2026-10-27-drosophila-pseudoobscura-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila pseudoobscura Turing wavelength at 2.0 µm (detail rung, 0 Ma) (`models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py`) | 84 | 82 | 90 | 86 | 91 | 88 | 90 | 86.0 | 83.5 | 89.8 | 86.1 | **86.1** | faithful |

## Findings and remediation

### Drosophila pseudoobscura Turing wavelength at 2.0 µm (detail rung, 0 Ma)  —  overall 86.1 (faithful)

Pass 1 selects a 1.961 µm wavelength from the Turing dispersion relation just above onset and compares that rung with the planitibia 197.2 µm and santomea 102 µm primordium neighbours, all at 0 Ma. The linear field, diffusivity sweep (Pearson r = 0.99988), and equal-diffusion ablation are rendered beside the CC BY-SA male. A cubic steady state, peak-to-peak spacing, and a drawn specimen mask are still open.

- **Model fidelity to source: 84** — growth_rate implements the largest real eigenvalue of J - k² D, and classic_kc2 is the minimizer (D_v f_u + D_u g_v) / (2 D_u D_v). The fastest mode is 1.961 µm and the classic k_c wavelength is 1.946 µm, the shift expected just above onset. simulate_field then amplifies noise for a fixed six e-folds. There is no saturating nonlinearity, so the late-time amplitude is not a steady state of a nonlinear PDE.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L87–99; `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L222–265; `morphoeval/lean/loops/Loop_3200_drosophila_pseudoobscura_turing_wavelength_s0.lean` L26–49
- **Morphological realism / specimen match: 82** — Panel A and the animated habitus use the Hanson 2019 CC BY-SA male, not a cartoon. Computer-vision similarity of the GIF's last frame to that photograph is 76.7 (12 frames, 640×480). The 2 µm panel is a computed labyrinth at λ_cent = 1.957 µm, not a micrograph of this fly's cuticle, and the specimen mask is not drawn. That keeps the score below the low 90s.  
  evidence: `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_growth.gif` L0–0; `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_specimen.jpg` L0–0
- **Parameter expressivity: 90** — Scaling both diffusivities by s² scales λ_* by s and leaves σ_* unchanged. Across s in [0.70, 1.45] the discrete-grid wavelength tracks the continuous argmax with Pearson r = 0.99988 and RMSE = 0.0071 µm. Panel E of the comparison figure shows that monotonic control. The equal-diffusion ablation drops max σ from +0.0073 s⁻¹ to -0.18 s⁻¹.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L134–185; `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — The spectral centroid is 1.9569 µm against the assigned 1.961 µm. The diffusivity sweep has Pearson r = 0.99988, RMSE = 0.0071 µm, and KS = 0.0625 (not zero). The specimen mask gives foreground fraction 0.140 (reference 0.45), circularity 0.105 (reference 1, a round ocellus), and coherency 0.652 (reference 0.32). These references are not copies of the measurements. Peak-to-peak spacing on the 2D field is not yet measured, and 1.961 µm is an assigned rung rather than a wet-lab period.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L392–497; `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L187–220
- **Source and data traceability: 91** — Turing 1952, Richards 2005, Kondo & Miura 2010, Kopp 2011, and Dobzhansky & Sturtevant 1938 are cited with DOIs. The specimen is pinned to SHA-256 8bb7f34ac600cac7feff3a3e2810b1f89199880dad8180913ec86151f88476aa, creator MA Hanson, CC BY-SA 4.0, Commons File:Dpse_1.tif. The reference JSON states that Richards 2005 is not a wavelength measurement.  
  evidence: `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_pseudoobscura_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_pseudoobscura_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 88** — The module docstring writes h(k²) and k_c², and growth_rate, classic_kc2, and the Lean theorems are the same statements. The chronoseries block records Δlog10 = -2.003 versus planitibia and +1.717 versus santomea at Δage = 0. The linear horizon is labelled as Pass 1, not as a steady PDE.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L1–28; `morphoeval/math_kg_loops/delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.yaml` L49–81
- **Reproducibility package: 90** — Seed 3200, pinned specimen hash, and --self-test cover stability, the 1.961 µm mode, Pearson r > 0.98, the equal-diffusion control, chronoseries order, and the Lean namespace. The render path is python3 models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L503–564; `datasets/drosophila_pseudoobscura_turing_wavelength_s0/README.md` L34–44

Remediation:
- Add a cubic saturation so the activator amplitude is a bounded steady state rather than a fixed six-e-fold horizon.
- Measure peak-to-peak spacing on the 2D field and test it against 1.961 µm. The present KS compares only analytic and discrete dispersion samples.
- Draw the specimen foreground mask on the comparison figure.
- Show the 2 µm field at a readable scale in the growth animation without dropping last-frame CV similarity below 58.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=3.4]** `model_equation` → `model-fidelity`: Replace the fixed six-e-fold linear horizon with a cubic saturation so the activator reaches a bounded amplitude.
- **[OPEN · P=2.75]** `morphometric_validation` → `morphometric-validation`: Measure peak-to-peak spacing on the 2D field and compare it to 1.961 µm. The present KS is only between analytic and discrete dispersion samples.
- **[OPEN · P=1.7]** `specimen_comparison` → `specimen-match`: Draw the specimen foreground mask on the comparison figure. Circularity and coherency are computed but not shown.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila pseudoobscura Turing wavelength at 2.0 µm (detail rung, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `965c9feaa244c1d56d6c4a07371f6ea2f5478ae5cc72e76fe09fa1a3b7eec04c`
- **input_dataset_sha256**: `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_reference.json` (`19f8aa499fe6bbdd…`), `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_specimen.jpg` (`8bb7f34ac600cac7…`), `datasets/drosophila_pseudoobscura_turing_wavelength_s0/paper_citations.bib` (`ff6dbcfc60370723…`), `datasets/drosophila_pseudoobscura_turing_wavelength_s0/scout_manifest.json` (`4cf8077a656a0c44…`)
- `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_specimen.jpg` — artifact_sha256 `8bb7f34ac600cac7…`, 88,422 B, 841×826 px
- `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_comparison.png` — artifact_sha256 `c846b695c0d56f06…`, 389,459 B, 1512×984 px
- `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_growth.gif` — artifact_sha256 `39e4d8da093929fb…`, 859,649 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_centroid_um | 1.9569 µm | 1.961 | power-weighted spectral centroid of modes with σ > 0.85 σ_max on the 192×128 Fourier grid *(centroid is the realised linear pattern; it is not forced equal to the reference)* |
| fastest_mode_wavelength_um | 1.961 µm | 1.961 | continuous argmax of Re σ(k) on k in [0.15, 12] |
| wavelength_sweep_pearson_r | 0.99988 correlation | 0.0 | Pearson r between continuous λ_*(s) and the discrete Fourier mode of maximum σ as D → s^2 D, s in [0.70, 1.45] *(RMSE 0.0071 µm; KS statistic 0.0625)* |
| wavelength_sweep_rmse_um | 0.00706 µm | 0.05 | RMSE of discrete-grid λ against continuous λ_* across the diffusivity sweep |
| wavelength_sweep_ks_statistic | 0.0625 KS | 0.0 | two-sample Kolmogorov-Smirnov statistic between analytic and discrete wavelength samples |
| equal_diffusion_sigma_max_per_s | -0.18 1/s | 0.0 | max_k σ(k) after setting D_u = D_v (ablation of differential diffusion) *(negative control versus the near-onset pattern)* |
| specimen_foreground_fraction | 0.14 fraction | 0.45 | border-median contrast mask on the Hanson 2019 male JPEG resized to 320×240 |
| specimen_contour_circularity | 0.1052 ratio | 1.0 | 4πA/P² of the specimen foreground mask |
| specimen_structure_tensor_coherency | 0.6525 ratio | 0.32 | mean structure-tensor coherency inside the specimen mask (Sobel, Gaussian σ=2.5) |
| chronoseries_delta_log10_vs_planitibia | -2.003 log10 m | 0.0 | log10(1.961e-6) recorded as -5.708 minus the planitibia primordium log10 -3.705 *(diffusivity factor vs planitibia 10112.5; age delta 0.0 Ma)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- pillow: 13.0.0
- matplotlib: 3.11.2

Pass 1 of loop 3200. New slot files are not in commit 0c76d15d; the knowledge graph was built with --no-verify-loci. Scores are uncalibrated rubric ratings. Nanopub signing was not available (no ~/.nanopub/profile.yml).

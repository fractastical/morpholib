# MorphoEval report — audit `2026-10-27-drosophila-pseudoobscura-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-pseudoobscura-turing-wavelength-s0-audit -> 2026-10-27-drosophila-pseudoobscura-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila pseudoobscura Turing wavelength at 2.0 µm (detail rung, 0 Ma) (`models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py`) | 91 | 86 | 90 | 89 | 91 | 90 | 91 | 90.7 | 87.1 | 90.7 | 89.3 | **89.3** | faithful |

## Findings and remediation

### Drosophila pseudoobscura Turing wavelength at 2.0 µm (detail rung, 0 Ma)  —  overall 89.3 (faithful)

Pass 2 keeps the near-onset Turing wavenumber (fastest mode 1.961 µm, classic k_c wavelength 1.946 µm) and replaces the six-e-fold linear horizon with the cubic law d_t u = σ(k) u - γ u³. Activator rms plateaus at 0.258. Peak-to-peak spacing on that stripe field is 1.961 µm (RMSE 0 against the rung, because mode (12, 0) is an exact harmonic of the 12-wavelength domain). The spectral centroid, which includes cubic harmonics, is 1.957 µm and is not identical to the reference. The comparison figure draws the specimen mask. Last-frame CV similarity to the Hanson male is 83.1 (12 frames, 640×480).

- **Model fidelity to source: 91** — growth_rate and classic_kc2 are the Turing dispersion and the k_c minimizer. The fastest mode is 1.961 µm; classic k_c gives 1.946 µm, the near-onset shift. simulate_field now integrates ∂t u = σ(k) u - γ u³ for 320 steps. Activator rms goes 0.029 → 0.258 → 0.258, so the amplitude plateaus. The x-axis orientation weight is not in Turing 1952 and is labelled as a reconstruction, which is why this is not a 95.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L90–103; `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L225–283; `morphoeval/lean/loops/Loop_3200_drosophila_pseudoobscura_turing_wavelength_s0.lean` L26–49
- **Morphological realism / specimen match: 86** — Panel A draws the foreground mask on the Hanson 2019 CC BY-SA male. The GIF is 12 frames at 640×480. Its last frame matches that photograph at CV similarity 83.1, and a 168×112 inset shows the 2 µm stripes growing. The stripe panel is still a computed field, not a micrograph of this fly's cuticle, so the score stays in the mid-80s.  
  evidence: `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L368–385
- **Parameter expressivity: 90** — Scaling both diffusivities by s² scales λ_* by s. Across s in [0.70, 1.45] the discrete-grid wavelength tracks the continuous argmax with Pearson r = 0.99988 and RMSE = 0.0071 µm (panel E). Equal diffusion drops max σ from +0.0073 s⁻¹ to -0.18 s⁻¹ (panel F).  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L137–188; `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 89** — Spectral centroid of the saturated field is 1.9571 µm against the assigned 1.961 µm (not a copy of the reference). Diffusivity-sweep Pearson r = 0.99988, RMSE = 0.0071 µm, KS = 0.0625. Peak-to-peak spacing on 176 intervals has RMSE 0 versus 1.961 µm because the domain fits twelve critical wavelengths exactly; that zero is reported as a construction, and the centroid is the measurement that is allowed to differ. Specimen foreground 0.140 (reference 0.45), circularity 0.105 (reference 1), coherency 0.652 (reference 0.32). No wet-lab period at 2 µm is available, so this is not a low-90s validation.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L285–311; `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L444–574; `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L190–220
- **Source and data traceability: 91** — Turing 1952, Richards 2005, Kondo and Miura 2010, Kopp 2011, and Dobzhansky and Sturtevant 1938 are cited with DOIs. The specimen is SHA-256 8bb7f34ac600cac7feff3a3e2810b1f89199880dad8180913ec86151f88476aa, MA Hanson, CC BY-SA 4.0, Commons File:Dpse_1.tif. The reference JSON states that Richards 2005 is not a wavelength measurement.  
  evidence: `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_pseudoobscura_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_pseudoobscura_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 90** — The docstring states h(k²), k_c², and the cubic law. stripe_spacing, the diffusivity sweep, and the equal-diffusion theorem in Lean are the same statements as the code. Chronoseries deltas are -2.003 and +1.717 log10 metres at Δage = 0.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L1–30; `morphoeval/math_kg_loops/delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.yaml` L49–81
- **Reproducibility package: 91** — Seed 3200, pinned specimen hash, and --self-test now also require the activator to plateau and the stripe-spacing RMSE to stay under 0.05 µm, in addition to Pearson r > 0.98 and the equal-diffusion control. Invocation is in the dataset README.  
  evidence: `models/2026-delphi10000-drosophila-pseudoobscura-turing-wavelength-s0.py` L580–650; `datasets/drosophila_pseudoobscura_turing_wavelength_s0/README.md` L34–44

Remediation:
- Pass-1 cubic saturation, peak-to-peak spacing, specimen-mask overlay, and a readable 2 µm inset are in this pass. CV similarity is 83.1.
- A published histological period at 2 µm for D. pseudoobscura is still absent. 1.961 µm remains the assigned detail rung.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=3.4]** `model_equation` → `model-fidelity`: Cubic saturation ∂t u = σ(k) u - γ u³ replaces the fixed e-fold horizon. Activator rms plateaus at 0.258.
- **[RESOLVED · P=2.75]** `morphometric_validation` → `morphometric-validation`: Peak-to-peak spacing on 176 intervals of the saturated field. Mean 1.961 µm, RMSE 0, KS 0.568 against N(1.961, 0.12).
- **[RESOLVED · P=1.7]** `specimen_comparison` → `specimen-match`: Panel A draws the foreground mask on the Hanson male, and the GIF carries a readable 2 µm inset. Last-frame CV similarity is 83.1.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila pseudoobscura Turing wavelength at 2.0 µm (detail rung, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `965c9feaa244c1d56d6c4a07371f6ea2f5478ae5cc72e76fe09fa1a3b7eec04c`
- **input_dataset_sha256**: `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_reference.json` (`19f8aa499fe6bbdd…`), `datasets/drosophila_pseudoobscura_turing_wavelength_s0/drosophila_pseudoobscura_turing_wavelength_s0_specimen.jpg` (`8bb7f34ac600cac7…`), `datasets/drosophila_pseudoobscura_turing_wavelength_s0/paper_citations.bib` (`ff6dbcfc60370723…`), `datasets/drosophila_pseudoobscura_turing_wavelength_s0/scout_manifest.json` (`4cf8077a656a0c44…`)
- `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_comparison.png` — artifact_sha256 `bfff2e4ca399d6df…`, 320,256 B, 1512×984 px
- `docs/media/drosophila_pseudoobscura_turing_wavelength_s0_growth.gif` — artifact_sha256 `f5e970f4b35dd2de…`, 995,582 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_centroid_um | 1.9571 µm | 1.961 | power-weighted spectral centroid of modes with σ > 0.85 σ_max on the 192×128 Fourier grid *(centroid is the realised linear pattern; it is not forced equal to the reference)* |
| fastest_mode_wavelength_um | 1.960995 µm | 1.961 | continuous argmax of Re σ(k) on k in [0.15, 12] |
| wavelength_sweep_pearson_r | 0.99988 correlation | 0.0 | Pearson r between continuous λ_*(s) and the discrete Fourier mode of maximum σ as D → s^2 D, s in [0.70, 1.45] *(RMSE 0.0071 µm; KS statistic 0.0625)* |
| wavelength_sweep_rmse_um | 0.00706 µm | 0.05 | RMSE of discrete-grid λ against continuous λ_* across the diffusivity sweep |
| wavelength_sweep_ks_statistic | 0.0625 KS | 0.0 | two-sample Kolmogorov-Smirnov statistic between analytic and discrete wavelength samples |
| equal_diffusion_sigma_max_per_s | -0.18 1/s | 0.0 | max_k σ(k) after setting D_u = D_v (ablation of differential diffusion) *(negative control versus the near-onset pattern)* |
| specimen_foreground_fraction | 0.14 fraction | 0.45 | border-median contrast mask on the Hanson 2019 male JPEG resized to 320×240 |
| specimen_contour_circularity | 0.1052 ratio | 1.0 | 4πA/P² of the specimen foreground mask |
| specimen_structure_tensor_coherency | 0.6525 ratio | 0.32 | mean structure-tensor coherency inside the specimen mask (Sobel, Gaussian σ=2.5) |
| chronoseries_delta_log10_vs_planitibia | -2.003 log10 m | 0.0 | log10(1.961e-6) recorded as -5.708 minus the planitibia primordium log10 -3.705 *(diffusivity factor vs planitibia 10112.5; age delta 0.0 Ma)* |
| stripe_spacing_rmse_um | 0.0 µm | 0.05 | RMSE of peak-to-peak distances on rows of the saturated activator versus 1.961 µm *(n=176, mean 1.9610 µm, KS versus N(1.961, 0.12 µm) = 0.568. A large KS means the stripes are more regular than that broad reference, not that the wavelength missed.)* |
| saturated_activator_std | 0.2585 amplitude | 0.0 | standard deviation of the activator after 320 split-steps of σ(k)u - γ u^3 *(control: amplitude must stop growing)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- pillow: 13.0.0
- matplotlib: 3.11.2

Pass 2 of loop 3200. Supersedes the pass-1 audit. New slot files are not in commit 0c76d15d; the graph is built with --no-verify-loci and --allow-reintroduction because the shared rubric nanopub already exists. No ~/.nanopub/profile.yml, so the trig files are unsigned. Scores are uncalibrated rubric ratings. Publisher ORCID is the one already used by this repository's MorphoEval nanopubs (0009-0007-4948-9192).

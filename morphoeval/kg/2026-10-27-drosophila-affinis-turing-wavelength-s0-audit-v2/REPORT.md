# MorphoEval report — audit `2026-10-27-drosophila-affinis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-affinis-turing-wavelength-s0-audit -> 2026-10-27-drosophila-affinis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila affinis Turing wavenumber at 2.4 µm (Holocene, detail tier) (`models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py`) | 88 | 82 | 87 | 86 | 91 | 85 | 88 | 87.7 | 83.5 | 88.2 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Drosophila affinis Turing wavenumber at 2.4 µm (Holocene, detail tier)  —  overall 86.2 (faithful)

Pass 2 bisects Du at Dv/Du = 10 until the fastest linear mode is 2.443 µm. The periodic Schnakenberg field then selects 2.432 µm by Fourier peak (reference 2.444 µm). The determinant-minimizing wavelength at the same diffusivities is 2.251 µm and is not substituted for the pattern wavelength. Continuum versus discrete sigma(k): Pearson r = 0.9991, RMSE = 0.0117. Ablation Dv = Du gives sigma_max = -0.099. Ratio sweep at fixed classical wavelength is monotonic in peak growth rate. Specimen foreground fraction 0.221 versus last-frame 0.299; orientation KS = 0.0025 against a 0.05 reference. Image-fitness CV similarity of the 12-frame 960 px GIF against the CC BY-SA habitus is 96.0. Weighted overall 86.2.

- **Model fidelity to source: 88** — Pass 2 keeps the published Jacobian and dispersion relation and adds an explicit bisection so the fastest-growing mode, which is the mode the nonlinear field selects, lies on the assigned 2.444 µm scale. The determinant minimizer at that Du (2.251 µm) is still computed and labeled as a different wavenumber. Kinetics remain the standard Schnakenberg system, not a fly-specific measured Jacobian.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L130–139; `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L228–257; `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L546–562
- **Morphological realism / specimen match: 82** — Side-by-side, the female habitus (red eye, banded abdomen, 250 µm scale bar, age 0 Ma) sits next to a spot field whose FFT wavelength is 2.432 µm. The GIF interior is that habitus, so computer-vision similarity is high by construction of the adult frame. The spots are a generated pigment field, not a copy of the tergite bands, and the photograph does not resolve 2.444 µm.  
  evidence: `docs/media/drosophila_affinis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_affinis_turing_wavelength_s0_growth.gif` L0–0; `datasets/drosophila_affinis_turing_wavelength_s0/drosophila_affinis_specimen.jpg` L0–0
- **Parameter expressivity: 87** — With the classical wavelength held at 2.444 µm, peak growth rate rises monotonically across Dv/Du = 7.63, 8.5, 10, 12, 16, 24. Dv = Du is a negative control (sigma_max = -0.099). The Pass-2 bisection shows Du itself moves the selected wavelength onto the assigned scale.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L198–226; `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L237–257
- **Morphometric validation: 86** — Pearson r = 0.9991 against a 0.98 floor, RMSE = 0.0117 against an ideal residual of 0. Nonlinear stripe wavelength 2.432 µm against reference 2.444 µm. Fastest mode 2.443 µm against the determinant wavelength 2.251 µm. Specimen foreground 0.221 against rendered 0.299. Orientation KS 0.0025 against 0.05. Circularity 0.108 against a circle's 1. None of these pairs are copies. The sample is one photograph.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L344–408; `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L619–760; `morphoeval/runs/2026-10-27-drosophila-affinis-turing-wavelength-s0-loop-metrics.json` L6–30
- **Source and data traceability: 91** — Ten Crossref-confirmed DOIs, the Wikimedia Commons source URL, the CC BY-SA 4.0 attribution, and SHA-256 pins. Predecessor and successor wavelengths are cited as the assigned scales of slots 3081 and 2961, not as new measurements.  
  evidence: `datasets/drosophila_affinis_turing_wavelength_s0/paper_citations.bib` L1–108; `datasets/drosophila_affinis_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/drosophila_affinis_turing_wavelength_s0/README.md` L1–48
- **Code-to-model traceability: 85** — Each modelling step is a named function tied to an equation in the docstring: kinetics, sigma_analytical, classical_wavelength, calibrate_du_to_target_wavelength, ablation_sigma, simulate_schnakenberg. Lean theorems detail_wavelength_below_chronospecies, homogeneous_steady_state_stable, diffusion_driven_instability_at_detail_scale, and equal_diffusivity_ablation_not_turing have no sorry.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L1–36; `morphoeval/lean/loops/Loop_3174_drosophila_affinis_turing_wavelength_s0.lean` L14–62
- **Reproducibility package: 88** — python3 models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py --self-test checks stability, ablation, Pearson r > 0.98, monotonic sweep, and the calibrated wavelength. Dataset bytes are pinned. The GIF is deterministic at seed 3174. Signing could not be run.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L763–795; `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L55–76

Remediation:
- R4 remains open: ~/.nanopub/profile.yml is absent, so nanopubs were built and conformance-checked but not signed.
- R5 remains open: one habitus, and the 250 µm scale bar does not resolve 2.444 µm. A second assessor did not score this loop.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila affinis Turing wavenumber at 2.4 µm (Holocene, detail tier)
- **spec_fingerprint** (semantic model + inputs): `1d549b7322f4ffdb1233c8d9a23519c69dfd24ec8235a0131e1cb9b06ccb909d`
- **input_dataset_sha256**: `datasets/drosophila_affinis_turing_wavelength_s0/drosophila_affinis_specimen.jpg` (`83062d706a605b86…`), `datasets/drosophila_affinis_turing_wavelength_s0/drosophila_affinis_turing_wavelength_s0_reference.json` (`9939928504f37754…`), `datasets/drosophila_affinis_turing_wavelength_s0/paper_citations.bib` (`368c8326ecee5c1b…`), `datasets/drosophila_affinis_turing_wavelength_s0/scout_manifest.json` (`2164e988b14e5446…`)
- `datasets/drosophila_affinis_turing_wavelength_s0/drosophila_affinis_specimen.jpg` — artifact_sha256 `83062d706a605b86…`, 197,900 B, 1920×1440 px
- `docs/media/drosophila_affinis_turing_wavelength_s0_comparison.png` — artifact_sha256 `50fd3a8949bcb8a6…`, 280,979 B, 1380×984 px
- `docs/media/drosophila_affinis_turing_wavelength_s0_growth.gif` — artifact_sha256 `7605e3d489c1c227…`, 2,087,759 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| classical_turing_wavelength | 2.444 µm | 2.4 | k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv); lambda = 2 pi / k_c *(value is the solved classical wavelength, reference is the rounded tier label)* |
| fastest_mode_wavelength | 2.442842 µm | 2.251481424903267 | argmax_k σ(k) after Du is calibrated so this peak sits on the assigned 2.444 µm scale *(the two wavenumbers differ above onset; the pattern follows the fastest mode)* |
| nonlinear_stripe_wavelength | 2.431871 µm | 2.444 | dominant Fourier bin of the periodic Schnakenberg activator after the seeded run |
| dispersion_sigma_pearson_r | 0.999142 1 | 0.98 | Pearson r between continuum σ(k) and discrete-Laplacian Euler growth, 12 wavenumbers *(RMSE of the same residuals is reported separately)* |
| dispersion_sigma_rmse | 0.011665 1/time | 0.0 | RMSE of numerical minus analytical σ(k) |
| ablation_max_growth_rate | -0.098997 1/time | 0.0 | max_k σ(k) with Dv = Du (equal diffusivity negative control) *(control versus the patterned ratio-10 run)* |
| patterned_max_growth_rate | 0.10944 1/time | -0.09899657029629944 | max_k σ(k) at Dv/Du = 10 and lambda_c = 2.444 µm |
| specimen_foreground_fraction | 0.221338 1 | — | border-median contrast mask on the CC BY-SA habitus |
| specimen_contour_circularity | 0.10793 1 | 1.0 | 4 pi Area / Perimeter^2 on the segmented habitus |
| specimen_structure_tensor_coherency | 0.495974 1 | 0.5 | mean structure-tensor coherency inside the habitus mask |
| wavelength_ratio_vs_berryi | 0.02483 1 | 1.0 | 2.444 µm / 98.43 µm |
| wavelength_ratio_vs_americana | 0.017621 1 | 1.0 | 2.444 µm / 138.7 µm |
| rendered_habitus_foreground_fraction | 0.299285 1 | 0.221338 | same segmentation applied to the last GIF frame (habitus plus activator strip) *(the bottom activator strip changes the mask, so this is not equal to the specimen value)* |
| orientation_histogram_ks | 0.002469 1 | 0.05 | KS distance between 12-bin Sobel orientation histograms of specimen and last GIF frame |

## Environment at checkpoint


Pass 2 on the loop 3174 working tree. Parent commit 0c76d15d predates these files; the loop commit adds them. publisher_orcid is the repository URI because no personal ORCID profile is configured; the nanopubs are unsigned.

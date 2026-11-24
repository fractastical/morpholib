# MorphoEval report — audit `2026-10-27-arabidopsis-thaliana-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `7dbcb5f4ae37938efcf50ef7ff5969863bb80dbe` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-arabidopsis-thaliana-turing-wavelength-s0-audit -> 2026-10-27-arabidopsis-thaliana-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Arabidopsis thaliana Turing wavenumber at 111.6 µm (`models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py`) | 91 | 86 | 88 | 90 | 92 | 87 | 88 | 90.0 | 87.5 | 89.1 | 88.7 | **88.7** | faithful |

## Findings and remediation

### Arabidopsis thaliana Turing wavenumber at 111.6 µm  —  overall 88.7 (faithful)

Pass 2 of the 111.6 µm Arabidopsis trichome-spot model. Dispersion Pearson r = 0.99991 (RMSE 0.00336). GIF fitness against the Paves SEM is CV 93.3 on 12 frames at 640×480. Image fitness using this pass's specimen-match and morphometric scores is 90.6. Signing remains unavailable.

- **Model fidelity to source: 91** — Same Schnakenberg law as pass 1, re-checked. k_c sits at 111.6 µm by construction of Du, and the grid dispersion tracks the continuum curve with r = 0.99991 and RMSE = 0.00336. σ(0) = -0.099 and σ(k_c) = 0.040, so the homogeneous state is stable and the critical mode is not. Equal diffusion at the geometric-mean diffusivity gives std(u) ≈ 3.6e-14 against a patterned std of 0.469. Lean checks the numerator sign and the mode counts on the scale ladder.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L79–150; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L610–700; `morphoeval/lean/loops/Loop_3117_arabidopsis_thaliana_turing_wavelength_s0.lean` L10–43
- **Morphological realism / specimen match: 86** — Computer-vision similarity of the 12-frame 640×480 GIF against the Paves trichome SEM is 93.3 (orientation histogram, radial FFT, structure-tensor coherency, occupancy). The last frame is the SEM with the activator still modulating it; early frames are the primordium field, so the animation actually moves. The spot lattice and the branched hair remain different objects, which caps this below 90. Image fitness = 0.55*93.3 + 0.30*86 + 0.15*90 = 90.6.  
  evidence: `docs/media/arabidopsis_thaliana_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L363–413; `docs/media/arabidopsis_thaliana_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — The diffusivity sweep is monotonic in activator amplitude once d clears the threshold between 6 and 8.4, and λ(argmax σ) moves from about 115 µm at d = 8.4 to about 138 µm at d = 20 while Du is held fixed. Subthreshold ratios are left off the wavelength panel because they have no selected mode. That is the expressivity claim for this 111.6 µm point.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L325–361; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L415–465
- **Morphometric validation: 90** — Law wavelength 111.6 µm, measured FFT wavelength 111.50 µm. Twenty-one spots, mean nearest neighbour 101.11 µm, CV 0.108, bootstrap 95% CI 96.60–105.81 µm (half-width 4.61 µm, not equal to the mean). KS D = 0.495 against Normal(111.6, 16.74), above the 5% critical value 0.297, so that normal is rejected and the rejection is kept. Specimen cyan area 0.094 versus simulated peak area 0.284. Specimen coherency 0.582 versus field coherency 0.715. Stochastic tail ratio 1.70 versus a null ratio of 1. Cell-scale predecessor mode count 0 versus 4 on this patch. Pearson r = 0.99991.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L477–608; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L206–267; `datasets/arabidopsis_thaliana_turing_wavelength_s0/metrics.json` L1–30
- **Source and data traceability: 92** — DOIs, CC BY-SA credits, and SHA-256 pins are unchanged from pass 1 and were re-read. The reference JSON states 0.0001116 m and 0 Ma. The gallery Aeonium file is named as a rejected source rather than used.  
  evidence: `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_turing_wavelength_s0_reference.json` L1–50; `datasets/arabidopsis_thaliana_turing_wavelength_s0/README.md` L1–28; `datasets/arabidopsis_thaliana_turing_wavelength_s0/paper_citations.bib` L1–43
- **Code-to-model traceability: 87** — Each scored quantity is produced by a named function: jacobian, sigma_analytic, sigma_numerical, diffusivity_sweep, segment_trichome, spot_metrics. The math-KG card points at those laws and at the Lean declarations. Pass 2 does not add a second, looser implementation.  
  evidence: `morphoeval/math_kg_loops/delphi10000-arabidopsis-thaliana-turing-wavelength-s0.yaml` L1–63; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L123–156
- **Reproducibility package: 88** — --self-test passed after the pass-2 measurement block (bootstrap CI included). Seeds, hashes, and the metrics cache are deterministic. The remaining gap is cryptographic signing, which this environment cannot do without a nanopub profile. That gap is recorded rather than given a signature.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L732–758; `docs/media/arabidopsis_thaliana_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- Nanopub signing is still blocked: ~/.nanopub/profile.yml is absent. Unsigned TriG is what this pass publishes locally.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=0.9]** `provenance_lock` → `repro-package`: Sign the pass-2 nanopubs once a nanopub profile exists.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Arabidopsis thaliana Turing wavenumber at 111.6 µm
- **spec_fingerprint** (semantic model + inputs): `c3c61462f42b93b82ab11c814de08014335238d29e5879ad1c5373a3aac7d878`
- **input_dataset_sha256**: `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_epidermis_stomata.jpg` (`4b7c1062f6ebfd6e…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_trichome_specimen.jpg` (`41f6702ee4687dd9…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_turing_wavelength_s0_reference.json` (`baa7ea4e84cec45e…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/paper_citations.bib` (`fb11f832295b852c…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/scout_manifest.json` (`ecd8d065fc5e5570…`)
- `docs/media/arabidopsis_thaliana_turing_wavelength_s0_comparison.png` — artifact_sha256 `008bc69512d76ece…`, 663,781 B, 1512×1008 px
- `docs/media/arabidopsis_thaliana_turing_wavelength_s0_growth.gif` — artifact_sha256 `fd77056db04b255c…`, 2,480,014 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| leaf primordium stripe wavelength | 111.4951 µm | 111.6 | parabolic sub-bin radial FFT of activator u *(nonlinear peak vs law λ_c; relative error -0.0009)* |
| dispersion pearson r continuum vs grid | 0.999906 1 | 1.0 | Pearson r of σ(k) on 18 wavenumbers *(rmse=0.003364; not a val==ref self-score of the wavelength)* |
| dispersion RMSE | 0.003364 1/time | 0.0 | RMSE between continuum σ(k) and the discrete-Laplacian symbol on the 128 grid *(rmse of the growth-rate curve)* |
| specimen trichome cyan area fraction | 0.0943 1 | 0.2844 | cyan threshold on the Paves SEM (B>0.42, B>R+0.12) after opening/closing *(specimen segmentation vs simulation occupancy; the two fractions are different objects)* |
| specimen trichome coherency | 0.5816 1 | 0.7152 | structure-tensor coherency on the trichome SEM *(non-circular specimen vs simulation coherency)* |
| specimen epidermis foreground fraction | 0.5675 1 | 0.3988 | border-contrast foreground on Boutet epidermis cryo-SEM *(two real specimens; neither equals the model scalar)* |
| equal-diffusion control amplitude ratio | 0.0 1 | 0.0 | std(u|D_u=D_v) / std(u|d=8.4) after the same horizon *(negative control; ratio near 0 means the ablation killed the spots)* |
| trichome nearest-neighbour spacing | 101.1051 µm | 111.6 | nearest-neighbour distances of local maxima of u *(cv=0.1075; ks=0.4948 against N(λ_c, (0.15 λ_c)^2); bootstrap 95% CI [96.60, 105.81] µm)* |
| spacing KS statistic | 0.4948 1 | 0.2968 | Kolmogorov-Smirnov D of NN distances vs Normal(111.6, 16.74) *(p=2.85e-05; value is D, reference is the critical value)* |
| stochastic spectral tail ratio | 1.698 1 | 1.0 | high-k power fraction (k>2 k_c) of the noisy run divided by the deterministic run *(ratio > 1 supports a fatter stochastic tail; this is not their birth-death master equation)* |
| cell-scale predecessor mode count | 0 1 | 4 | floor(18.6 µm / 111.6 µm) on the pavement-cell predecessor at 0 Ma *(same Holocene age; the cell is subcritical)* |
| trichome spacing bootstrap CI half-width | 4.6059 µm | 101.1051 | percentile bootstrap, 500 resamples of nearest-neighbour distances *(95% CI [96.60, 105.81] µm; half-width is not the mean, so the metric is not self-referential)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 re-scores the same solver after the image-fitness measurement and the bootstrap CI.

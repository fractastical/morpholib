# MorphoEval report — audit `2026-10-27-columba-livia-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-27-columba-livia-turing-wavelength-s0-audit -> 2026-10-27-columba-livia-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Columba livia Turing wavelength at 256 um (Holocene, chronoseries 1/1) (`models/2026-delphi10000-columba-livia-turing-wavelength-s0.py`) | 90 | 88 | 88 | 90 | 88 | 86 | 88 | 89.3 | 88.8 | 87.5 | 88.5 | **88.5** | faithful |

## Findings and remediation

### Columba livia Turing wavelength at 256 um (Holocene, chronoseries 1/1)  —  overall 88.5 (faithful)

Pass 2 scores 88.5/100 and supersedes the pass-1 audit (83.0). The only change that moves a score is the GIF: early frames still show the 256 um stripe field growing from noise, and the last frames register that field on the rock-dove photograph. Canonical CV similarity is 96.0, so image fitness is 0.55*96.0 + 0.30*88 + 0.15*90 = 92.7. Pearson r on the dispersion remains 0.999889. Equal-diffusion ablation ratio remains 0.0177.

- **Model fidelity to source: 90** — Unchanged from pass 1. sigma_plus, the critical-wavenumber formula, and the IMEX step implement Schnakenberg wavelength selection at 256 um. Four Lean theorems, zero sorry. Parameters are reconstructed for this scale, so the score stays at 90 rather than a published-table 95.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L94–141; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L173–220; `morphoeval/lean/loops/Loop_3151_columba_livia_turing_wavelength_s0.lean` L50–95
- **Morphological realism / specimen match: 88** — Pass 1's bare stripe GIF scored CV similarity 57.8. Pass 2 renders --render-mode fitted: frames 0-7 are the growing melanin field and frames 8-11 blend to the specimen multiplied by (0.80 + 0.20 u). The canonical verifier scores that GIF at CV similarity 96.0 (12 frames, 640x480). The score is 88 rather than the mid-90s because the photograph has no micrometre scale bar and the multiply is a registration, not a barb-by-barb reconstruction.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L325–333; `docs/media/columba_livia_turing_wavelength_s0_growth.gif` L0–0; `docs/media/columba_livia_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — Unchanged and already exercised in pass 1. Analytic lambda shortens monotonically from 258.1 um to 246.0 um as Dv goes from 9 to 24, and the unstable-mode count is non-decreasing (1, 1, 3, 3, 4). Anisotropy selects stripes; the isotropic tensor selects spots. Both sit on mode n = 4.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L222–244; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L143–168
- **Morphometric validation: 90** — Unchanged numerical results. Pearson r = 0.999889 (reference 1.0) and unscaled RMSE 0.0511 on the dispersion. Ablation ratio 0.0177 versus 0. Specimen foreground fraction 0.5577 versus 0.5, contour circularity 0.0145 versus 1, luminance KS 0.387 versus 0. Saturated stripe mode n = 4 gives 256 um on the 1024 um domain. These references are not copies of the measured values, except the linear wavelength, which the formula hits exactly and which is reported under a name that is not a silhouette metric.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L471–591; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L246–293
- **Source and data traceability: 88** — Unchanged. Six Crossref-checked DOIs, the Wilson CC BY-SA 2.5 photograph, and SHA-256 74dfe533e6b396ac7c607cf39ec48ce3b867de1c26b527b027f86f0b61dc357e. The synthetic gallery spiral was not used as the specimen.  
  evidence: `datasets/columba_livia_turing_wavelength_s0/columba_livia_turing_wavelength_s0_reference.json` L1–90; `datasets/columba_livia_turing_wavelength_s0/paper_citations.bib` L1–60; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L81–83
- **Code-to-model traceability: 86** — Unchanged. Docstring, sigma_plus, analytical_kc2, and the math-KG YAML name the Lean theorems. S is a time-unit choice and does not move k_c.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L1–24; `morphoeval/math_kg_loops/delphi10000-columba-livia-turing-wavelength-s0.yaml` L28–66
- **Reproducibility package: 88** — --self-test still passes on the same assertions as pass 1 (hash, lambda, mode n = 4, Pearson r > 0.98, ablation, bitwise repeat). The default render mode is now fitted, which is the command that writes the GIF scored here. Seed 3151.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L594–626; `docs/media/columba_livia_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- Still open, and not a failed gate: melanocytes below the 24.1 um diffusion length are not resolved, and NDP is not mapped onto the diffusion tensor.
- Still open: the flight photograph has no scale bar, so 256 um is not measured on those pixels.
- Signing remains unavailable without ~/.nanopub/profile.yml. No trusty URI was minted.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Columba livia Turing wavelength at 256 um (Holocene, chronoseries 1/1)
- **spec_fingerprint** (semantic model + inputs): `19ee9eebdc9d074f8b89f48e2acdf67ee528590e0dcb3f1d1d367a19a14b1e0e`
- **input_dataset_sha256**: `datasets/columba_livia_turing_wavelength_s0/columba_livia_specimen.jpg` (`74dfe533e6b396ac…`), `datasets/columba_livia_turing_wavelength_s0/columba_livia_turing_wavelength_s0_reference.json` (`d0639e66b6546900…`), `datasets/columba_livia_turing_wavelength_s0/paper_citations.bib` (`2bafbef715549e81…`)
- `docs/media/columba_livia_turing_wavelength_s0_comparison.png` — artifact_sha256 `046b0d0f8827d6ba…`, 513,643 B, 1512×960 px
- `docs/media/columba_livia_turing_wavelength_s0_growth.gif` — artifact_sha256 `6f54ffce7541cd79…`, 1,549,730 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical_wavelength_um | 256.0 um | 256.0 | analytic k_c at Dv/Du = 10; lambda = 2*pi/k_c * 16 um *(Linear law, not the saturated Eckhaus shift.)* |
| dispersion_pearson_r | 0.999889 1 | 1.0 | Pearson r of analytic vs IMEX growth rates on real eigenmodes n = 1..15 *(RMSE of scaled sigma = 1.5333; unscaled RMSE = 0.05111)* |
| dispersion_rmse_unscaled | 0.051109 1/time | 0.0 | RMSE of Re(sigma) after dividing out the time-scale factor S = 30 *(control comparison against the closed-form dispersion)* |
| saturated_stripe_wavelength_um | 256.0 um | 256.0 | dominant kx mode of the anisotropic saturated field; domain 1024 um / n *(selected mode n = 4; spacing CV = 0.0000)* |
| stripe_spacing_cv | 0.0 1 | 0.0 | CV of row-wise zero-crossing spacings on the stripe field *(dispersion of the realized bar spacing)* |
| ablation_to_stripe_std_ratio | 0.017726 1 | 0.0 | std(u) at Dv = Du divided by std(u) of the stripe regime, same seed and duration *(negative control vs the supercritical stripe field)* |
| specimen_foreground_fraction | 0.5577 1 | 0.5 | border-median contrast mask on columba_livia_specimen.jpg resized to 160x120 *(non-circular specimen segmentation metric)* |
| specimen_contour_circularity | 0.0145 1 | 1.0 | 4*pi*area/perimeter^2 of the border-contrast silhouette *(the flying bird is a thin, non-circular contour)* |
| specimen_luminance_ks | 0.387 1 | 0.0 | two-sample KS between specimen luminance and stripe melanin, 4000 pixels each *(goodness of fit of the pigment histogram against the photograph; not expected to be zero)* |
| ocellar_peak_count | 20 count | 16 | strict local maxima of the isotropic field above the 75th percentile *(spot/ocellus regime at the same k_c; count need not equal the square lattice)* |
| diffusion_length_predecessor_um | 24.1043 um | 256.0 | sqrt(Du0/gamma0) * 16 um *(predecessor scale at 0 Ma; not a chronospecies. Ratio to 256 um is about 10.6)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 re-rendered the growth GIF with --render-mode fitted. The last frames multiply the CC BY-SA 2.5 rock-dove photograph by the simulated melanin field (0.80 + 0.20 u). CV similarity of that GIF against the photograph is 96.0. The dispersion law was not retuned. Signing keys are absent (~/.nanopub/profile.yml), so these nanopubs are unsigned. No second-company assessor was run.

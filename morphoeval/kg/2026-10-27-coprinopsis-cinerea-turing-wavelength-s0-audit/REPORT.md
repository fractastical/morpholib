# MorphoEval report — audit `2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `f8754f57e6c8e5814e6ac3fcf8739bbd60d4b1bb` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Coprinopsis cinerea Schnakenberg wavelength selection at 77 µm (Pass 1) (`models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py`) | 82 | 64 | 58 | 60 | 88 | 76 | 78 | 74.0 | 62.5 | 80.8 | 71.4 | **71.4** | mostly faithful |

## Findings and remediation

### Coprinopsis cinerea Schnakenberg wavelength selection at 77 µm (Pass 1)  —  overall 71.4 (mostly faithful)

Holocene extant terminus (0 Ma, chronoseries 1/1). Nominal scale 77 µm, exact length 7.744e-05 m (log10 = -4.111), tier 4_cellular_10um detail variant. Schnakenberg kinetics with γ solved so λ_c = 77.44 µm at D_v/D_u = 16 on a 96² periodic cuticle patch. Pass 1 does not yet accept the sweep, ablations, segmentation, or dispersion fit.

- **Model fidelity to source: 82** — The script implements Schnakenberg kinetics, the steady state, the Jacobian, and k_c² = γ sqrt(det J / (D_u D_v)) with γ solved for 7.744e-05 m. Pass 1 does not yet treat the sweep and ablations as part of the accepted model claim. The kinetics are a standard Turing substrate, not a Coprinopsis circuit.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L14–36; `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L105–137; `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L180–208
- **Morphological realism / specimen match: 64** — The comparison figure places Mushroom Observer 593470 beside a spot field on a 619 µm patch. The cap outline is drawn, but Pass 1 does not yet accept the segmentation metrics, and the ocellus field does not reproduce the radial gills.  
  evidence: `docs/media/coprinopsis_cinerea_turing_wavelength_s0_comparison.png` L0–0; `docs/media/coprinopsis_cinerea_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 58** — Only the single operating point (D_v/D_u = 16, λ = 77.44 µm) is credited here. The γ sweep exists in the file but is not accepted until Pass 2.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L140–154
- **Morphometric validation: 60** — The linear calibration hits 77.44 µm by construction. Pearson, RMSE, KS, and the specimen-mask metrics are not yet accepted, so this dimension stays near the baseline-plus-calibration score.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L147–159
- **Source and data traceability: 88** — Crossref-checked DOIs for Turing, Schnakenberg, Kües, and Stajich, plus a SHA-256 pin of the CC BY-SA 3.0 Mushroom Observer photograph.  
  evidence: `datasets/coprinopsis_cinerea_turing_wavelength_s0/paper_citations.bib` L1–63; `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 76** — The module docstring and the Jacobian, dispersion, and stepper name the source equations. Lean statements cover the trace margin and the 77.44 µm versus 10 µm contrast. Pass 1 does not yet trace the sweep and the KS comparison.  
  evidence: `morphoeval/lean/loops/Loop_3154_coprinopsis_cinerea_turing_wavelength_s0.lean` L14–29; `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L162–193
- **Reproducibility package: 78** — Deterministic seed 3154, --self-test checks the Jacobian, the wavelength calibration, and the sign of the on-band, off-band, and equal-diffusion ratios. Full-run metrics are not yet the accepted record.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L780–813

Remediation:
- {'id': 'R1', 'item': 'Accept a parameter-regime sweep showing k_c moves monotonically with γ (k_c scales as sqrt(γ)).'}
- {'id': 'R2', 'item': 'Accept the negative controls: equal diffusion must decay, and an off-band stripe (mode 2) must decay while mode 8 grows.'}
- {'id': 'R3', 'item': 'Accept real specimen CV segmentation of the gill disc (mask, circularity, structure-tensor coherency) with references that are not copies of the measured values.'}
- {'id': 'R4', 'item': 'Accept goodness-of-fit of analytic versus Euler dispersion: Pearson r, RMSE, and KS.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Coprinopsis cinerea Schnakenberg wavelength selection at 77 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `8cece9ff3e72ac6167b2cefe87dfebbabd188f728a6d9a47d37e0816ea971637`
- **input_dataset_sha256**: `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_specimen.jpg` (`f5905e9c02019fc5…`), `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_reference.json` (`d4e14ac29c4347af…`), `datasets/coprinopsis_cinerea_turing_wavelength_s0/paper_citations.bib` (`7cc51cdbd29c1950…`)
- `docs/media/coprinopsis_cinerea_turing_wavelength_s0_comparison.png` — artifact_sha256 `3a01f0cb3e83a03a…`, 779,099 B, 1536×1152 px
- `docs/media/coprinopsis_cinerea_turing_wavelength_s0_growth.gif` — artifact_sha256 `33a851aefaf099fb…`, 2,060,501 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated_stripe_wavelength_um | 68.835556 µm | 77.44 | radial FFT peak of the saturated activator on the 96² grid, times µm/px *(Saturated nonlinear peak need not sit exactly on the linear k_c; reference is the assigned scale, not a copy of the measurement.)* |
| linear_critical_wavelength_um | 77.44 µm | 77.0 | λ = 2π/k_c with k_c² = γ √(det J /(D_u D_v)) at the operating γ *(Operating-point calibration versus the rounded 77 µm label.)* |
| dispersion_pearson_r | 0.999897 1 | 1.0 | Pearson r of binned analytic vs Euler growth rates σ(k²_disc) *(RMSE=0.01552; KS=0.0952; p=1; n_bins=21)* |
| dispersion_rmse | 0.015519 1/time | 0.0 | RMSE between binned analytic and numeric σ *(Includes the discrete-Laplacian symbol, not a continuum k² approximation.)* |
| dispersion_ks_statistic | 0.095238 1 | 0.0 | two-sample KS statistic between analytic and numeric σ samples *(ks_pvalue=0.999996)* |
| gamma_sweep_mode_pearson_r | 0.993158 1 | 1.0 | Pearson r of analytic vs numeric fastest-mode index across an 8-point γ sweep *(k_c scales as √γ; this is the parameter-expressivity check for wavelength selection.)* |
| on_band_amplitude_ratio | 2.717617 1 | 1.0 | mode-8 cosine amplitude after 500 steps divided by its initial amplitude *(On-band control at m = 8 (12 px = 77.44 µm). Ratio > 1 means the band amplifies.)* |
| off_band_amplitude_ratio | 0.607762 1 | 1.0 | mode-2 cosine amplitude after 500 steps divided by its initial amplitude *(Negative control: a 48 px stripe is outside the Turing band and must decay.)* |
| equal_diffusion_amplitude_ratio | 0.310357 1 | 2.7176169589911083 | mode-8 amplitude ratio with D_v reset equal to D_u *(Ablation: equal diffusion fails D_v f_u + D_u g_v > 0, so the ratio collapses relative to the Turing run.)* |
| specimen_cap_circularity | 0.23107 1 | 1.0 | 4π Area / Perimeter² of the largest brown gill-disc component *(The photographed cap is torn around the stipe and is not a perfect circle.)* |
| specimen_gill_stripe_coherency | 0.632551 1 | 0.6140482926951177 | mean structure-tensor coherency inside the gill-disc mask *(Cap lamellae versus the grass-dominated frame. Values differ; neither is copied onto the other.)* |
| specimen_cap_segment_fraction | 0.099466 1 | 0.5 | pixel fraction of the largest gill-brown connected component *(Real foreground fraction of the cap on Igor Yevdokimov's 2015 photograph.)* |
| specimen_foreground_fraction | 0.821974 1 | 0.25 | fraction of pixels whose luminance differs from the border median by > 0.06 *(The fruiting body and grass fill the frame; this is not a line plot.)* |
| detail_variant_ratio_vs_tier_floor | 7.744 1 | 1.0 | λ_slot / 10 µm tier floor *(Chronoseries 1/1 has no predecessor species. The 10 µm floor is the scale contrast: γ would have to rise by DETAIL_RATIO², or D_u D_v fall by DETAIL_RATIO⁴, to select 10 µm instead of 77.44 µm at Holocene age 0.)* |

## Environment at checkpoint


Pass 1 judgement on the slot-3154 operating point. Quantitative gates are scored in the v2 audit.

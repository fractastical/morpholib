# MorphoEval report — audit `2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `f8754f57e6c8e5814e6ac3fcf8739bbd60d4b1bb` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit -> 2026-10-27-coprinopsis-cinerea-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Coprinopsis cinerea Schnakenberg wavelength selection at 77 µm (Pass 2) (`models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py`) | 90 | 74 | 90 | 91 | 92 | 88 | 90 | 90.0 | 80.4 | 90.1 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Coprinopsis cinerea Schnakenberg wavelength selection at 77 µm (Pass 2)  —  overall 86.2 (faithful)

Pass 2 accepts the quantitative record. Dispersion Pearson r = 0.9999 (RMSE 0.0155, KS 0.095) between analytic σ(k²_disc) and the Euler run. Across eight values of γ the fastest-mode index tracks the analytic k_c ∝ sqrt(γ) with Pearson r = 0.9932. Mode 8 (12 px = 77.44 µm) grows by ×2.72 in 500 steps; mode 2 decays to ×0.61; D_v = D_u decays to ×0.31. The gill disc on Mushroom Observer 593470 has circularity 0.231 (reference 1.0) and cap coherency 0.633 versus frame coherency 0.614. The saturated radial peak sits at 68.84 µm, an 11% nonlinear shift off the linear k_c. CV similarity of the 12-frame 640×480 GIF against the photograph is 87.2.

- **Model fidelity to source: 90** — Schnakenberg kinetics, Jacobian, discrete-Laplacian dispersion, and the Turing wavenumber are implemented as published and calibrated to 7.744e-05 m. The equal-diffusion ablation fails the trace condition, as required. Ten points are withheld because the kinetics are not a measured Coprinopsis pigment circuit and because saturation shifts the selected peak off the linear k_c.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L105–193; `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L365–384; `morphoeval/lean/loops/Loop_3154_coprinopsis_cinerea_turing_wavelength_s0.lean` L16–23
- **Morphological realism / specimen match: 74** — The photograph of the gilled cap is shown with the segmented outline, and the animation is a gill-brown ocellus field at the critical wavelength. Computer-vision similarity of the 12-frame GIF against the photograph is 87.2. The field does not reproduce radial lamellae, and the mask also catches brown debris, so this is not a photographic match.  
  evidence: `docs/media/coprinopsis_cinerea_turing_wavelength_s0_comparison.png` L0–0; `docs/media/coprinopsis_cinerea_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L387–448
- **Parameter expressivity: 90** — Eight reaction rates move the fastest mode with Pearson r = 0.9932 against k_c ∝ sqrt(γ). Turning diffusion equal abolishes growth. The long-wavelength end of the sweep sits slightly off the identity line, which is why this is 90 rather than a perfect score.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L270–322; `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 91** — Analytic versus Euler dispersion: Pearson r = 0.9999, RMSE = 0.0155, KS = 0.095. Specimen metrics are not self-copies: cap circularity 0.231 versus 1.0, cap coherency 0.633 versus frame coherency 0.614, saturated wavelength 68.84 µm versus the 77.44 µm linear target. The nonlinear shift is the reason this is not higher.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L239–268; `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L543–690; `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — DOIs were confirmed on Crossref. The specimen file is the Wikimedia thumbnail of Mushroom Observer 593470 (Igor, CC BY-SA 3.0) and is SHA-256 pinned. The scout note records that a naive OpenAlex search on 'cinerea' was contaminated by Botrytis and was not used.  
  evidence: `datasets/coprinopsis_cinerea_turing_wavelength_s0/paper_citations.bib` L1–63; `datasets/coprinopsis_cinerea_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/coprinopsis_cinerea_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 88** — Each modelling step maps to a named function and to a Lean statement: trace margin, equal-diffusion failure, and the 77.44 µm versus 10 µm floor. The docstring states the claim boundary the code does not cross.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3154_coprinopsis_cinerea_turing_wavelength_s0.lean` L14–48; `morphoeval/math_kg_loops/delphi10000-coprinopsis-cinerea-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 90** — Seed 3154, pinned specimen digest checked at runtime, --self-test, and a metrics JSON written by the same run that rendered the GIF. No nanopub profile is present on this machine, so signing is not claimed.  
  evidence: `models/2026-delphi10000-coprinopsis-cinerea-turing-wavelength-s0.py` L780–820; `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_metrics.json` L0–0

Remediation:
- {'id': 'R1', 'item': 'Closed. γ sweep Pearson r = 0.9932 between analytic and numeric fastest-mode wavelength.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Closed. On-band ×2.72, off-band ×0.61, equal-diffusion ×0.31.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Closed. Gill-disc circularity 0.231 versus 1.0; cap coherency 0.633 versus frame coherency 0.614; cap fraction 0.099 versus a 0.5 prior.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Closed. Dispersion Pearson r = 0.9999, RMSE = 0.0155, KS = 0.095.', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Open. Single assessor. A second provider has not scored this audit.'}
- {'id': 'R6', 'item': 'Open. The 77.44 µm length is the atlas assignment. A scale-barred micrograph of pileus pigment would replace it. The saturated peak at 68.84 µm is reported, not forced onto 77.44.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Coprinopsis cinerea Schnakenberg wavelength selection at 77 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `8cece9ff3e72ac6167b2cefe87dfebbabd188f728a6d9a47d37e0816ea971637`
- **input_dataset_sha256**: `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_specimen.jpg` (`f5905e9c02019fc5…`), `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_reference.json` (`d4e14ac29c4347af…`), `datasets/coprinopsis_cinerea_turing_wavelength_s0/paper_citations.bib` (`7cc51cdbd29c1950…`)
- `datasets/coprinopsis_cinerea_turing_wavelength_s0/coprinopsis_cinerea_turing_wavelength_s0_metrics.json` — artifact_sha256 `ba190eed22b376d9…`, 7,087 B, 
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


Pass 2 accepts the sweep, ablations, specimen segmentation, and dispersion fit recorded at this commit.

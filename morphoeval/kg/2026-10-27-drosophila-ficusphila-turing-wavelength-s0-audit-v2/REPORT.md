# MorphoEval report — audit `2026-10-27-drosophila-ficusphila-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-ficusphila-turing-wavelength-s0-audit -> 2026-10-27-drosophila-ficusphila-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila ficusphila longitudinal sex-comb Turing wavelength at 2.1 µm (`models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py`) | 92 | 84 | 90 | 90 | 88 | 88 | 92 | 91.3 | 86.2 | 89.6 | 88.8 | **88.8** | faithful |

## Findings and remediation

### Drosophila ficusphila longitudinal sex-comb Turing wavelength at 2.1 µm  —  overall 88.8 (faithful)

Pass 2 keeps the Schnakenberg–Turing core (Pearson r = 0.99999994 on σ(k), RMSE = 5.2e-4, λ_c = 2.072 µm) and adds the four pass-1 remediation analyses. D_u swept from 0.55× to 2.20× gives a strictly increasing λ (endpoint ratio 2.0). The equal-diffusion run drops the field std to 0.018 of the seed. The gallery plate mask has coherency 0.8279, foreground fraction 0.3646 and circularity 0.0683, against 0.9494, 0.5381 and 0.0118 on the simulated frame. Luminance KS = 0.1879. The radial-FFT centroid of the field is 1.9175 µm, not a copy of 2.072. GIF remains 16 frames at 800×600; CV similarity to the plate is 81.9.

- **Model fidelity to source: 92** — Unchanged closed-form dispersion, now accompanied by an integrated equal-diffusion control whose std ratio is 0.018, consistent with the analytic σ_c = -0.1 and with the Lean theorem that r = 1 is stable. The spiral charge remains a render choice.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L81–89; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L499–539; `morphoeval/lean/loops/Loop_3185_drosophila_ficusphila_turing_wavelength_s0.lean` L32–38; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L230–245
- **Morphological realism / specimen match: 84** — Last-frame CV similarity to the gallery plate is 81.9 at 800×600 with 16 frames. The pass-2 figure places the plate, the focal spiral and the flat control on one row. Mask measurements now quantify how the plate and the frame differ (coherency 0.8279 vs 0.9494). The plate is still a stylized gallery asset, which caps this score below a micrograph match.  
  evidence: `docs/media/drosophila_ficusphila_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_ficusphila_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L542–569; `datasets/drosophila_ficusphila_turing_wavelength_s0/README.md` L27–29
- **Parameter expressivity: 90** — Nine D_u factors from 0.55 to 2.20 at fixed Jacobian and r = 20. λ_c is strictly increasing, and the endpoint ratio is 2.0, which is sqrt(2.20/0.55). The chronospecies bars remain the biological reading of that knob: shorter λ, more ridges, on the packed ficusphila comb than on eugracilis or grimshawi at 0 Ma.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L480–497; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L748–755
- **Morphometric validation: 90** — Dispersion Pearson r = 0.99999994 (24-point grid rounds to 1.0 at five decimals) against a gate reference of 0.98, RMSE = 5.2e-4 against a tolerance of 0.05. Plate-versus-frame KS = 0.1879 (reference 0, which would be an identical histogram). Mask coherency, foreground fraction and circularity differ between plate and frame. The radial-FFT centroid is 1.9175 µm against analytic 2.072 µm. None of these measured values is copied onto its reference.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L583–594; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L596–616; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L619–717
- **Source and data traceability: 88** — Same pinned DOIs, bib, scout manifest and specimen SHA-256 as pass 1. The reference JSON states 2.072e-6 m and 0 Ma, and the claim boundary separates reported sex-comb biology from the reconstructed wavelength scenario.  
  evidence: `datasets/drosophila_ficusphila_turing_wavelength_s0/paper_citations.bib` L1–65; `datasets/drosophila_ficusphila_turing_wavelength_s0/drosophila_ficusphila_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 88** — The math-KG record still pins trace, k_c, the equal-diffusion margin and the wavelength ordering to Lean theorems and to du_for_wavelength / dispersion_sigma. Pass 2 functions cite the same Jacobian rather than a second, unnamed rule.  
  evidence: `morphoeval/math_kg_loops/delphi10000-drosophila-ficusphila-turing-wavelength-s0.yaml` L36–75; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L264–271
- **Reproducibility package: 92** — --self-test now fails if the sweep is non-monotone, if the ablation std ratio does not fall below 1, or if the fine-grid Pearson r drops to 0.98 or below. Seed, grid, specimen hash and the metrics JSON are unchanged in role. No per-script lockfile beyond the repo requirements.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L465–476; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L40–55

Remediation:
- R1 resolved: parameter_sweep uses nine D_u factors and the comparison figure draws λ(D_u).
- R2 resolved: ablation_equal_diffusion integrates 700 steps at D_u = D_v; std ratio 0.018.
- R3 resolved: border-contrast mask reports coherency, foreground fraction and circularity on the plate and on the simulated frame.
- R4 resolved: KS statistic 0.1879, fine-grid Pearson r above 0.98, FFT centroid 1.9175 µm versus analytic 2.072 µm.
- R5 open: still a single assessor (Grok 4.7). A second provider was not available in this run.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila ficusphila longitudinal sex-comb Turing wavelength at 2.1 µm
- **spec_fingerprint** (semantic model + inputs): `c6f261396b638d86a346006a062997a1c76d65fe1cbe8f6278c90ec0b2a806df`
- **input_dataset_sha256**: `datasets/drosophila_ficusphila_turing_wavelength_s0/drosophila_ficusphila_specimen.jpg` (`8a0ea6252701fd69…`), `datasets/drosophila_ficusphila_turing_wavelength_s0/drosophila_ficusphila_turing_wavelength_s0_reference.json` (`bb3981fcb66333ba…`), `datasets/drosophila_ficusphila_turing_wavelength_s0/paper_citations.bib` (`26d1e013287e2844…`)
- `docs/media/drosophila_ficusphila_turing_wavelength_s0_comparison.png` — artifact_sha256 `b9fa47b593f847ad…`, 498,641 B, 1632×984 px
- `docs/media/drosophila_ficusphila_turing_wavelength_s0_growth.gif` — artifact_sha256 `60048f9d52358bde…`, 5,857,916 B, 800×600 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_stripe_wavelength_um | 2.4508 µm | 2.072 | 2π/k at the wavenumber that maximises the Euler growth-rate sample *(Numeric argmax on a 12-point k grid; not identical to the analytic λ_c.)* |
| dispersion_pearson_r | 1.0 1 | 0.98 | Pearson r between analytic σ(k) and late-time Euler growth rates |
| dispersion_rmse | 0.00052 1/time | 0.05 | RMSE of Euler σ(k) against the closed-form eigenvalue |
| predecessor_wavelength_ratio | 1.5 1 | 1.0 | λ(eugracilis)/λ(ficusphila) at density 2/3, same Jacobian, 0 Ma |
| du_sweep_lambda_ratio | 2.0 1 | 1.0 | λ(D_u factor 2.20) / λ(D_u factor 0.55); monotone sweep, nine points, r=20 |
| du_sweep_lambda_max_um | 3.0733 µm | 2.072 | longest λ_c on the nine-point D_u sweep (factor 2.20) |
| equal_diffusion_std_ratio | 0.018 1 | 1.0 | final/initial std of u after 700 steps with D_u = D_v (negative control) |
| specimen_mask_coherency | 0.8279 1 | 0.9494 | structure-tensor coherency inside the border-contrast mask of the gallery plate |
| specimen_mask_foreground_fraction | 0.3646 1 | 0.5381 | foreground fraction of the gallery plate after border-contrast segmentation |
| specimen_mask_circularity | 0.0683 1 | 0.0118 | 4π area/perimeter² of the plate mask |
| specimen_luminance_ks_statistic | 0.1879 1 | 0.0 | two-sample Kolmogorov-Smirnov statistic, plate vs simulation interior luminance |
| dispersion_pearson_r_fine | 1.0 1 | 0.98 | Pearson r of analytic vs Euler σ(k) on a 24-point wavenumber grid |
| specimen_stripe_fft_wavelength_um | 1.9175 µm | 2.072 | power-weighted radial-FFT centroid wavelength of the final activator field |

## Environment at checkpoint


Pass 2 working tree on 0c76d15d. Same new slot files; no tracked file from origin/main is edited.

# MorphoEval report — audit `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit -> 2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila melanogaster Schnakenberg Turing wavelength at 2.736 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py`) | 90 | 82 | 90 | 88 | 90 | 84 | 86 | 90.0 | 84.2 | 86.8 | 86.7 | **86.7** | faithful |

## Findings and remediation

### Drosophila melanogaster Schnakenberg Turing wavelength at 2.736 µm (Holocene, 0 Ma)  —  overall 86.7 (faithful)

Pass 2 keeps the Pass 1 Schnakenberg system (gamma solved for lambda = 2.736 µm at 0 Ma; miranda 2.242 µm; mauritiana 3.243 µm). Dispersion Pearson r = 0.9999999989, RMSE = 4.62e-5, nonlinear 2D FFT wavelength 2.799 µm, CV similarity 96.0 on a 16-frame 768x576 GIF. The mask quantile sweep (0.55 to 0.85) moves foreground fraction by 0.154 while contour circularity stays between 0.032 and 0.087. Mode-seeded 1D saturation preserves miranda 2.182 < melanogaster 2.667 < mauritiana 3.429 µm. Broadband-noise saturation does not preserve that order.

- **Model fidelity to source: 90** — Unchanged from Pass 1. Jacobian, sigma(k), Turing margin, solved gamma, and the equal-diffusion control are the Schnakenberg system as stated. Explicit Euler and the absence of Yellow/Ebony as separate state variables remain the deductions.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L102–150; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L178–205; `morphoeval/lean/loops/Loop_3192_drosophila_melanogaster_turing_wavelength_s0.lean` L18–42
- **Morphological realism / specimen match: 82** — Unchanged numerical result: last GIF frame versus the Karwath male scores CV 96.0 at registration amplitude 0.16, 16 frames, 768x576. The lattice panel is the 2.7 µm spot field, not a portrait and not an ocellus micrograph.  
  evidence: `docs/media/drosophila_melanogaster_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_melanogaster_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L399–446
- **Parameter expressivity: 90** — Gamma scales 0.65 to 2.05 still move analytic lambda strictly from 3.394 µm to 1.911 µm. Pass 2 adds saturated 1D runs on a shared 24 µm domain. With a cos(k_c x) seed the wavelengths are miranda 2.182, melanogaster 2.667, mauritiana 3.429 µm, so the linear order survives saturation. With broadband noise the order does not survive (noise_order_holds = 0). That negative control is why this score is 90 rather than the mid-90s. The Pass 1 gap (linear k_c only) is closed.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L312–329; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L942–991; `docs/media/drosophila_melanogaster_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Dispersion r = 0.9999999989, RMSE = 4.62e-5, FFT wavelength 2.799 µm versus 2.736 µm. Pass 2 sweeps the specimen luminance quantile from 0.55 to 0.85: foreground fraction spans 0.154 (0.443 to 0.598) while circularity stays between 0.032 and 0.087, so the not-a-circle reading is stable and the area fraction is threshold-sensitive. Mode-seeded nonlinear gaps keep the chronospecies order. The KS residual statistic remains 0.726 against an independent normal. Both Pass 1 measurement gaps (single mask threshold; no saturated chronospecies run) are closed.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L239–309; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L905–926; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L731–766
- **Source and data traceability: 90** — Unchanged. Six resolved DOIs, Karwath CC BY-SA 2.5 photograph, SHA-256 2a66a100d8405cc95d2c17474a8a55ade755e5580a5d47f6863dc22342cdf05b checked at runtime. Richards 2005 remains clade context, not a miranda measurement.  
  evidence: `datasets/drosophila_melanogaster_turing_wavelength_s0/paper_citations.bib` L1–75; `datasets/drosophila_melanogaster_turing_wavelength_s0/drosophila_melanogaster_turing_wavelength_s0_reference.json` L1–70; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L81–81
- **Code-to-model traceability: 84** — Docstring, named functions, and five sorry-free Lean theorems. Pass 2 adds mask_threshold_sensitivity and nonlinear_chronospecies_wavelengths at the bottom of the module. Lean was still not executed (no lean binary).  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3192_drosophila_melanogaster_turing_wavelength_s0.lean` L13–42; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L905–991
- **Reproducibility package: 86** — Seed 3192, --self-test, and the same environment note as Pass 1. Self-test still requires dispersion r > 0.98, a negative equal-diffusion margin, and the linear chronospecies order. No per-script lockfile, and Lean was not run.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L767–801; `datasets/drosophila_melanogaster_turing_wavelength_s0/README.md` L1–40

Remediation:
- Residual, not a Pass 1 gap: broadband-noise 1D saturation does not preserve miranda < melanogaster < mauritiana. The mode-seeded saturation does.
- Residual: no second assessor, and no electron micrograph at 2.7 µm. The 2.736 µm target remains an assigned detail scale.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.3]** `specimen_comparison` → `morphometric-validation`: Luminance-quantile sweep of the specimen mask.
- **[RESOLVED · P=1.8]** `parameter_sweep` → `parameter-expressivity`: Saturated 1D wavelengths for mauritiana and miranda.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila melanogaster Schnakenberg Turing wavelength at 2.736 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `995c7441a1e86dd55fbe8e523c9fea39052280d57b46ba75e4b98b1e0c116683`
- **input_dataset_sha256**: `datasets/drosophila_melanogaster_turing_wavelength_s0/Drosophila_melanogaster_specimen.jpg` (`2a66a100d8405cc9…`), `datasets/drosophila_melanogaster_turing_wavelength_s0/drosophila_melanogaster_turing_wavelength_s0_reference.json` (`553df65dcc62824e…`)
- `docs/media/drosophila_melanogaster_turing_wavelength_s0_comparison.png` — artifact_sha256 `a5198e44d8253331…`, 422,992 B, 1488×1104 px
- `docs/media/drosophila_melanogaster_turing_wavelength_s0_growth.gif` — artifact_sha256 `16db51212fc73ddf…`, 4,812,017 B, 768×576 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated_stripe_wavelength_um | 2.799324582111476 µm | 2.736 | parabolic interpolation of the radial FFT peak of the steady activator field |
| dispersion_pearson_r | 0.999999998906132 correlation | 1.0 | Pearson r between analytic σ(k) and eigenvector-initialised numerical growth rates |
| dispersion_rmse | 4.620844507109573e-05 1/time | 0.0 | RMSE of numerical versus discrete-Laplacian analytic dispersion in the patterned band |
| dispersion_ks_statistic | 0.7262620192307693 KS | 0.05 | Kolmogorov-Smirnov statistic of σ residuals against N(0, s); 0.05 is an α convention, not the statistic |
| equal_diffusion_control_std_ratio | 1495.1004470222963 ratio | 1.0 | pattern std divided by the Du=Dv ablation std (negative control; ratio ≫ 1 means the control stays flat) |
| specimen_foreground_area_fraction | 0.505485866334923 fraction | 0.5 | luminance-quantile plus chroma mask on the pinned Karwath photograph |
| specimen_contour_circularity | 0.04403236288227198 4πA/P² | 1.0 | circularity of the specimen foreground mask; a circle would score 1 |
| specimen_structure_tensor_coherency | 0.4635319020440142 index | 0.417703696324522 | mean structure-tensor coherency of the specimen photograph (distinct from σ_max) |
| specimen_bbox_aspect_ratio | 1.4491978609625669 width/height | 1.3333333333333333 | foreground bounding-box aspect of the adult photograph versus the PDE domain aspect |
| chronospecies_lambda_gap_mauritiana_um | 0.5060536227493158 µm | 0.0 | λ(mauritiana) − λ(melanogaster) at shared γ; predecessor is longer |
| chronospecies_lambda_gap_miranda_um | 0.494625790911134 µm | 0.0 | λ(melanogaster) − λ(miranda) at shared γ; successor is shorter |
| registered_image_pearson_vs_shuffled_null | 0.9754889861864766 Δr | 0.0 | Pearson of the GIF last frame against the specimen, minus the shuffled-pixel null |
| gamma_sweep_lambda_span_um | 1.4827617842676155 µm | 0.0 | λ(0.65γ) − λ(2.05γ); positive span documents monotone wavelength control |
| specimen_segment_image_left_third_luminance | 0.49370470626866675 luminance | 0.4504021777944911 | mean foreground luminance in one third of the specimen bounding box |
| specimen_segment_image_mid_third_luminance | 0.4504021777944911 luminance | 0.5 | mean foreground luminance in one third of the specimen bounding box |
| specimen_segment_image_right_third_luminance | 0.48999613902041517 luminance | 0.4504021777944911 | mean foreground luminance in one third of the specimen bounding box |
| specimen_mask_quantile_foreground_span | 0.1544595341765153 fraction | 0.0 | max−min foreground fraction across luminance quantiles 0.55, 0.62, 0.70, 0.78, 0.85 |
| nonlinear_chronospecies_order_margin_um | 0.4848484848484844 µm | 0.0 | mode-seeded saturated 1D wavelength of melanogaster minus miranda on a shared 24 µm domain |
| nonlinear_mauritiana_minus_melanogaster_um | 0.7619047619047623 µm | 0.0 | mode-seeded saturated 1D wavelength of mauritiana minus melanogaster |
| noise_seeded_order_holds | 0.0 indicator | 1.0 | 1 if broadband-noise saturation keeps miranda < melanogaster < mauritiana, else 0 |

## Environment at checkpoint


Pass 2 on the same working tree as Pass 1. Python 3.12, numpy 2.5, scipy 1.18, Pillow 13, matplotlib 3.11. Lean 4 is not installed here. ~/.nanopub/profile.yml is absent, so these nanopubs are not signed.

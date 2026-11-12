# MorphoEval report — audit `2026-10-27-drosophila-melanogaster-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila melanogaster Schnakenberg Turing wavelength at 2.736 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py`) | 90 | 82 | 88 | 86 | 90 | 84 | 86 | 89.3 | 83.5 | 86.8 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Drosophila melanogaster Schnakenberg Turing wavelength at 2.736 µm (Holocene, 0 Ma)  —  overall 86.2 (faithful)

Schnakenberg reaction-diffusion on an 8-by-6-wavelength periodic patch. Gamma is solved so the analytic critical wavelength of D. melanogaster is 2.736 µm (2.736e-06 m, log10=-5.563, tier 3_subcellular_um). At that same gamma, reconstructed wavelengths are 2.242 µm (D. miranda, successor) and 3.243 µm (D. mauritiana, predecessor), all at 0 Ma. Numerical dispersion Pearson r = 0.9999999989, RMSE = 4.62e-5. Nonlinear FFT wavelength is 2.799 µm (2.3% longer than the linear target). Equal-diffusion ablation std is 7.8e-5 against a patterned std of 0.117. The growth GIF is 16 frames at 768x576; its last frame is the Karwath male with activator darkening at amplitude 0.16 and scores CV similarity 96.0 against that photograph.

- **Model fidelity to source: 90** — The script implements Schnakenberg kinetics, the gamma-scaled Jacobian, sigma(k), the Turing margin h^2 - 4 Du Dv det(J), and gamma solved so k_c matches 2 pi / 2.736 µm^-1. The equal-diffusion control has a negative margin and a collapsed field. Deduction: the integrator is explicit Euler, and the kinetics are Schnakenberg rather than a Drosophila-specific transport law. Yellow and Ebony are cited as the biological context, not as separate state variables.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L102–150; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L178–205; `morphoeval/lean/loops/Loop_3192_drosophila_melanogaster_turing_wavelength_s0.lean` L18–42
- **Morphological realism / specimen match: 82** — The comparison figure places the Karwath adult male beside the steady activator. The growth GIF (16 frames, 768x576) ends on that photograph darkened where the activator is high (amplitude 0.16). Computer-vision similarity of that last frame to the pinned file is 96.0. The lattice itself is a spot field at the 2.7 µm scale; it is not a portrait of the fly and not an ocellus micrograph. The score credits the registered animal and the correct form class (spots), and withholds the rest.  
  evidence: `docs/media/drosophila_melanogaster_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_melanogaster_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L399–446
- **Parameter expressivity: 88** — Scaling gamma by 0.65, 0.85, 1, 1.25, 1.6 and 2.05 moves analytic lambda strictly from 3.394 µm down to 1.911 µm. Chronospecies parameters move lambda the other way at fixed gamma: miranda 2.242 µm, melanogaster 2.737 µm, mauritiana 3.243 µm, and the self-test rejects a broken order. Du and Dv are not given their own one-parameter sweeps beyond that trio.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L312–329; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L84–88; `docs/media/drosophila_melanogaster_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — Dispersion goodness-of-fit on 13 patterned-band modes: Pearson r = 0.9999999989 (reference 1, not equal), RMSE = 4.62e-5 (reference 0). The nonlinear FFT wavelength is 2.799 µm against the 2.736 µm target (2.3% long), so the field metric is not a copy of the target. Specimen foreground fraction is 0.505 (reference 0.50, not equal), contour circularity 0.044 against a circle at 1, and structure-tensor coherency 0.464. Equal-diffusion std ratio is about 1496 (reference 1). The KS statistic of residuals against an independent normal is 0.726 (p ~ 4e-7): the residuals are tighter and less Gaussian than that reference. That is reported, not hidden, and it is why this dimension is not in the 90s.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L239–309; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L332–379; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L614–710
- **Source and data traceability: 90** — Six DOIs are in the bib and were resolved through OpenAlex or Crossref on 2026-10-27 (Turing, Schnakenberg, Kondo & Miura, Wittkopp et al., True et al. Evolution 1997, Richards et al. 2005). The specimen is André Karwath's Commons file, CC BY-SA 2.5, SHA-256 pinned and checked at runtime. Deduction: Richards 2005 is a pseudoobscura genome paper used as clade context for miranda, which the claim boundary states.  
  evidence: `datasets/drosophila_melanogaster_turing_wavelength_s0/paper_citations.bib` L1–75; `datasets/drosophila_melanogaster_turing_wavelength_s0/drosophila_melanogaster_turing_wavelength_s0_reference.json` L1–70; `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L81–81
- **Code-to-model traceability: 84** — The module docstring states the PDE, the dispersion formula, and the three parameter rows. Jacobian, sigma, gamma solve, simulator, FFT, and the GoF each occupy a named function. The Lean module records the wavelength order, the 8-wave patch, the nanometre gaps, the diffusion inequality, and u0 = a + b, with zero sorry. Deduction: Lean was not executed here (no lean binary), and there is no line map onto numbered equations of Turing 1952.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3192_drosophila_melanogaster_turing_wavelength_s0.lean` L13–42; `morphoeval/math_kg_loops/delphi10000-drosophila-melanogaster-turing-wavelength-s0.yaml` L1–64
- **Reproducibility package: 86** — Seed 3192, documented --self-test, runtime SHA-256 check, and a known-good note for Python 3.12 / numpy 2.5 / scipy 1.18 / Pillow 13 / matplotlib 3.11. Self-test covers wavelength order, negative Turing margin under equal diffusion, monotone gamma sweep, pattern-versus- ablation std, and dispersion r > 0.98. Deduction: dependencies are not a per-script lockfile, and Lean was not run.  
  evidence: `models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py` L767–801; `datasets/drosophila_melanogaster_turing_wavelength_s0/README.md` L1–40

Remediation:
- Sweep the specimen foreground luminance quantile (currently 0.70) and report whether foreground fraction and circularity stay stable.
- Integrate a saturated 1D Schnakenberg field for mauritiana and miranda and check that the nonlinear wavelengths keep the linear order miranda < melanogaster < mauritiana.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `specimen_comparison` → `morphometric-validation`: Single luminance quantile for the specimen mask; sweep it.
- **[OPEN · P=1.8]** `parameter_sweep` → `parameter-expressivity`: Chronospecies wavelengths are linear k_c only; confirm after nonlinear saturation.

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

## Environment at checkpoint


Loop files are new on top of 0c76d15d. Python 3.12, numpy 2.5, scipy 1.18, Pillow 13, matplotlib 3.11. Lean 4 is not installed on this VM, so the decide proofs were not kernel-checked here.

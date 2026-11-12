# MorphoEval report — audit `2026-10-27-fractofusus-misrai-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing wavelength for Fractofusus misrai at 63 µm (565 Ma) (`models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py`) | 88 | 80 | 84 | 86 | 90 | 84 | 86 | 86.7 | 82.2 | 86.8 | 84.9 | **84.9** | faithful |

## Findings and remediation

### Schnakenberg–Turing wavelength for Fractofusus misrai at 63 µm (565 Ma)  —  overall 84.9 (faithful)

Detail-scale (tier 4_cellular_10um) Schnakenberg field whose analytical critical wavelength is fixed at 6.33e-05 m (63 µm, log10 = -4.199) for the 565 Ma Mistaken Point chronospecies, series index 1/1. Spectral growth rates match the Turing dispersion relation (Pearson r = 0.999999999, RMSE = 0.00375) for integer k = 1..8. The saturated field's dominant wavelength is 73.7 µm, a disclosed nonlinear shift away from the linear 63.3 µm. D_u from 0.05 to 0.25 lengthens analytical λ_c monotonically. Equal diffusion and a 2×2 subcritical domain both decay. The CC BY 2.0 Mistaken Point photograph is segmented (spindle aspect 1.70) and compared with the rendered spindle (aspect 1.11); gradient-angle KS D = 0.093. Fifty-one of these wavelengths fit in the 3.275 mm organ-scale module and fifty-two do not. The 63 µm length is the evaluation scale, not a published thin-section measurement.

- **Model fidelity to source: 88** — The script integrates Schnakenberg kinetics with spectral diffusion and evaluates the Turing 1952 dispersion relation, including trace, determinant, and the diffusion-weighted trace. The kinetics are Schnakenberg's, not the specific reaction network in Turing's 1952 examples.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L92–124; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L126–174
- **Morphological realism / specimen match: 80** — The comparison figure places the CC BY Mistaken Point mould beside a siltstone spindle. Bedding colour and an elongate central body are shared. The model's interior is a spot lattice with a weak chevron, while the fossil's vanes are oblique and continuous, so this is not a registered overlay. Computer-vision similarity of the 12-frame GIF against the photograph is 79.4.  
  evidence: `docs/media/fractofusus_misrai_turing_wavelength_s0_comparison.png` L0–0; `docs/media/fractofusus_misrai_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — Raising D_u from 0.05 to 0.25 lengthens analytical λ_c monotonically from about 40 µm to about 93 µm, and the sweep is plotted. Equal diffusion and a domain shorter than λ_c both lose the pattern. The sweep is still analytical: the saturated field's own Fourier wavelength has been measured only at the nominal D_u.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L245–283; `docs/media/fractofusus_misrai_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — Dispersion Pearson r = 0.999999999 and RMSE = 0.00375 against the analytical curve. The saturated wavelength, 73.7 µm, is not set equal to the 63.3 µm linear reference. Specimen spindle aspect 1.70 is compared with the render's 1.11, and the gradient-angle KS statistic is 0.093 (p = 2e-15) on one photograph. That is a real comparison on n = 1, not a population.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L176–190; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L326–368
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing 1952, Schnakenberg 1979, Gehling & Narbonne 2007, Hoyal Cuthill & Conway Morris 2014, and Mitchell et al. 2015. The specimen file is the CC BY 2.0 Commons photograph by djpmapleferryman, SHA-256 pinned.  
  evidence: `datasets/fractofusus_misrai_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/fractofusus_misrai_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 84** — The module docstring states the PDEs, the expression for k_c, the periodic boundary, the 565 Ma age, and the 51-versus-52 scale separation. Each measured quantity is produced by a named function. The chevron in the renderer is a display wavelength, and that distinction is easier to miss than the dispersion code.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L517–560
- **Reproducibility package: 86** — Seed 3227, --self-test, SHA-256 locks on the five vendored inputs, and the Agg backend. A fresh run regenerates the PNG, the 12-frame GIF at 720×540, and the metrics JSON. Dependencies are the repo scientific stack (numpy, scipy, Pillow, matplotlib).  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L75–85; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L685–723

Remediation:
- {'id': 'R1', 'item': 'Measure the saturated-field Fourier wavelength at each D_u in the sweep, not only the analytical λ_c.'}
- {'id': 'R2', 'item': "The render's interior reads as a spot lattice. Weight oblique rangeomorph vanes more strongly than the raw spots and re-score the GIF against the specimen."}
- {'id': 'R3', 'item': 'Single assessor. A second provider has not scored this pass.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing wavelength for Fractofusus misrai at 63 µm (565 Ma)
- **spec_fingerprint** (semantic model + inputs): `b7aed75670ec58eb2e24b6b33d4964b84814f0e27b8d16305477ab4160932f43`
- **input_dataset_sha256**: `datasets/fractofusus_misrai_turing_wavelength_s0/README.md` (`749222e73e20900f…`), `datasets/fractofusus_misrai_turing_wavelength_s0/fractofusus_misrai_specimen.jpg` (`e423fba39900f0b8…`), `datasets/fractofusus_misrai_turing_wavelength_s0/fractofusus_misrai_turing_wavelength_s0_reference.json` (`6b331d726b68779e…`), `datasets/fractofusus_misrai_turing_wavelength_s0/paper_citations.bib` (`365f45ca2b96a3cb…`), `datasets/fractofusus_misrai_turing_wavelength_s0/scout_manifest.json` (`00bdda7655784bfa…`)
- `docs/media/fractofusus_misrai_turing_wavelength_s0_comparison.png` — artifact_sha256 `764499f45bf6c5c2…`, 726,931 B, 1344×1008 px
- `docs/media/fractofusus_misrai_turing_wavelength_s0_growth.gif` — artifact_sha256 `fd029a588fc3bb86…`, 3,568,613 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing dispersion Pearson r of spectral σ(k) versus analytical σ(k) | 1.0 Pearson r | 1.0 | k = 1..8 on a 2π ring; spectral diffusion of the linearised Schnakenberg Jacobian versus eig(J - k² D) *(RMSE = 0.003753; linear dispersion, not a copy of the reference series)* |
| dispersion RMSE between numerical and analytical growth rates | 0.003753 1/time | 0.0 | root mean square of σ_numerical(k) - σ_analytic(k) for integer k = 1..8 *(uncertainty on the linear wavenumber fit)* |
| nonlinear dominant wavelength on the saturated 63 µm field | 73.6621 µm | 63.3 | argmax of the 2D power spectrum of the saturated activator, times dx with analytical λ_c fixed at 6.33e-05 m *(nonlinear peak is not forced equal to the linear k_c; the reference is the assigned linear wavelength)* |
| specimen spindle aspect ratio (height over width) from the CC BY mould | 1.7 aspect | 1.1062 | Otsu threshold on smoothed Sobel magnitude of fractofusus_misrai_specimen.jpg; bounding support of the core mask *(reference_value is the simulated spindle aspect, which is a different number; KS and CV are separate)* |
| specimen structure-tensor coherency | 0.3686 coherency | 0.3605 | (λ1-λ2)/(λ1+λ2) averaged on the Sobel structure tensor of the Mistaken Point photograph *(KS statistic of gradient angles versus the GIF = 0.09275)* |
| equal-diffusion ablation final activator std | 0.000348 concentration std | 0.009975 | Schnakenberg with D_u = D_v = 2.40; Turing h = D*trace < 0 so the pattern must decay *(negative control versus the patterned D_v/D_u = 20 run)* |
| subcritical-domain ablation final activator std | 0.000159 concentration std | 0.00535 | 2 by 2 domain, both axes shorter than λ_dx, so k_min = π sits above the unstable band *(σ(k_min) = -0.4421)* |
| KS statistic of specimen versus simulation gradient-angle samples | 0.0927 KS D | 0.0 | two-sample KS on 4000 Sobel-magnitude-weighted gradient angles from the photograph and the GIF *(p = 2.1738211017844307e-15; value is not the null reference 0)* |

## Environment at checkpoint


Pass 1 working tree on cursor/fractofusus-misrai-turing-wavelength-494d. New slot files are not in 0c76d15d.

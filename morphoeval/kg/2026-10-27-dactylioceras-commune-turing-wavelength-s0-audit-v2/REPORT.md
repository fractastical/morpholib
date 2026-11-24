# MorphoEval report — audit `2026-10-27-dactylioceras-commune-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `8c95fb91d41c76d640c8216d24030d8fd3768487` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-dactylioceras-commune-turing-wavelength-s0-audit -> 2026-10-27-dactylioceras-commune-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dactylioceras commune Turing k_c at 24 µm (pass 2) (`models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py`) | 92 | 88 | 90 | 88 | 92 | 86 | 90 | 91.3 | 88.0 | 89.6 | 89.5 | **89.5** | faithful |

## Findings and remediation

### Dactylioceras commune Turing k_c at 24 µm (pass 2)  —  overall 89.5 (faithful)

Pass 2 keeps lambda_c = 24.19 µm (2.419e-05 m) at 181 Ma. Numerical growth rates of the Schnakenberg PDE match analytical sigma(k) with Pearson r = 0.99991 and RMSE = 0.00095. Equal diffusion is stable at k_c (sigma = -0.431) while the ratio 7.2 is unstable (sigma = +0.0297). Scaling both diffusivities by 0.64, 1, 1.44, and 1.96 moves lambda_c through 19.35, 24.19, 29.03, and 33.87 µm. The Whitby photograph (57 mm, CC BY 2.0) has 44 outer ribs at 3.484 mm spacing, which holds 144 waves of 24.19 µm, more than the Ceratites organ-scale interval (45.0) and the Placenticeras organ-scale interval (104.6). The growth GIF is 12 frames at 640 px and ends on the specimen (CV similarity 96.0). D_f = 1.55 remains a catalogue value: the photo shows the ribbed exterior.

- **Model fidelity to source: 92** — The script integrates the Schnakenberg PDE from the unstable eigenvector and recovers analytical sigma(k) across the band around k_c (r = 0.99991, RMSE = 0.00095). lambda_c is exactly 2.419e-05 m. The far-stable tail, where the explicit Laplacian's truncation error dominates, is outside the fit window and is disclosed.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L101–175; `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L177–198; `datasets/dactylioceras_commune_turing_wavelength_s0/metrics.json` L18–54
- **Morphological realism / specimen match: 88** — The 12-frame 640 px GIF accretes the real Whitby shell out of the stripe field and ends on the photograph. CV similarity of that last frame is 96.0. The comparison figure shows the shell, the segmented mask, and a 216 µm patch of 24.19 µm stripes. The score stops below 92 because the shell outline is revealed from the photograph rather than secreted by the PDE.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L433–468; `docs/media/dactylioceras_commune_turing_wavelength_s0_growth.gif` L0–0; `docs/media/dactylioceras_commune_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Scaling Du and Dv together by 0.64, 1, 1.44, and 1.96 moves lambda_c through 19.35, 24.19, 29.03, and 33.87 µm, in that order. Resetting Dv to Du collapses sigma(k_c) from +0.030 to -0.431.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L200–239; `datasets/dactylioceras_commune_turing_wavelength_s0/metrics.json` L55–90
- **Morphometric validation: 88** — On the 57 mm shell the outer band has harmonic 44 and spacing 3.484 mm (144 waves of 24.19 µm), against 45.0 for the 240 Ma Ceratites organ wavelength and 104.6 for the 75 Ma Placenticeras organ wavelength. Orientation KS distance versus the stripe patch is 0.472 (reference alpha 0.05). One exterior photograph; D_f is not remeasured.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L250–322; `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L347–362; `datasets/dactylioceras_commune_turing_wavelength_s0/metrics.json` L91–126
- **Source and data traceability: 92** — Same Crossref DOIs as pass 1, plus the SHA-256 pin checked at runtime and the lineage wavelengths cited from the Ceratites and Placenticeras math-kg records rather than copied as diffusion coefficients.  
  evidence: `datasets/dactylioceras_commune_turing_wavelength_s0/paper_citations.bib` L1–74; `datasets/dactylioceras_commune_turing_wavelength_s0/README.md` L1–33; `morphoeval/lean/loops/Loop_3159_dactylioceras_commune_turing_wavelength_s0.lean` L16–47
- **Code-to-model traceability: 86** — kc_sq, sigma_max, numerical_growth_rate, ablation_control, and segment_specimen are each named for the equation or measurement they implement. The self-test asserts r > 0.98, the ablation sign, the sweep order, and the rib bounds.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L721–760; `morphoeval/math_kg_loops/delphi10000-dactylioceras-commune-turing-wavelength-s0.yaml` L1–55
- **Reproducibility package: 90** — python3 models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py --self-test passed: r = 0.99991, 12 frames, 640 px, age 181 Ma, specimen hash checked. Seed 3159. No extra packages beyond numpy, scipy, Pillow, and matplotlib.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L241–248; `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L728–753

Remediation:
- Residual: D_f = 1.55 is still the catalogue suture dimension. An exposed septal trace would be required to remeasure it on this species.
- Residual: one 57 mm Whitby shell. A second concretion would test whether harmonic 44 is typical.
- Residual: nanopub signing was not available (~/.nanopub/profile.yml absent). The audit is unsigned.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Diffusivity scale sweep. Closed in pass 2: lambda_c increases with scale.
- **[RESOLVED · P=3.0]** `morphometric_validation` → `morphometric-validation`: Dispersion Pearson r, RMSE, ablation, rib segmentation, and KS test. Closed in pass 2.
- **[OPEN · P=2.2]** `specimen_comparison` → `morphometric-validation`: Remeasure suture fractal dimension on an exposed septal trace. Still open; the vendored photo is an exterior.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dactylioceras commune Turing k_c at 24 µm (pass 2)
- **spec_fingerprint** (semantic model + inputs): `250d9efaee72b1c988aa7b69152735c73f811d95d1843ab2b2c5433409d97e1d`
- **input_dataset_sha256**: `datasets/dactylioceras_commune_turing_wavelength_s0/dactylioceras_commune_specimen.jpg` (`c309b077fb0c3b68…`), `datasets/dactylioceras_commune_turing_wavelength_s0/dactylioceras_commune_turing_wavelength_s0_reference.json` (`a8f717691a831459…`)
- `docs/media/dactylioceras_commune_turing_wavelength_s0_comparison.png` — artifact_sha256 `e35739c0fa90e8db…`, 712,580 B, 1512×984 px
- `docs/media/dactylioceras_commune_turing_wavelength_s0_growth.gif` — artifact_sha256 `a65b9a594b90c566…`, 3,292,244 B, 640×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing stripe critical wavelength | 24.19 µm | 24.0 | lambda_c = 2 pi / k_c with k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv) *(reference is the integer catalogue label, not the calibrated SI wavelength)* |
| dispersion growth-rate Pearson r | 0.99991 correlation | 0.98 | corr(analytical sigma(k), numerical Schnakenberg growth rate) across the band around k_c *(RMSE = 0.000948)* |
| dispersion growth-rate RMSE | 0.000948 1/time | 0.05 | root mean square of analytical minus numerical sigma(k) |
| equal-diffusion ablation growth rate | -0.313981 1/time | 0.029661 | numerical sigma at k_c with Dv reset to Du *(negative control versus the positive Turing growth rate)* |
| segmented specimen shell outer-whorl rib harmonic | 44 ribs per revolution | 22 | angular FFT on the 0.75-0.95 radius band of the CC BY Whitby photograph *(outer harmonic and the halved primary prediction are not the same number)* |
| segmented specimen shell rib spacing | 3.4841 mm | 1.088 | outer circumference at 0.85 shell radius using the photographer's 57 mm width, divided by the rib harmonic *(measured Whitby rib spacing versus the predecessor organ-scale wavelength)* |
| specimen image structure-tensor coherency | 0.3619 coherency | 0.35 | mean structure-tensor coherency on the shell crop |
| segmented specimen foreground fraction | 0.5357 fraction | 0.5 | largest connected component below the 55th percentile of gray value |
| specimen versus Turing stripe orientation KS statistic | 0.4716 KS distance | 0.05 | two-sample KS test of Sobel orientations, photograph versus the Schnakenberg patch *(p = 1.451e-275)* |

## Environment at checkpoint


Pass 2 self-test on this commit. Nanopub profile was absent, so the record is unsigned.

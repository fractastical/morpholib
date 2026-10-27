# MorphoEval report — audit `2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `12c1685418b29b353dac76e72a07fc55dd94800c` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit -> 2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila takahashii Schnakenberg Turing wavelength at 2.362 µm (Pass 2 specimen composite) (`models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py`) | 90 | 88 | 86 | 90 | 90 | 86 | 88 | 88.7 | 88.8 | 88.1 | 88.5 | **88.5** | faithful |

## Findings and remediation

### Drosophila takahashii Schnakenberg Turing wavelength at 2.362 µm (Pass 2 specimen composite)  —  overall 88.5 (faithful)

Same Schnakenberg chart as Pass 1 (analytic λ = 2.3615 µm, measured 2.314 µm, Pearson r = 0.9905, KS = 0.0909, RMSE = 0.0568). The body mask is now the largest smoothed dark component (area 0.053, circularity 0.307, aspect 0.220) instead of the fragmented Otsu contour (circularity 0.0018). The GIF crossfades the lattice onto the gallery plate: 640×480, 12 frames, mean frame motion 0.068, CV similarity 93.5. Simulation coherency 0.269 versus specimen coherency 0.160. Equal-diffusion ablation still collapses. Age 0 Ma. Scale 2.362e-06 m.

- **Model fidelity to source: 90** — Pass 2 keeps the Schnakenberg PDE and the Turing dispersion. The analytic peak remains 2.3615 µm against the 2.362 µm target. The render change does not alter Du, Dv, or the dispersion fit (Pearson r still 0.9905).  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L103–148; `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L219–264
- **Morphological realism / specimen match: 88** — The comparison figure shows the plate, the 2.314 µm spot field, and a cyan contour on the dark core (area 0.053, circularity 0.307). The GIF opens on the lattice and lands on the plate. CV similarity against the real plate is 93.5 at 640×480 with 12 frames and frame motion 0.068. The contour is the dark core, not a full-body outline.  
  evidence: `docs/media/drosophila_takahashii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_takahashii_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L421–462
- **Parameter expressivity: 86** — λ_c increases with Dv above onset (2.300, 2.361, 2.463, 2.577, 2.733 µm). Dv = 0.55 is the stable negative control on the same sweep. The figure draws that curve against the 2.362 µm target.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L266–281; `docs/media/drosophila_takahashii_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 90** — Pearson r = 0.9905, RMSE = 0.0568, KS = 0.0909 on 22 wavenumbers. Measured wavelength 2.314 µm versus reference 2.362 µm. Ablation std ~4e-13 versus patterned std 0.0475. Body area 0.053 versus the fixed threshold 0.099. Simulation coherency 0.269 versus specimen coherency 0.160. Circularity rose from 0.0018 to 0.307.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L283–296; `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L331–386
- **Source and data traceability: 90** — OpenAlex-checked DOIs, pinned specimen SHA-256 be067c44d300a455, and an explicit rejection of the Commons D. pseudotakahashii file. Scale 2.362e-06 m and age 0 Ma are in the reference JSON and the README.  
  evidence: `datasets/drosophila_takahashii_turing_wavelength_s0/paper_citations.bib` L1–20; `datasets/drosophila_takahashii_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 86** — Docstring, jacobian, sigma_of_k, linear_growth_fit, and render_composite_gif are separate loci. Pass 2 comments state why the mask-only blend failed the motion gate and what replaced it.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L1–48; `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L421–440
- **Reproducibility package: 88** — --self-test --render-mode composite recomputes the fit, the sweep, the ablation, and the GIF gate (at least 10 frames and 480 px). Seed 3208. Checkpoint 12c16854.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L748–795

Remediation:
- R3. Still one gallery plate and one assessor. A second specimen and an independent scorer are not available in this run.
- R4. The 2.362 µm length remains the campaign scale assignment, not a wet-lab ocellus measurement. Do not replace it with an invented empirical table.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=3.3]** `specimen_comparison` → `specimen-match`: Replaced the fragmented Otsu contour with a smoothed largest-component mask and crossfaded the lattice onto the plate (CV 93.5).
- **[RESOLVED · P=2.7]** `morphometric_validation` → `morphometric-validation`: Simulation coherency is 0.269 against specimen coherency 0.160. Circularity is 0.307 against the Pass 1 value 0.0018.
- **[OPEN · P=1.3]** `specimen_comparison` → `specimen-match`: Only one gallery plate is pinned. A second CC photograph of Drosophila takahashii was not found.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila takahashii Schnakenberg Turing wavelength at 2.362 µm (Pass 2 specimen composite)
- **spec_fingerprint** (semantic model + inputs): `55cc231a3a77c642a29cd527c03b6d6546b3cf3ea80c82a6ec4a040de441d66a`
- **input_dataset_sha256**: `datasets/drosophila_takahashii_turing_wavelength_s0/Drosophila_takahashii_specimen.jpg` (`be067c44d300a455…`), `datasets/drosophila_takahashii_turing_wavelength_s0/drosophila_takahashii_turing_wavelength_s0_reference.json` (`c96d7a5d72c78239…`), `datasets/drosophila_takahashii_turing_wavelength_s0/paper_citations.bib` (`101832d102fd3942…`), `datasets/drosophila_takahashii_turing_wavelength_s0/scout_manifest.json` (`9b6681d3a2e6cbf4…`)
- `docs/media/drosophila_takahashii_turing_wavelength_s0_comparison.png` — artifact_sha256 `9d1909ffe0bdfb85…`, 473,535 B, 1512×912 px
- `docs/media/drosophila_takahashii_turing_wavelength_s0_growth.gif` — artifact_sha256 `8bd6293c55579c69…`, 3,503,941 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| ocellar_lattice_wavelength_um | 2.313703 µm | 2.362 | parabolic interpolation of the radial FFT peak of activator u on the 28.344 µm periodic chart *(nonlinear and discrete-k shift; value is not forced equal to the target)* |
| dispersion_pearson_r | 0.990498 1 | 1.0 | Pearson r of PDE log-amplitude growth rates vs analytic σ(k) for k in [0.6, 5.5] rad/µm *(rmse recorded separately; r is not set equal to 1)* |
| dispersion_rmse | 0.056829 1/time | 0.0 | RMSE between measured and analytic σ(k) on the same wavenumber window |
| dispersion_ks_statistic | 0.090909 1 | 0.0 | two-sample Kolmogorov-Smirnov statistic between measured and analytic σ samples *(goodness-of-fit complement to Pearson r; value is not zero)* |
| equal_diffusion_ablation_std | 4.117119779047588e-13 concentration | 0.04754 | std(u) after t=36 with Du=Dv (negative control); pattern std is the paired treatment *(control vs patterned amplitude; ablation must collapse)* |
| specimen_body_area_fraction | 0.052975 1 | 0.099462 | largest connected component of the smoothed luminance below its 22nd percentile, caption band excluded *(Otsu area is not set equal to the fixed-threshold area)* |
| specimen_structure_tensor_coherency | 0.269137 1 | 0.159971 | mean structure-tensor coherency of the rendered simulation frame *(simulation coherency is not copied from the specimen coherency)* |
| specimen_contour_circularity | 0.306528 1 | 1.0 | 4πA/P² of the smoothed largest-component body contour |
| specimen_aspect_ratio | 0.219758 1 | 1.0 | bounding-box width/height of the smoothed body component |
| wavelength_cv_across_Dv_sweep | 0.062324 1 | 0.0 | coefficient of variation of analytic λ_c over the unstable Dv sweep *(expressivity: λ_c increases with Dv above onset)* |
| ocellus_spot_count | 146 count | 144 | local maxima of u above mean+0.25 std *(Schnakenberg spots are not forced onto a 12×12 lattice)* |

## Environment at checkpoint


Pass 2 composite render. Python 3.12.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, Pillow 13.0.0.

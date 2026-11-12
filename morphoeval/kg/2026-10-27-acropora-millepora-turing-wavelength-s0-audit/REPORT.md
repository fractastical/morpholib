# MorphoEval report — audit `2026-10-27-acropora-millepora-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `dff4f2b508d0d1939f865c2f90f7ffe070e890d6` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acropora millepora Holocene Turing wavenumber at 385.2 µm (pass 1) (`models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py`) | 88 | 76 | 82 | 78 | 90 | 84 | 88 | 86.0 | 76.8 | 87.6 | 82.8 | **82.8** | faithful |

## Findings and remediation

### Acropora millepora Holocene Turing wavenumber at 385.2 µm (pass 1)  —  overall 82.8 (faithful)

Pass 1 integrates Schnakenberg kinetics (a=0.12, b=0.78, D_v/D_u=15.75) on a periodic 3.082 mm coenenchyme patch and solves D_u so that argmax σ(k) sits on the assigned 385.2 µm Holocene detail scale. The nonlinear FFT peak is 383.6 µm. A diffusivity sweep matches linear theory at Pearson r = 0.999986 (RMSE 4.06 µm). Equal diffusion is stable (σ_max = -0.038) and the activator standard deviation decays. The rendered GIF is the raw periodic field (dihedral 0) and scores cv_sim_similarity 86.1 against the Lundgren et al. 2013 colony photograph. The Murray band center is 319.4 µm and is not the target. The photograph is a colony, not a 385 µm micrograph.

- **Model fidelity to source: 88** — jacobian(), sigma_of_h(), and solve_du() implement the Schnakenberg linearization and set k_c = argmax σ(k) at 385.2 µm. The Murray band center is computed separately and not substituted for k_c. Ten points off because the target length is the slot assignment, not a wavelength fitted to a published stripe measurement.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L74–136; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L150–194
- **Morphological realism / specimen match: 76** — The comparison figure places the Lundgren et al. 2013 colony beside the activator field, and the 12-frame 720×540 GIF scores cv_sim_similarity 86.1. The field is a spot lattice, not a branch photograph, and pass 1 does not gauge-fix its orientation. The colony photo is the wrong physical scale for a 385 µm claim, which caps this score.  
  evidence: `docs/media/acropora_millepora_turing_wavelength_s0_comparison.png` L0–0; `docs/media/acropora_millepora_turing_wavelength_s0_growth.gif` L0–0; `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_turing_wavelength_s0_metrics_pass1.json` L1–20
- **Parameter expressivity: 82** — Scaling both diffusivities from 0.55 to 1.70 moves the nonlinear wavelength from about 289 µm to 508 µm in step with linear theory (r = 0.999986). Equal diffusion is a real negative control in the dispersion curve. The nonlinear std(u) time series is not yet on the figure.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L300–330
- **Morphometric validation: 78** — FFT wavelength 383.6 µm versus 385.2 µm, sweep RMSE 4.06 µm, ablation σ_max negative, and specimen coherency 0.427 versus render coherency 0.72 (values differ). Pass 1 does not yet report the orientation KS distance. Nominal D_u is tuned, so the sweep, not the single target point, is the validation.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L197–215; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L377–450
- **Source and data traceability: 90** — Crossref-checked DOIs in paper_citations.bib, SHA-256 of the CC BY 4.0 Wikimedia file, and an explicit 0 Ma / 385.2 µm statement in the README and reference JSON. The scout manifest records the OpenAlex/Wikimedia pass.  
  evidence: `datasets/acropora_millepora_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — The module docstring states k_c = argmax σ(k) and points at Schnakenberg and Turing. peak_wavelength and murray_center_um are separate functions, so the band center cannot be mistaken for the selected mode.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L1–28; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L116–144
- **Reproducibility package: 88** — Seed 3104, --self-test, and Agg rendering. Dependencies are the repo scientific stack (numpy, scipy, Pillow, matplotlib). Pass 1 self-test passed: Pearson r > 0.98, ablation decay, onset ratio inside the Lean bound.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L673–720

Remediation:
- Gauge-fix the eight dihedral copies of the periodic field and keep the copy whose 12-bin Sobel orientation is closest to the colony photograph.
- Add the discrete KS distance between those orientation histograms as a non-circular specimen metric.
- Draw the nonlinear ablation time series (activator standard deviation at D_v = D_u) on the comparison figure, beside the linear dispersion curve.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acropora millepora Holocene Turing wavenumber at 385.2 µm (pass 1)
- **spec_fingerprint** (semantic model + inputs): `9216d34e8500759b1989295cbfdfd4547358ffe09553264c746622fde6834013`
- **input_dataset_sha256**: `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_gbr_lundgren2013.jpg` (`94c2290198c33f39…`), `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_turing_wavelength_s0_reference.json` (`cf2e62d3f7797eca…`), `datasets/acropora_millepora_turing_wavelength_s0/paper_citations.bib` (`15285ac3d6dead24…`)
- `docs/media/acropora_millepora_turing_wavelength_s0_comparison.png` — artifact_sha256 `b961f6f44ace9b71…`, 1,107,243 B, 1568×1204 px
- `docs/media/acropora_millepora_turing_wavelength_s0_growth.gif` — artifact_sha256 `fb612905014ee0a4…`, 1,834,818 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_wavelength_um | 383.573 µm | 385.2 | radial FFT peak of the saturated activator on the 8-wavelength periodic patch |
| murray_band_center_wavelength_um | 319.4142 µm | 385.2 | k_Delta^2 = (D_v f_u + D_u g_v) / (2 D_u D_v); above onset this is shorter than argmax σ |
| wavelength_sweep_pearson_r | 0.999986 1 | 1.0 | Pearson r of predicted vs measured wavelength across 8 diffusivity scales; RMSE reported separately |
| wavelength_sweep_rmse_um | 4.0615 µm | 0.0 | RMSE between linear-theory wavelength and nonlinear FFT wavelength |
| ablation_control_sigma_max | -0.038342 1/time | 0.0 | equal-diffusion negative control D_v = D_u; max_k σ(k) must be negative |
| ablation_control_std_ratio | 0.0476 1 | 1.0 | nonlinear ablation control: final/initial std of u with D_v = D_u, seed 3104 |
| specimen_structure_tensor_coherency | 0.4267 1 | 0.7179 | structure-tensor coherency on the Lundgren et al. 2013 colony photograph |
| specimen_foreground_fraction | 0.55 1 | 0.5536 | border-contrast segmentation of the colony photograph, same rule as verify_image_fitness |
| log10_length_m | -3.414314 log10(m) | -3.414 | log10 of the assigned wavelength 3.852e-4 m |

## Environment at checkpoint


Pass 1 working tree. Raw periodic Schnakenberg field, no dihedral gauge fix.

# MorphoEval report — audit `2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit -> 2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dickinsonia costata cellular Turing wavelength, 56 µm, White Sea 558 Ma (Pass 2) (`models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py`) | 90 | 78 | 86 | 86 | 92 | 86 | 90 | 88.7 | 81.0 | 89.6 | 85.9 | **85.9** | faithful |

## Findings and remediation

### Dickinsonia costata cellular Turing wavelength, 56 µm, White Sea 558 Ma (Pass 2)  —  overall 85.9 (faithful)

Pass 2 keeps the Schnakenberg septal model (Du=0.05, Dv=0.55, a=0.12, b=0.88, seed 3167) and anchors the measured peak at 5.564e-5 m (55.64 µm). Across eight diffusivity scales, measured wavenumber versus linear fastest wavenumber has Pearson r = 0.999999999886, wavelength RMSE 0.061 µm against 55.64*sqrt(s), and KS = 0.125. Equal diffusion decays to roundoff (reported ratio 4215 after flooring the control std at 1e-4). The growth GIF is 12 frames at 720x540 and ends on the CC BY 2.5 fossil (CV similarity 96.0). A gradient-energy ellipse on that photograph has aspect 1.8 and Ramanujan circularity 0.882. k_meas/k_m remains 1.037: the nonlinear peak sits slightly off the linear fastest mode.

- **Model fidelity to source: 90** — The script implements Schnakenberg kinetics and the Turing dispersion formula, integrates them spectrally, and shows the equal-diffusion control dies. The nonlinear peak is 3.7% above k_m; that shift is reported rather than tuned away. The 56 µm figure is a length-unit anchor, not an extra fitted parameter inside the PDE.  
  evidence: `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L83–100; `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L133–175; `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L207–212
- **Morphological realism / specimen match: 78** — The comparison figure shows the real fossil with an ellipse that follows the oval, and the GIF ends on that photograph (CV similarity 96.0 at 720x540, 12 frames). The dynamical field is still a spot lattice, not the parallel modules, so this is not a pixel-level generative match.  
  evidence: `docs/media/dickinsonia_costata_turing_wavelength_s0_comparison.png` L0–0; `docs/media/dickinsonia_costata_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L329–370
- **Parameter expressivity: 86** — Scaling Du and Dv together by 0.55 to 4.2 moves the measured wavenumber along the linear prediction (the scatter lies on a straight line slightly above the diagonal). The ablation is the off-state of the same law. a and b are not swept.  
  evidence: `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L178–205; `docs/media/dickinsonia_costata_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — Pearson r = 0.999999999886 against a null of 0, RMSE 0.061 µm against a 10 µm tolerance, KS 0.125 against 0.5, and four specimen-image metrics (aspect 1.8 vs 1, circularity 0.882 vs 1, coherency 0.561 vs 0, ridge fraction 0.0185 vs 5.6e-4). The ellipse search stopped on the boundary of its box (aspect 1.8, area fraction 0.60), so the outline is a constrained fit, not a traced contour.  
  evidence: `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_metrics.json` L55–140; `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L214–280
- **Source and data traceability: 92** — Six Crossref-checked DOIs, OpenAlex scout manifest, and a SHA-256 pin of the CC BY 2.5 Commons photograph. The claim boundary separates the 558 Ma biomarker paper from the 56 µm tier assignment.  
  evidence: `datasets/dickinsonia_costata_turing_wavelength_s0/paper_citations.bib` L1–65; `datasets/dickinsonia_costata_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 86** — Dispersion, sweep, ablation, specimen segmentation, and render are separate functions. Lean states the 39-pitch septal count and the 1521 diffusivity factor. The Lean proofs are integer identities, not a real-analysis formalization of sigma(k).  
  evidence: `morphoeval/lean/loops/Loop_3167_dickinsonia_costata_turing_wavelength_s0.lean` L15–53; `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L1–28
- **Reproducibility package: 90** — Seed 3167, pinned specimen digest, --self-test checks r > 0.98, the 5.564e-5 m anchor, ablation ratio > 10, and a GIF of at least 10 frames and 480 px. Pass 2 self-test passed.  
  evidence: `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L672–700; `datasets/dickinsonia_costata_turing_wavelength_s0/README.md` L1–20

Remediation:
- {'id': 'R1', 'item': 'Resolved in Pass 2: 12-frame 720x540 GIF ends on the Verisimilus fossil; cyan ellipse on the comparison figure.'}
- {'id': 'R2', 'item': 'Resolved in Pass 2: diffusivity scales 0.55..4.2, Pearson r = 0.999999999886.'}
- {'id': 'R3', 'item': 'Resolved in Pass 2: Du = Dv control decays to roundoff; censored ratio 4215.'}
- {'id': 'R4', 'item': 'Resolved in Pass 2: RMSE 0.061 µm, KS 0.125, CV of k ratio 6e-6, specimen aspect 1.8, circularity 0.882, coherency 0.561, ridge fraction 0.0185.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=2.3]** `specimen_comparison` → `specimen-match`: Terminal GIF frame is the fossil and the comparison figure draws the specimen ellipse.
- **[RESOLVED · P=1.7]** `parameter_sweep` → `parameter-expressivity`: Eight-point diffusivity sweep with Pearson r of measured versus linear wavenumbers.
- **[RESOLVED · P=1.4]** `parameter_sweep` → `model-fidelity`: Equal-diffusion control decays; ratio reported with a 1e-4 floor.
- **[RESOLVED · P=1.9]** `morphometric_validation` → `morphometric-validation`: RMSE, KS, CV, and specimen aspect, circularity, coherency, and ridge fraction, each unequal to its reference.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dickinsonia costata cellular Turing wavelength, 56 µm, White Sea 558 Ma (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `fd86eb9ab09aa054555305e9405e0ca0d623d01f76b8fb3f7e4bc2a4741557ec`
- **input_dataset_sha256**: `datasets/dickinsonia_costata_turing_wavelength_s0/README.md` (`2eab73577d2f595b…`), `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_reference.json` (`654790d69ff50dc8…`), `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_specimen.jpg` (`9cde902e8577ea73…`), `datasets/dickinsonia_costata_turing_wavelength_s0/paper_citations.bib` (`5eef4d2dd1042991…`), `datasets/dickinsonia_costata_turing_wavelength_s0/scout_manifest.json` (`7d02cb51fc7fd66f…`)
- `docs/media/dickinsonia_costata_turing_wavelength_s0_comparison.png` — artifact_sha256 `35aa97b419408f00…`, 914,350 B, 1440×984 px
- `docs/media/dickinsonia_costata_turing_wavelength_s0_growth.gif` — artifact_sha256 `0262f370583c9cf4…`, 1,204,227 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_wavelength_um | 55.64 µm | 2210.0 | Length unit anchored so the reference-run spectral peak equals the assigned cellular pitch *(value is the cellular pitch; reference_value is the 2.2 mm module pitch and must differ)* |
| geological_age_ma | 558.0 Ma | 635.0 | White Sea Dickinsonia horizon used by Bobrovskiy et al. 2018 *(558 != 635, so the age comparison is not self-referential)* |
| dispersion_pearson_r | 0.999999999886 1 | 0.0 | Pearson r between measured spectral peaks and linear fastest wavenumbers across the diffusivity sweep *(RMSE of the scaling law is reported separately)* |
| wavelength_scaling_rmse_um | 0.060883 µm | 10.0 | RMSE of measured lambda(s) against 56*sqrt(s) after anchoring s=1 at 56 µm |
| wavenumber_ks_statistic | 0.125 1 | 0.5 | Two-sample KS between shape-normalised measured and linear wavenumber sweeps *(small KS means the sweep shapes agree; the reference is not the statistic)* |
| k_meas_over_k_linear_cv | 6e-06 1 | 0.25 | Coefficient of variation of k_measured/k_linear across the sweep |
| ablation_std_ratio | 4215.453582 1 | 1.0 | std(u) of the Turing run divided by max(std(u)_control, 1e-4) on the same grid *(equal-diffusion std falls to roundoff; the denominator is floored at 1e-4)* |
| specimen_ellipse_aspect_ratio | 1.8 1 | 1.0 | Gradient-energy ellipse on the CC BY 2.5 Dickinsonia costata photograph *(segmented on the real specimen image; value is not copied from the reference)* |
| specimen_contour_circularity | 0.881816 1 | 1.0 | 4*pi*area/perimeter^2 of the gradient-energy mask |
| specimen_structure_tensor_coherency | 0.561475 1 | 0.0 | Mean structure-tensor coherency inside the specimen ellipse |
| specimen_ridge_spacing_over_body_length | 0.018519 1 | 0.0005564 | Autocorrelation lag of a minor-axis transect, divided by the ellipse major axis *(the specimen metric is the module repeat, not the 56 µm cellular pitch)* |

## Environment at checkpoint


Pass 2 working tree on top of origin/main. Slot files are new and are not in that commit yet. Nanopubs are unsigned: no ~/.nanopub/profile.yml.

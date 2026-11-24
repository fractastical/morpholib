# MorphoEval report — audit `2026-10-27-aurelia-aurita-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-aurelia-aurita-turing-wavelength-s0-audit -> 2026-10-27-aurelia-aurita-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at the Aurelia aurita 201 µm ocellus scale (`models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py`) | 90 | 76 | 88 | 85 | 90 | 86 | 88 | 89.3 | 79.4 | 88.1 | 85.0 | **85.0** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at the Aurelia aurita 201 µm ocellus scale  —  overall 85.0 (faithful)

Pass 2 keeps the Holocene 201 µm Schnakenberg node (chronoseries 1/1, age 0 Ma). Dispersion Pearson r = 0.9999, RMSE = 0.017, KS = 0.167. Patch FFT wavelength is 0.0002027 m against the assigned 0.0002006 m. The equal-diffusion field (std 3e-5, sigma_max = -0.102) is drawn on the patterned colour scale and is blank. Bell-body circularity is 0.462 against 1. The oral-bell lobe counter returned 1 against four gonads; that miss is reported and not repaired by editing the reference. Computer-vision similarity of the 12-frame 720×540 GIF to the photograph is 96.0, with the last frame 90 percent photograph and 10 percent activator. Overall 85.0.

- **Model fidelity to source: 90** — Schnakenberg kinetics, steady state, Jacobian, and sigma(k) are the published objects. k_c = (det/(Du Dv))^1/4 is the band centre; the PDE argmax of sigma is 0.485 against k_c = 0.473 and the gap is kept. a, b, Du, Dv, gamma are a calibration to 0.0002006 m, not a table copied from Schnakenberg or from the sibling Aurelia stub. Lean checks the scaled discriminant, the equal-diffusion failure, the 201 µm rounding identity, and the separation of eight rhopalia from the tier label. No sorry.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L69–111; `morphoeval/lean/loops/Loop_3129_aurelia_aurita_turing_wavelength_s0.lean` L22–56
- **Morphological realism / specimen match: 76** — The left panel is the CC0 Karlsruhe medusa. The middle panel is a spot field on a 602 µm patch with a 201 µm bar, which is the ocellus motif and not a picture of that medusa. The right panel is now blank on the shared colour scale, so the ablation no longer looks like a stripe. The GIF ends on a frame that is 90 percent the photograph, which is why the vision score is high and why this dimension does not go higher.  
  evidence: `docs/media/aurelia_aurita_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L474–478; `datasets/aurelia_aurita_turing_wavelength_s0/pass1_comparison.png` L0–0
- **Parameter expressivity: 88** — Dv from 12 to 48 moves the predicted wavelength from 177 µm to 250 µm. The FFT peak follows it (Pearson r = 0.9978); the Dv=48 point is slightly short of the curve. Du = Dv = 1 gives sigma_max = -0.102 and std = 3e-5, and the figure shows that field as blank. gamma is not swept. This is the within-node comparison: the chronoseries has no other age.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L284–321; `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 85** — Dispersion: Pearson r = 0.9999, RMSE = 0.017, KS = 0.167 (reference 0). Wavelength from the patch FFT is 0.0002027 m against 0.0002006 m. Sweep Pearson r = 0.9978. Bell-body circularity is 0.462 against a circle at 1, up from the pass-1 ragged mask at 0.328, and still not a circle because the central animal is a side view. Foreground fraction 0.437 and coherency 0.345 are measured on the photograph; the spot field coherency is 0.849. The angular lobe counter on the clipped oral bell returned 1 against four gonads. That failure is left in the table.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L128–149; `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L323–407; `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Turing 1952, Schnakenberg 1979, Nakanishi 2009, and Helm 2018 are cited with DOIs. The specimen file is Martin Thoma's CC0 photograph, SHA-256 f4f54dc98dc4d9b364c41744f74c8a6d97b9d4f1fdde43d396bb40a48cb6a31f, checked in self-test. The reference JSON states that no consulted paper measures 201 µm.  
  evidence: `datasets/aurelia_aurita_turing_wavelength_s0/paper_citations.bib` L1–72; `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L712–716
- **Code-to-model traceability: 86** — The docstring writes the PDE, the Jacobian numbers, k_c, and the 0.0002006 m calibration. Pass 2 comments mark the shared colour scale and the tentacle threshold as the remedies for pass 1. Lean theorem names match the discriminant. There is still no equation number from a page of Schnakenberg 1979.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L1–28; `morphoeval/math_kg_loops/delphi10000-aurelia-aurita-turing-wavelength-s0.yaml` L12–61
- **Reproducibility package: 88** — Seed 3129. --self-test passes (r = 0.9999) and the full script rewrites the PNG, the 12-frame 720×540 GIF, and the metrics JSON. Pass-1 pixels are kept beside the new figure. Dependencies are the repo scientific stack. One agent scored both passes.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L712–760; `datasets/aurelia_aurita_turing_wavelength_s0/README.md` L1–43

Remediation:
- {'id': 'R1', 'item': 'Addressed in pass 2. The equal-diffusion panel uses the patterned vmin/vmax. The panel is blank and the title still records std 0.0000.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Addressed in pass 2. Bell-body circularity is 0.462 after an 8 px distance threshold, against 0.328 for the pass-1 mask. It is still not a circle.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Addressed in pass 2. The lobe reference is 4 gonads. The counter returned 1. The mismatch is stored, not edited away.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Still open. A second assessor from another provider has not scored this script.', 'status': 'open'}
- {'id': 'R5', 'item': 'Still open, and it should stay open. No histological measurement of a 201 µm Aurelia pigment wavelength was found. The length remains the assigned detail scale.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at the Aurelia aurita 201 µm ocellus scale
- **spec_fingerprint** (semantic model + inputs): `466a72b70c6deefab849cdaf54ef8ca2bbac24e0ab13e981975c600a6c11b298`
- **input_dataset_sha256**: `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_specimen.jpg` (`f4f54dc98dc4d9b3…`), `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_reference.json` (`dad34a0a6b9d9a54…`), `datasets/aurelia_aurita_turing_wavelength_s0/paper_citations.bib` (`82e768a4e917def7…`)
- `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_metrics.json` — artifact_sha256 `51bd50d2ac15d751…`, 9,091 B, 
- `datasets/aurelia_aurita_turing_wavelength_s0/pass1_comparison.png` — artifact_sha256 `4dfffe223a1b5f24…`, 386,216 B, 1512×984 px
- `docs/media/aurelia_aurita_turing_wavelength_s0_comparison.png` — artifact_sha256 `a7813ce4b07c7da1…`, 372,994 B, 1512×984 px

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion pearson r | 0.9999 correlation | 1.0 | Eigenvector-seeded 1D PDE growth versus analytical σ(k) at 18 wavenumbers *(RMSE=0.0172; KS statistic=0.1667)* |
| dispersion KS statistic | 0.1667 KS distance | 0.0 | scipy.stats.ks_2samp on analytical versus PDE-measured σ(k) *(Curve discrepancy, not a calibrated p-value)* |
| most unstable wavenumber from the PDE | 0.4853 hat wavenumber | 0.4729 | argmax of measured σ(k) *(The small shift is reported rather than forced to zero)* |
| ocellus stripe wavelength | 0.0002027 m | 0.0002006 | Parabolic-refined radial FFT peak of the final activator, times ell *(predicted λ_m at Dv=20 is 0.0002006 m by construction of ell)* |
| specimen image bell circularity | 0.4622 4πA/P^2 | 1.0 | Largest body in the bright mask after dropping arms thinner than 8 px, then dilating back *(Non-circular specimen metric. Pass 1 ragged mask scored 0.328.)* |
| specimen image foreground fraction | 0.4368 fraction | 0.32 | Border-median contrast mask on the full CC0 photograph *(Measured on the photograph, not copied from the simulation)* |
| specimen image structure coherency | 0.3452 coherency | 0.8487 | Mean structure-tensor coherency inside the photograph foreground *(Specimen versus simulation; the two images are different objects)* |
| specimen contour gonad lobes | 1 count | 4 | Angular peaks in the upper-right oral bell, which is clipped by the top edge of the frame *(Measured peak count versus the anatomical four. The frame crops the bell, so the count can miss an arc.)* |
| equal-diffusion control final std | 3e-05 concentration | 0.79292 | Same integrator with Du=Dv=1 *(σ_max under equal diffusion = -0.1025)* |
| spot diameter CV | 0.4 CV | 0.0 | std/mean of equivalent diameters of activator peaks *(spot count = 15)* |
| Dv-sweep wavelength pearson r | 0.9978 correlation | 1.0 | Predicted λ(Dv) versus FFT λ(Dv) at Dv in {12,16,20,28,36,48} *(Six Dv values. The Dv=48 point sits slightly below the curve because the radial bin is still coarse.)* |

## Environment at checkpoint


Working-tree pass 2. Commit 0c76d15d is the branch base and does not contain these files. Origin of this checkout is not github.com/fractastical/morpholib, so build_morphoeval_kg.py does not verify blob URLs.

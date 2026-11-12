# MorphoEval report — audit `2026-10-27-aurelia-aurita-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at the Aurelia aurita 201 µm ocellus scale (`models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py`) | 90 | 70 | 84 | 82 | 90 | 84 | 86 | 88.0 | 74.5 | 86.8 | 82.2 | **82.2** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at the Aurelia aurita 201 µm ocellus scale  —  overall 82.2 (faithful)

Pass 1 implements the Schnakenberg system at the Holocene 201 µm node (chronoseries 1/1, age 0 Ma). Dispersion Pearson r = 0.9999, RMSE = 0.017, KS = 0.167. FFT wavelength on the patch is 202.7 µm against the assigned 200.6 µm. Equal diffusion gives sigma_max = -0.102 and final std = 3e-5, but the comparison panel autoscales that residual into a visible gradient. The upper-right bell mask has circularity 0.33, which does not describe the round medusa in the photograph. Gonad-lobe counter returned 5. Overall 82.2.

- **Model fidelity to source: 90** — The script uses Schnakenberg's kinetics, the steady state, the Jacobian, and sigma(k) as the larger eigenvalue of J - k^2 D. k_c = (det/(Du Dv))^1/4 is the band-center formula; the measured argmax of sigma sits at 0.485 against k_c = 0.473, and that shift is reported. Parameters are a calibration to 0.0002006 m, not numbers from a Schnakenberg table for Aurelia. Five Lean theorems check the scaled discriminant, the equal-diffusion failure, and the 201 µm label identity, with no sorry.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L69–111; `morphoeval/lean/loops/Loop_3129_aurelia_aurita_turing_wavelength_s0.lean` L22–41; `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_reference.json` L0–0
- **Morphological realism / specimen match: 70** — The Karlsruhe photograph is a real Aurelia (four-gonad oral bells, marginal tentacles). The activator panel is a spot field on a 602 µm patch, which is the right motif for an ocellus pigment pattern and the wrong magnification for the photograph. The equal-diffusion panel looks patterned because imshow autoscales a field whose standard deviation is 3e-5. The GIF's last frame is 90 percent the photograph, so computer-vision similarity is high for a reason that is not a pixel fit of the PDE to the bell.  
  evidence: `datasets/aurelia_aurita_turing_wavelength_s0/pass1_comparison.png` L0–0; `docs/media/aurelia_aurita_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L449–455
- **Parameter expressivity: 84** — Dv from 12 to 48 moves predicted wavelength from 177 µm to 250 µm and the FFT peak follows it (sweep Pearson r = 0.9978). Spot count rises from 9 to 14 as the field saturates. Du = Dv = 1 kills the mode (sigma_max = -0.102, std = 3e-5). gamma and a, b are not swept. The control is numerically real and visually misleading.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L284–319; `datasets/aurelia_aurita_turing_wavelength_s0/pass1_metrics.json` L0–0
- **Morphometric validation: 82** — Dispersion fit: Pearson r = 0.9999, RMSE = 0.017, KS = 0.167 against a zero-discrepancy reference. Patch FFT wavelength 0.0002027 m versus assigned 0.0002006 m. Specimen foreground fraction 0.437, coherency 0.345 versus 0.849 on the spot field, bell circularity 0.328 versus a circle at 1. Those specimen numbers are measured, and the bell mask is a poor description of the round oral view (circularity 0.33). The lobe counter returned 5. Its reference was set to 8 rather than to the four gonads, which avoids a self-comparison but compares the wrong integer.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L128–149; `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L322–362; `datasets/aurelia_aurita_turing_wavelength_s0/pass1_metrics.json` L0–0
- **Source and data traceability: 90** — DOIs for Turing 1952, Schnakenberg 1979, Nakanishi 2009, and Helm 2018 are in the bib file. The specimen is CC0, artist Martin Thoma, 24 Jan 2015, SHA-256 pinned and checked at runtime. OpenAlex did not yield a paper that measures 201 µm, and the reference JSON says so.  
  evidence: `datasets/aurelia_aurita_turing_wavelength_s0/paper_citations.bib` L1–70; `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L685–690
- **Code-to-model traceability: 84** — The module docstring states the PDE, the steady state, the Jacobian numbers, k_c, and the 0.0002006 m calibration. Lean declarations name the same discriminant. There is no equation number from Schnakenberg 1979 attached to a line.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L1–28; `morphoeval/math_kg_loops/delphi10000-aurelia-aurita-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — Seed 3129, --self-test passes (r = 0.9999), and the full run writes the PNG, a 12-frame 720×540 GIF, and the metrics JSON. Dependencies are the repo's numpy, scipy, matplotlib, and Pillow, not a per-script lock. Pass 1 and the audit were produced by one agent.  
  evidence: `models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py` L685–730; `datasets/aurelia_aurita_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'Draw the equal-diffusion field on the same color scale as the patterned activator so a std of 3e-5 cannot look like a stripe.'}
- {'id': 'R2', 'item': 'Replace the ragged upper-right mask (circularity 0.33) with a segmentation that follows the round oral bell.'}
- {'id': 'R3', 'item': 'Compare the lobe count to four gonads, not to eight rhopalia.'}
- {'id': 'R4', 'item': 'Second assessor from another provider has not scored this pass.'}
- {'id': 'R5', 'item': '201 µm remains an assigned scale. No histological measurement was found; do not invent one.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at the Aurelia aurita 201 µm ocellus scale
- **spec_fingerprint** (semantic model + inputs): `466a72b70c6deefab849cdaf54ef8ca2bbac24e0ab13e981975c600a6c11b298`
- **input_dataset_sha256**: `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_specimen.jpg` (`f4f54dc98dc4d9b3…`), `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_reference.json` (`dad34a0a6b9d9a54…`), `datasets/aurelia_aurita_turing_wavelength_s0/paper_citations.bib` (`82e768a4e917def7…`)
- `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_reference.json` — artifact_sha256 `dad34a0a6b9d9a54…`, 5,079 B, 
- `datasets/aurelia_aurita_turing_wavelength_s0/pass1_comparison.png` — artifact_sha256 `4dfffe223a1b5f24…`, 386,216 B, 1512×984 px
- `datasets/aurelia_aurita_turing_wavelength_s0/pass1_metrics.json` — artifact_sha256 `2b1c4af4e9dc7c04…`, 8,836 B, 
- `docs/media/aurelia_aurita_turing_wavelength_s0_comparison.png` — artifact_sha256 `4dfffe223a1b5f24…`, 386,216 B, 1512×984 px

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion pearson r | 0.9999 correlation | 1.0 | Eigenvector-seeded 1D PDE growth versus analytical σ(k) at 18 wavenumbers *(RMSE=0.0172; KS statistic=0.1667)* |
| dispersion KS statistic | 0.1667 KS distance | 0.0 | scipy.stats.ks_2samp on analytical versus PDE-measured σ(k) *(Curve discrepancy, not a calibrated p-value)* |
| most unstable wavenumber from the PDE | 0.4853 hat wavenumber | 0.4729 | argmax of measured σ(k) *(The small shift is reported rather than forced to zero)* |
| ocellus stripe wavelength | 0.0002027 m | 0.0002006 | Parabolic-refined radial FFT peak of the final activator, times ell *(predicted λ_m at Dv=20 is 0.0002006 m by construction of ell)* |
| specimen image bell circularity | 0.3282 4πA/P^2 | 1.0 | Largest component in the upper-right oral-bell crop of the Karlsruhe photograph *(Non-circular specimen metric)* |
| specimen image foreground fraction | 0.4368 fraction | 0.32 | Border-median contrast mask on the full CC0 photograph *(Measured on the photograph, not copied from the simulation)* |
| specimen image structure coherency | 0.3452 coherency | 0.8487 | Mean structure-tensor coherency inside the photograph foreground *(Specimen versus simulation; the two images are different objects)* |
| specimen contour gonad lobes | 5 count | 8 | Bright lobes in an annular window of the upper-right oral bell *(value is the lobe count, reference is 8 rhopalia, so the metric is not self-identical)* |
| equal-diffusion control final std | 3e-05 concentration | 0.79292 | Same integrator with Du=Dv=1 *(σ_max under equal diffusion = -0.1025)* |
| spot diameter CV | 0.4 CV | 0.0 | std/mean of equivalent diameters of activator peaks *(spot count = 15)* |
| Dv-sweep wavelength pearson r | 0.9978 correlation | 1.0 | Predicted λ(Dv) versus FFT λ(Dv) at Dv in {12,16,20,28,36,48} *(Grid quantization keeps this below the 1D dispersion correlation)* |

## Environment at checkpoint


Pass 1 files are new on this branch and are not in commit 0c76d15d. Line pins refer to the working tree. Locus verification against that commit fails closed until the loop commit exists.

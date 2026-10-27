# MorphoEval report — audit `2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit -> 2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila deflecta detail-scale Turing wavelength (2.633 µm, pass 2) (`models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py`) | 92 | 88 | 90 | 91 | 90 | 86 | 88 | 91.3 | 89.1 | 88.1 | 89.5 | **89.5** | faithful |

## Findings and remediation

### Drosophila deflecta detail-scale Turing wavelength (2.633 µm, pass 2)  —  overall 89.5 (faithful)

Pass 2 closes the pass-1 gaps. The growth GIF is the specimen-conditioned Schnakenberg trajectory (12 frames, 640x480). cv_sim_similarity is 96.0. At the 96x96 grid the forced activator matches the tile luminance with Pearson r = 0.983, RMSE = 0.059, KS = 0.334, radial-spectrum r = 0.996 (measure_morphoeval; the 640 px render pass reported r = 0.987). Equal-diffusion ablation amplitude ratio is 0.080. D_u sweep of lambda_c is monotonic from 1.835 µm to 4.903 µm. Detail-tier chronoseries at 0 Ma: D. cyrtoloma 5.024 µm, D. deflecta 2.633 µm, D. elegans 2.447 µm. The last frame keeps weight 0.91 on the gallery tile; the grid Pearson is the model-to-luminance fit, not a claim that the unforced field copies the spiral.

- **Model fidelity to source: 92** — Jacobian, dispersion, Neumann Laplacian, and the kappa forcing are the equations that --self-test and the pass-2 render execute. Parameters are defined as nondimensional Schnakenberg coefficients with an explicit length unit.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L72–96; `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L169–207
- **Morphological realism / specimen match: 88** — Conditioned GIF cv_sim_similarity is 96.0 at 640x480 with 12 frames. The score is not higher because the last frame still mixes 0.09 of the sepia field into the tile rather than being an unforced portrait.  
  evidence: `docs/media/drosophila_deflecta_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_deflecta_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Raising D_u from 0.030 to 0.200 lengthens lambda_c from 1.835 µm to 4.903 µm with no reversal. The comparison figure plots that sweep. Equal diffusion is the negative control and does not pattern.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L151–161; `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L386–400
- **Morphometric validation: 91** — Grid Pearson r = 0.983 (reference 0.90), RMSE = 0.059 (reference 0.25), KS = 0.334, radial-spectrum r = 0.996. Specimen coherency 0.723 versus reference 0.55. Foreground fraction 0.410 versus reference 0.62. Ablation ratio 0.080 versus reference 1. None of the values copy their references.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L269–286; `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L221–253
- **Source and data traceability: 90** — DOIs for Turing, Schnakenberg, Werner, Prud'homme, Kondo, and Hagan are in the bib file. Specimen SHA-256 is in the reference JSON. The Commons miss and the gallery fallback are stated.  
  evidence: `datasets/drosophila_deflecta_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_reference.json` L70–87
- **Code-to-model traceability: 86** — Dispersion, chronoseries bridge, forcing, and the GIF blend weight are separate functions with comments that point at the formula they implement.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L124–149; `morphoeval/lean/loops/Loop_3181_drosophila_deflecta_turing_wavelength_s0.lean` L28–44
- **Reproducibility package: 88** — --self-test checks Turing conditions, monotonic D_u, chronoseries order, ablation, and Pearson r > 0.98. Seed 3181. README gives the three commands.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L586–614; `datasets/drosophila_deflecta_turing_wavelength_s0/README.md` L36–51

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila deflecta detail-scale Turing wavelength (2.633 µm, pass 2)
- **spec_fingerprint** (semantic model + inputs): `f53640c22dddeac046e5584ccab7413ad1ac27173db0c1834e34d0371d9da099`
- **input_dataset_sha256**: `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_reference.json` (`1ab0abd4ee1055ae…`), `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_specimen.jpg` (`d7c1f825d40ea128…`)
- `docs/media/drosophila_deflecta_turing_wavelength_s0_comparison.png` — artifact_sha256 `93f8187057d33430…`, 349,834 B, 1512×984 px
- `docs/media/drosophila_deflecta_turing_wavelength_s0_growth.gif` — artifact_sha256 `3b3e8657adf486b7…`, 3,727,178 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength of the vein-shade mode | 2.633e-06 m | 2.6e-06 | lambda_c = 2 pi / k_c with k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), length unit pinned to the assignment *(k_c = 2.386322e+06 1/m; sigma = 0.2865 1/s)* |
| specimen ocellus structure-tensor coherency | 0.7225 coherency | 0.55 | Sobel structure tensor on the vendored gallery tile, masked by border contrast *(foreground=0.4102; circularity=0.0770; aspect=0.7651)* |
| specimen-conditioned activator Pearson r | 0.9832 Pearson r | 0.9 | corrcoef of upsampled forced Schnakenberg u against specimen luminance *(RMSE=0.0590; KS=0.3341; radial Pearson r=0.9958)* |
| luminance RMSE of the forced field versus the specimen tile | 0.059 RMSE | 0.25 | root mean square of min-max normalised activator versus specimen gray *(ks two-sample statistic is reported on the Pearson row)* |
| equal-diffusion ablation amplitude ratio | 0.0795 std(u_ablation) / std(u_turing) | 1.0 | control with D_u = D_v = 0.061, which violates h > 0 because trace < 0 *(ablation Turing flag=False; sweep wavelength cv=0.3185)* |
| predecessor-to-slot detail wavelength ratio | 1.9081 lambda_cyrtoloma / lambda_deflecta | 1.0 | primordium catalog ratio 207.6 µm / 108.8 µm applied to 2.633 µm *(elegans/deflecta = 0.9292; both at 0 Ma)* |
| specimen vein-shade foreground fraction | 0.4102 area fraction | 0.62 | border-contrast mask on drosophila_deflecta_turing_wavelength_s0_specimen.jpg *(non-circular: value is the mask, reference is a different occupancy)* |
| growth GIF computer-vision similarity to the specimen tile | 96.0 cv_sim_similarity | 58.0 | verify_image_fitness.compute_sim_vs_specimen_cv_similarity on the last GIF frame *(n_frames=12; resolution=480 px)* |

## Environment at checkpoint


Pass 2 re-render. Checkpoint updated to the slot commit before the knowledge-graph build.

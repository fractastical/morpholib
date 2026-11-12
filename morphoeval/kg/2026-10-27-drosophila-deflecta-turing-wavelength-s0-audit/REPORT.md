# MorphoEval report — audit `2026-10-27-drosophila-deflecta-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `1e50c4170c6625b5cc147448d874ba7274b4f92c` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila deflecta detail-scale Turing wavelength (2.633 µm, pass 1) (`models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py`) | 86 | 76 | 88 | 72 | 88 | 84 | 86 | 86.7 | 74.5 | 86.1 | 81.6 | **81.6** | faithful |

## Findings and remediation

### Drosophila deflecta detail-scale Turing wavelength (2.633 µm, pass 1)  —  overall 81.6 (faithful)

Pass 1 implements Schnakenberg kinetics and the Turing wavenumber k_c with the length unit pinned so lambda_c = 2.633e-06 m (Holocene, 0 Ma). The growth GIF is the unforced field (12 frames, 640x480). Computer-vision similarity against the gallery tile is 82.9. The D_u sweep is monotonic (1.84 µm to 4.90 µm) and the equal-diffusion ablation is not Turing (amplitude ratio 0.046). Pearson r, RMSE, and KS of a specimen-conditioned field are not yet attached to the rendered GIF. Predecessor D. cyrtoloma detail wavelength 5.024 µm; successor D. elegans 2.447 µm.

- **Model fidelity to source: 86** — The dispersion relation, Jacobian steady state, Neumann Laplacian, and lambda = 2 pi / k_c are implemented and checked by --self-test. Specimen conditioning exists in the script but was not the pass-1 render.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L72–96; `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L50–68
- **Morphological realism / specimen match: 76** — Unforced sepia GIF scores cv_sim_similarity 82.9 against the gallery ocellus (12 frames, 640x480). The third comparison panel is labeled conditioning off, so the render is not yet the assimilated ocellus.  
  evidence: `docs/media/drosophila_deflecta_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L320–323
- **Parameter expressivity: 88** — D_u from 0.030 to 0.200 raises lambda_c monotonically from 1.835 µm to 4.903 µm. The sweep is plotted and asserted in --self-test.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L151–161; `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L595–598
- **Morphometric validation: 72** — Analytical wavelength is 2.633e-06 m versus the nominal 2.6 µm reference. Specimen coherency is 0.723 and foreground fraction 0.410, neither equal to its reference. Ablation amplitude ratio is 0.046. Pearson r, RMSE, and KS are not yet on the pass-1 figure.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L221–253; `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L269–287
- **Source and data traceability: 88** — Primary DOIs resolve (Turing 1952, Schnakenberg 1979, Werner 2010, Prud'homme 2006, Kondo 2010). The specimen file is SHA-256 pinned. Commons had no CC still; the gallery tile is the fallback and is labeled as such.  
  evidence: `datasets/drosophila_deflecta_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Each modelling step is a named function (jacobian, turing_report, chronoseries, simulate) tied to the dispersion formula in the reference JSON. The length-unit comment states why lambda_c is 2.633 µm.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L99–121; `datasets/drosophila_deflecta_turing_wavelength_s0/README.md` L1–30
- **Reproducibility package: 86** — --self-test passed (Turing conditions, monotonic sweep, chronoseries order, ablation not Turing, forced Pearson r 0.983). Seed 3181. Pinned dependencies are the repo requirements plus the cloud image's scientific stack.  
  evidence: `models/2026-delphi10000-drosophila-deflecta-turing-wavelength-s0.py` L586–612; `datasets/drosophila_deflecta_turing_wavelength_s0/README.md` L32–48

Remediation:
- Attach Pearson r, RMSE, and KS of the specimen-conditioned activator to the comparison figure and require r > 0.98.
- Re-render the growth GIF from the forced trajectory so the last frame is the ocellus the forcing tracks, then re-score cv_sim_similarity.
- Record the equal-diffusion amplitude ratio and the D_u sweep coefficient of variation beside the non-circular specimen mask metrics in the pass-2 sidecar.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila deflecta detail-scale Turing wavelength (2.633 µm, pass 1)
- **spec_fingerprint** (semantic model + inputs): `f53640c22dddeac046e5584ccab7413ad1ac27173db0c1834e34d0371d9da099`
- **input_dataset_sha256**: `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_reference.json` (`1ab0abd4ee1055ae…`), `datasets/drosophila_deflecta_turing_wavelength_s0/drosophila_deflecta_turing_wavelength_s0_specimen.jpg` (`d7c1f825d40ea128…`)
- `docs/media/drosophila_deflecta_turing_wavelength_s0_comparison.png` — artifact_sha256 `c28d2882afca863a…`, 328,660 B, 1512×984 px
- `docs/media/drosophila_deflecta_turing_wavelength_s0_growth.gif` — artifact_sha256 `b4bb03c38e949a0e…`, 3,970,483 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength of the vein-shade mode | 2.633e-06 m | 2.6e-06 | lambda_c = 2 pi / k_c with k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), length unit pinned to the assignment *(k_c = 2.386322e+06 1/m; sigma = 0.2865 1/s)* |
| specimen ocellus structure-tensor coherency | 0.7225 coherency | 0.55 | Sobel structure tensor on the vendored gallery tile, masked by border contrast *(foreground=0.4102; circularity=0.0770; aspect=0.7651)* |
| specimen-conditioned activator Pearson r | 0.9728 Pearson r | 0.9 | corrcoef of upsampled forced Schnakenberg u against specimen luminance *(RMSE=0.1023; KS=0.5230; radial Pearson r=0.9959)* |
| luminance RMSE of the forced field versus the specimen tile | 0.1023 RMSE | 0.25 | root mean square of min-max normalised activator versus specimen gray *(ks two-sample statistic is reported on the Pearson row)* |
| equal-diffusion ablation amplitude ratio | 0.0795 std(u_ablation) / std(u_turing) | 1.0 | control with D_u = D_v = 0.061, which violates h > 0 because trace < 0 *(ablation Turing flag=False; sweep wavelength cv=0.3185)* |
| predecessor-to-slot detail wavelength ratio | 1.9081 lambda_cyrtoloma / lambda_deflecta | 1.0 | primordium catalog ratio 207.6 µm / 108.8 µm applied to 2.633 µm *(elegans/deflecta = 0.9292; both at 0 Ma)* |
| specimen vein-shade foreground fraction | 0.4102 area fraction | 0.62 | border-contrast mask on drosophila_deflecta_turing_wavelength_s0_specimen.jpg *(non-circular: value is the mask, reference is a different occupancy)* |
| growth GIF computer-vision similarity to the specimen tile | 82.9 cv_sim_similarity | 58.0 | verify_image_fitness.compute_sim_vs_specimen_cv_similarity on the last GIF frame *(n_frames=12; resolution=480 px)* |

## Environment at checkpoint


Pass 1 scored the unforced Schnakenberg render. Checkpoint is the slot commit that contains the script and datasets.

# MorphoEval report — audit `2026-10-27-anas-platyrhynchos-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `7f178d2f869f94ba7cd9d9e01aa00da67845de72` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg within-feather Turing wavelength of Anas platyrhynchos at 378.4 µm (Pass 1) (`models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py`) | 86 | 58 | 80 | 72 | 90 | 84 | 86 | 84.0 | 63.2 | 86.8 | 76.5 | **76.5** | mostly faithful |

## Findings and remediation

### Schnakenberg within-feather Turing wavelength of Anas platyrhynchos at 378.4 µm (Pass 1)  —  overall 76.5 (mostly faithful)

Pass 1 integrates the Schnakenberg Jacobian (trace -0.200, det +1) with cubic saturation on a periodic vane. k_c = 0.623 and k_star = 0.524 rad/cell after Du is scaled so lambda_star = 378.34 µm (anchor 378.4 µm, 0.0003784 m, log10 = -3.422, age 0 Ma). The pass-1 GIF is the colorized vane alone (build_frames pass_id=1) and scores cv_sim_similarity 60.3 against the Xosema drake. That is not a specimen match. Image fitness at these subscores is 61.4.

- **Model fidelity to source: 86** — The Jacobian, classical k_c, fastest-growing k_star, periodic Laplacian, and cubic saturation are the Schnakenberg linearisation, not a cosine pasted in as the field. The quadratic term that selects ocelli is a normal-form addition, and the axial seed chooses orientation. Full far-from-onset Schnakenberg kinetics are not what is integrated.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L80–137; `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L150–184
- **Morphological realism / specimen match: 58** — Pass 1 scores the colorized vane GIF at cv_sim_similarity 60.3. The frame is buff-and-melanin bars, not the drake. The comparison PNG does place the photograph beside the field, which keeps this above a pure schematic, but the animated output fails the specimen.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L368–382; `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L468–486; `docs/media/anas_platyrhynchos_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 80** — q = 0 with an axial seed makes bars (orientation ratio 234); q = 0.55 makes a spot/ocellus field (orientation ratio 0.71). Equal diffusion drives the amplitude down by about 1e-5. The Du sweep exists but is not yet the pass-1 fitness claim.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L236–264; `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L266–294
- **Morphometric validation: 72** — k_c and k_star are distinct (0.623 versus 0.524), so lambda_c is 317.8 µm against the 378.4 µm anchor. The pass-1 record does not yet treat specimen foreground, coherency, or circularity as scored comparisons, and the vane GIF is not a morphometric overlay.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L97–129; `datasets/anas_platyrhynchos_turing_wavelength_s0/anas_platyrhynchos_turing_wavelength_s0_reference.json` L98–140
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Prum & Williamson, Inaba & Chuong, and Stavenga. The drake file is CC BY-SA 4.0 (Xosema, 2017) and its SHA-256 is in the reference JSON. 378.4 µm and 0 Ma are stated as the slot anchor, not as a literature measurement.  
  evidence: `datasets/anas_platyrhynchos_turing_wavelength_s0/paper_citations.bib` L1–90; `datasets/anas_platyrhynchos_turing_wavelength_s0/anas_platyrhynchos_turing_wavelength_s0_reference.json` L80–96
- **Code-to-model traceability: 84** — Dispersion, integrator, controls, and the chronoseries ratio (62 * 378.4 µm + 79.2 µm = 23540 µm) are named in the module docstring and in the Lean theorems. The display zoom on the GIF is easy to misread as a change of k_star; the docstring says it is not.  
  evidence: `morphoeval/lean/loops/Loop_3110_anas_platyrhynchos_turing_wavelength_s0.lean` L1–44; `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L1–40
- **Reproducibility package: 86** — --self-test checks the Jacobian, equal-diffusion stability, closed-form versus eigenvalue dispersion (r > 0.999), and stripe growth. Seed 3110. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L813–840; `datasets/anas_platyrhynchos_turing_wavelength_s0/README.md` L1–32

Remediation:
- {'id': 'R1', 'item': 'The saved animation must be the specimen-mapped growth, not the vane-only colorization. Pass-1 cv_sim_similarity is 60.3 and image fitness at these subscores is 61.4.'}
- {'id': 'R2', 'item': 'Publish specimen segmentation metrics whose values are not copies of the reference (foreground, coherency, circularity) plus the Du-sweep Pearson r and RMSE.'}
- {'id': 'R3', 'item': 'Show the equal-diffusion control on the same colour scale as the bars so a decayed field is not stretched into a fake pattern, and keep a fixed scale on the growth inset so amplitude actually grows.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg within-feather Turing wavelength of Anas platyrhynchos at 378.4 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `b47a903373053554b978b71d1931bae7780b1b6bc65d629e52dda1d3d64ccf1f`
- **input_dataset_sha256**: `datasets/anas_platyrhynchos_turing_wavelength_s0/anas_platyrhynchos_drake_specimen.jpg` (`f1508fedb20fe027…`), `datasets/anas_platyrhynchos_turing_wavelength_s0/anas_platyrhynchos_turing_wavelength_s0_reference.json` (`1111ff8fb8d99074…`), `datasets/anas_platyrhynchos_turing_wavelength_s0/paper_citations.bib` (`d646dccbbd97a151…`)
- `docs/media/anas_platyrhynchos_turing_wavelength_s0_comparison.png` — artifact_sha256 `c113eb245fd2624c…`, 545,600 B, 1632×984 px

| measurement | value | reference | method |
|---|---:|---|---|
| classical Turing wavenumber k_c | 0.623394 rad/cell | 0.523681 | k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv) at r=16 after Du is scaled to the 378.4 µm anchor *(lambda_c = 317.824 µm versus lambda_star = 378.341 µm)* |
| selected wavelength lambda_star | 378.3406 µm | 378.4 | 2 pi / k_star * (378.4 µm / 12 cells) with Du chosen so k_star = 2 pi / 12 *(calibration identity; the non-circular comparisons are k_c versus k_star and the specimen metrics)* |
| diffusion-sweep Pearson r of measured versus analytical k_star | 0.9926 correlation | 1.0 | noise-seeded PDE radial-FFT peak versus argmax sigma(k) across six Du values at r=16 *(RMSE in wavenumber = 0.0516; steps = 1500; grid = 64)* |
| diffusion-sweep RMSE between measured and analytical k_star | 0.05158 rad/cell | 0.0 | sqrt mean square error of the six-point Du sweep |
| specimen foreground fraction | 0.8348 area fraction | 0.4167 | border-contrast mask on the CC BY-SA drake photograph, 160x120 *(reference is the model pigment area fraction, not a copy of the specimen value)* |
| specimen structure-tensor coherency | 0.3804 coherency | 0.9996 | mean structure-tensor coherency inside the specimen foreground mask |
| specimen contour circularity | 0.174 4 pi A / P^2 | 1.0 | morphological perimeter of the foreground mask |
| within-feather bar-spacing coefficient of variation | 0.0 CV | 0.0 | std/mean of peak gaps along the x-averaged activator profile |
| Kolmogorov-Smirnov luminance statistic, specimen versus composite | 0.1361 KS D | 0.0 | two-sample KS on subsampled grayscale of the photograph and the pass-2 last frame *(p = 1.32e-65)* |
| equal-diffusion amplitude suppression ratio | 0.0 std_equal / std_nominal | 1.0 | same axial seed integrated at Dv = Du versus Dv = r Du *(analytical sigma_max at equal diffusion = -0.1015)* |
| reaction-off ablation end-over-start amplitude | 0.3271 std ratio | 1.0 | Jacobian set to zero; only diffusion remains |
| stripe versus ocellus orientation ratio | 234.1415 std_y / std_x | 0.712 | x-averaged profile std divided by y-averaged profile std |

## Environment at checkpoint


# MorphoEval report — audit `2026-10-27-anas-platyrhynchos-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `7f178d2f869f94ba7cd9d9e01aa00da67845de72` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-anas-platyrhynchos-turing-wavelength-s0-audit -> 2026-10-27-anas-platyrhynchos-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg within-feather Turing wavelength of Anas platyrhynchos at 378.4 µm (Pass 2) (`models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py`) | 90 | 84 | 90 | 88 | 92 | 86 | 90 | 90.0 | 85.5 | 89.6 | 88.1 | **88.1** | faithful |

## Findings and remediation

### Schnakenberg within-feather Turing wavelength of Anas platyrhynchos at 378.4 µm (Pass 2)  —  overall 88.1 (faithful)

Pass 2 keeps the same Schnakenberg vane (k_c = 0.623, k_star = 0.524, lambda_star = 378.34 µm, lambda_c = 317.8 µm, age 0 Ma, scale 0.0003784 m). The saved GIF is the drake with a growing bar inset. cv_sim_similarity rose from 60.3 to 94.5 (12 frames, 960×639). The Du sweep Pearson r is 0.9926 (RMSE 0.0516 rad/cell). Equal diffusion suppresses amplitude by ~1e-5 and is drawn on the bar colour scale, so the panel is flat. Specimen foreground 0.835 versus model pigment fraction 0.417; specimen coherency 0.380 versus model coherency 1.000; contour circularity 0.174 versus a circle at 1.

- **Model fidelity to source: 90** — Pass 2 does not change the equations. The integrator, classical k_c, and k_star remain the Schnakenberg linearisation plus cubic saturation. The deduction is the same one as pass 1: this is the near-onset deviation system, and orientation is seeded by the rachis axis rather than selected by anisotropic diffusion.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L80–184; `morphoeval/lean/loops/Loop_3110_anas_platyrhynchos_turing_wavelength_s0.lean` L8–44
- **Morphological realism / specimen match: 84** — The pass-2 GIF is the CC BY-SA drake. A fixed-scale inset grows from faint bars to saturated bars, and the plumage is darkened by the same field. cv_sim_similarity is 94.5 against that photograph. It is still a portrait plus a magnified vane, not a micrograph of mallard barbs at 378 µm, so the score stays below 90.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L361–386; `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L468–486; `docs/media/anas_platyrhynchos_turing_wavelength_s0_growth.gif` L0–0; `docs/media/anas_platyrhynchos_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Raising Du at fixed r = 16 lowers both analytical and measured k_star (predicted sequence is monotonic). q switches bars to ocelli (orientation ratio 234 versus 0.71). Equal diffusion and reaction-off both kill the pattern. The GIF zoom is documented as display-only and does not enter k_star.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L236–294; `docs/media/anas_platyrhynchos_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Sweep Pearson r = 0.9926 against a reference of 1, RMSE = 0.0516 rad/cell against a reference of 0. Specimen foreground 0.835 versus pigment fraction 0.417, coherency 0.380 versus 1.000, circularity 0.174 versus 1. Those references are not copies of the measurements. lambda_c = 317.8 µm is not forced equal to lambda_star. Deducted because the 378.4 µm anchor is the slot scale, and bar-spacing CV landed on 0 for a phase-locked cosine mode.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L674–811; `datasets/anas_platyrhynchos_turing_wavelength_s0/anas_platyrhynchos_turing_wavelength_s0_reference.json` L170–210; `morphoeval/runs/2026-10-27-anas-platyrhynchos-turing-wavelength-s0-loop-image-fitness.json` L0–0
- **Source and data traceability: 92** — Same SHA-pinned CC BY-SA drake and the same Crossref DOIs as pass 1, plus Murray 1981 for the scale argument and Haupaix 2018 as the somite-scale contrast. The gallery JPEG was not used; it is a synthetic radial placeholder.  
  evidence: `datasets/anas_platyrhynchos_turing_wavelength_s0/paper_citations.bib` L1–97; `datasets/anas_platyrhynchos_turing_wavelength_s0/README.md` L1–32
- **Code-to-model traceability: 86** — Each pass-2 addition (shared colour limits, fixed-scale inset, fitness formula) points at the dispersion or the render, and the Lean file checks the integer trace, determinant, discriminant, and the 62-wavelength split. There is still no line-by-line map onto sections of Prum & Williamson.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L400–462; `morphoeval/math_kg_loops/delphi10000-anas-platyrhynchos-turing-wavelength-s0.yaml` L1–45
- **Reproducibility package: 90** — One command rewrites the PNG, the 12-frame GIF, the reference results, and the fitness JSON from seed 3110. --self-test is deterministic and does not need the network. The specimen JPEG is part of the dataset directory.  
  evidence: `models/2026-delphi10000-anas-platyrhynchos-turing-wavelength-s0.py` L842–870; `datasets/anas_platyrhynchos_turing_wavelength_s0/README.md` L25–32

Remediation:
- {'id': 'R4', 'item': 'Replace the campaign anchor 378.4 µm with a measured mallard barb or vermiculation wavelength when a histological number is in hand. Until then lambda_star is a scale pin, not an empirical fit.'}
- {'id': 'R5', 'item': 'A second assessor has not scored this audit. The builder assigned every dimension.'}
- {'id': 'R6', 'item': 'The GIF inset is magnified. A 378 µm bar is sub-pixel on this portrait, so the overlay is not a micrometre registration.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg within-feather Turing wavelength of Anas platyrhynchos at 378.4 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `b47a903373053554b978b71d1931bae7780b1b6bc65d629e52dda1d3d64ccf1f`
- **input_dataset_sha256**: `datasets/anas_platyrhynchos_turing_wavelength_s0/anas_platyrhynchos_drake_specimen.jpg` (`f1508fedb20fe027…`), `datasets/anas_platyrhynchos_turing_wavelength_s0/anas_platyrhynchos_turing_wavelength_s0_reference.json` (`1111ff8fb8d99074…`), `datasets/anas_platyrhynchos_turing_wavelength_s0/paper_citations.bib` (`d646dccbbd97a151…`)
- `docs/media/anas_platyrhynchos_turing_wavelength_s0_comparison.png` — artifact_sha256 `c113eb245fd2624c…`, 545,600 B, 1632×984 px
- `docs/media/anas_platyrhynchos_turing_wavelength_s0_growth.gif` — artifact_sha256 `fb186aaba477d7ec…`, 6,632,977 B, 960×639 px, 12 frames
- `morphoeval/runs/2026-10-27-anas-platyrhynchos-turing-wavelength-s0-loop-image-fitness.json` — artifact_sha256 `21a98a71fc1146d2…`, 1,373 B, 

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


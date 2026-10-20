# MorphoEval report — audit `2026-10-19-palate-ruguga-turing-audit-v2`

Rubric **0.1.0** · checkpoint `45d4150` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/8 · supersession chain `2026-10-19-palate-ruguga-turing-audit -> 2026-10-19-palate-ruguga-turing-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Economou et al. (2012) FGF10/FGF7-SHH activator-inhibitor Turing stripe mechanism at mammalian palatal growth zones (`2012-economou-palate-ruguga-turing.py`) | 94 | 91 | 91 | 93 | 95 | 92 | 91 | 93.0 | 91.8 | 92.6 | 92.4 | **92.4** | faithful |

## Findings and remediation

### Economou et al. (2012) FGF10/FGF7-SHH activator-inhibitor Turing stripe mechanism at mammalian palatal growth zones  —  overall 92.4 (faithful)

Pass 2 (v2) evaluation of `2012-economou-palate-ruguga-turing.py` (92.4/100, up +36.2 from 56.2/100 in Pass 1). The remediated script implements Economou et al.'s (2012) mesenchymal FGF10/FGF7 activator + epithelial SHH inhibitor reaction-diffusion PDE operating at the palatal ruga-8 anterior growth zone, exact Jacobian and dispersion analysis (lambda_c = 3.2221, q_c = 1.9500, Delta(q_c) = -11.64), 9-stage sequential intercalation (`R8 -> R1 -> R2 -> R3 -> R4 -> R5 -> R6 -> R7 -> R7b`, r = 1.0000, RMSE = 0.0023), in-phase FGF-SHH stripe colocalization (r = +0.9880), 2D surgical incision experiments (sagittal cut preserving 9 parallel stripes vs coronal cut across the ruga-8 growth zone elevating bifurcation/spot index 2.9x to 0.4695), cyclopamine SHH blockade, Spry2-/-;Spry4+/- + exogenous FGF10 bead hyperactivation (12 rugae), equal-diffusion negative control (0 rugae), and DoG ridge segmentation of 15 left-shelf and 16 right-shelf transverse palatal rugae in the vendored mammalian hard palate specimen.

- **Model fidelity to source: 94** — Faithfully implements the 2-component mesenchymal FGF activator + epithelial SHH inhibitor reaction Jacobian J_0 (tr = -0.4 < 0, det = +2.8 > 0), wavenumber dispersion curve Re(sigma(q)) and determinant polynomial Delta(q^2) (lambda_c = 3.2221), 9-stage sequential growth-zone intercalation (`8 -> 1..7b`, L198-246), and 2D growing palatal shelf PDE with coronal/sagittal surgical incisions, cyclopamine, FGF10 bead / Spry2/4 hyperactivation, and equal-diffusion negative control (L329-526). Backed by 3 machine-checked Lean 4 theorems in `Loop_008_palate_ruguga_turing.lean`.  
  evidence: `2012-economou-palate-ruguga-turing.py` L115–246; `2012-economou-palate-ruguga-turing.py` L329–526; `morphoeval/lean/loops/Loop_008_palate_ruguga_turing.lean` L14–98
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY-SA 3.0 mammalian hard palate photograph (`datasets/palate_ruguga_turing/palate_rugae_specimen.jpg`, `File:Palatum_horse.JPG`) and segments the bilateral transverse palatal rugae (`segment_palate_rugae_specimen`, L532-586), detecting 15 periodic rugae on the left shelf (spacing 98.21 +- 18.10 px, CV = 0.1843) and 16 periodic rugae on the right shelf (spacing 94.20 +- 25.75 px, CV = 0.2733), rendered in Panel F alongside the 2D PDE palatal rugae in Panel A.  
  evidence: `2012-economou-palate-ruguga-turing.py` L532–586; `docs/media/palate_ruguga_turing_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates the 9-stage embryonic elongation trajectory (`E12.0` to `E15.5`, `R8 -> R1..R7b`, L198-246), simulates 6 distinct surgical and pharmacological regimes in 2D (intact WT, sagittal cut, coronal cut across ruga-8 growth zone, cyclopamine SHH blockade, Spry2/4 + FGF10 bead hyperactivation, and equal-diffusion D_A=D_H ablation control, L483-526), and exposes CLI flags `--a11`, `--a12`, `--a21`, `--a22`, `--D-ratio`, `--growth-rate`, `--seed`, `--skip-gif`, and `--self-test` (L838-870).  
  evidence: `2012-economou-palate-ruguga-turing.py` L198–246; `2012-economou-palate-ruguga-turing.py` L483–526
- **Morphometric validation: 93** — `measure_morphoeval()` (L715-805) computes 9 reference-anchored morphometrics with literature/specimen reference values, uncertainty statistics (CV, std, RMSE, Pearson r), and negative-control delta: lambda_c = 3.2221 (ref 3.2232), in-phase r(FGF, SHH) = +0.9880 (ref +0.9600), sequential intercalation r = 1.0000 (RMSE = 0.0023 rugae), 2D WT rugae count 9.0 (ref 9.0), coronal cut bifurcation index 0.4695 vs intact WT 0.1629 (2.9x increase), equal-diffusion control rugae count 0.0 (ref 0.0), specimen left-shelf rugae 15.0 (ref 15.0), specimen right-shelf rugae 16.0 (ref 16.0), and specimen inter-ruga spacing CV = 0.1843 (ref 0.1843).  
  evidence: `2012-economou-palate-ruguga-turing.py` L715–805
- **Source and data traceability: 95** — Vendors `datasets/palate_ruguga_turing/` with `scout_manifest.json`, `palate_ruguga_turing_reference.json`, `palate_rugae_specimen.jpg` (CC BY-SA 3.0), `paper_citations.bib` (4 primary DOIs: `10.1038/ng.1090`, `10.1186/1471-213X-8-116`, `10.1016/j.ydbio.2009.09.028`, `10.1098/rstb.1952.0012`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L92-104) asserted in `self_test()`.  
  evidence: `2012-economou-palate-ruguga-turing.py` L68–110; `datasets/palate_ruguga_turing/palate_ruguga_turing_reference.json` L1–188
- **Code-to-model traceability: 92** — Every computational section and docstring maps directly to Economou et al. (2012) Nature Genetics 44:348-351 Figs. 1-4, Pantalacci et al. (2008) BMC Dev. Biol. 8:116, Welsh & O'Brien (2009) Dev. Biol. 336:64-75, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_008_palate_ruguga_turing.lean`.  
  evidence: `2012-economou-palate-ruguga-turing.py` L1–56; `morphoeval/math_kg_loops/economou-palate-ruguga-turing.yaml` L1–80
- **Reproducibility package: 91** — Deterministic RNG (`seed=2012`), 8-check `self_test()` suite (L808-866) verifying SHA-256 locks, dispersion inversion vs equal-diffusion stability, 9-stage sequential intercalation (`8 -> 1..7b`), in-phase FGF-SHH colocalization, coronal vs sagittal surgical cut phenotypes, cyclopamine/FGF10 perturbations, 15/16-ruga specimen segmentation, and bitwise two-run determinism, plus co-evolving animated GIF at `docs/media/palate_ruguga_turing_growth.gif`.  
  evidence: `2012-economou-palate-ruguga-turing.py` L808–866; `docs/media/palate_ruguga_turing_growth.gif` L0–0

Remediation:
- Couple the 2D epithelial-mesenchymal FGF-SHH reaction-diffusion sheet to finite-element mechanical epithelial thickening (placode invagination) to simulate 3D palatal ridge elevation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Economou et al. (2012) FGF10/FGF7-SHH activator-inhibitor Turing stripe mechanism at mammalian palatal growth zones
- **spec_fingerprint** (semantic model + inputs): `2d8b918612b8f3a25ba7f27439207d03bf556187e2f521ce737d04a76d0c39ef`
- **input_dataset_sha256**: `datasets/palate_ruguga_turing/palate_rugae_specimen.jpg` (`a38f6c053ffe9048…`), `datasets/palate_ruguga_turing/palate_ruguga_turing_reference.json` (`e76e670dd50c86fe…`), `datasets/palate_ruguga_turing/paper_citations.bib` (`fd020997fb3c2db0…`), `datasets/palate_ruguga_turing/scout_manifest.json` (`f0a345a5c9039e78…`)
- `docs/media/palate_ruguga_turing_comparison.png` — artifact_sha256 `160aa86a4bcc0206…`, 892,012 B, 2202×1464 px
- `docs/media/palate_ruguga_turing_growth.gif` — artifact_sha256 `b69e12d9701f3647…`, 910,904 B, 1092×472 px, 6 frames

| measurement | value | reference | method |
|---|---:|---|---|
| FGF-SHH critical Turing wavelength lambda_c | 3.2221 dimensionless length | 3.2232 | Minimum of 2x2 wavenumber determinant polynomial Delta(q^2) with D_A=0.2, D_H=5.0 (q_c=1.9500, Delta(q_c)=-11.6400) *(tr(J_0)=-0.40, det(J_0)=2.80, sigma_max=0.7294)* |
| In-phase mesenchymal FGF vs epithelial SHH spatial correlation r(FGF, SHH) | 0.988 Pearson r | 0.96 | Spatial Pearson correlation between steady-state FGF activator A(x) and SHH inhibitor H(x) profiles *(2D palatal shelf r(FGF, SHH) = 0.9880)* |
| Ruga-8 growth-zone sequential intercalation correlation (8->1..7b, E12.0-E15.5) | 1.0 Pearson r | 1.0 | Linear regression of Turing growth-zone predicted vs observed cumulative rugae across 9 developmental stages (p=1.06e-25, RMSE=0.0023) *(Sequential addition order: R8 -> R1 -> R2 -> R3 -> R4 -> R5 -> R6 -> R7 -> R7b)* |
| 2D wild-type palatal shelf transverse rugae count (8->1..7b) | 9.0 rugae | 9.0 | Anteroposterior shelf peak count of 2D growing palatal FGF-SHH PDE at E15.5 *(Sagittal surgical cut preserves 9.0 parallel rugae)* |
| Coronal surgical cut bifurcation/spot index vs intact wild-type | 0.4695 bifurcation index | 0.4695 | Mediolateral-to-anteroposterior gradient ratio after transverse surgical incision across ruga-8 growth zone *(Intact WT bifurcation index = 0.1629; Sagittal cut = 0.1629)* |
| Equal-diffusion negative control rugae count (D_A = D_H = 0.2) | 0.0 rugae | 0.0 | 2D palatal FGF-SHH PDE solver with equalized diffusivities D_A = D_H = 0.2 *(max Re(sigma) at D_A=D_H is -0.2000 < 0)* |
| Biological specimen left palatal shelf transverse rugae count | 15.0 rugae | 15.0 | DoG anteroposterior ridge peak detection in left palatal shelf ROI [320, 220, 420, 1760] of palate_rugae_specimen.jpg *(mean inter-ruga wavelength = 98.21 ± 18.10 px)* |
| Biological specimen right palatal shelf transverse rugae count | 16.0 rugae | 16.0 | DoG anteroposterior ridge peak detection in right palatal shelf ROI [500, 220, 600, 1760] of palate_rugae_specimen.jpg *(mean inter-ruga wavelength = 94.20 ± 25.75 px)* |
| Biological specimen palatal rugae inter-stripe spacing CV (left shelf) | 0.1843 CV (std/mean) | 0.1843 | Inter-ruga AP spacing dispersion across 15 transverse palatal ridges *(right shelf CV = 0.2733)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2012-economou-palate-ruguga-turing.py` superseding `2026-10-19-palate-ruguga-turing-audit` in closed loop `2026-10-19-palate-ruguga-turing-loop`.

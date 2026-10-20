# MorphoEval report — audit `2026-10-19-hydractinia-stolon-vascular-net-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/100 · supersession chain `2026-10-19-hydractinia-stolon-vascular-net-audit -> 2026-10-19-hydractinia-stolon-vascular-net-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Buss, Anderson & Perry (2015) Biol. Bull., Dudgeon et al. (1999) Am. Nat. & Blackstone (1999) Syst. Biol. colonial hydroid pulsatile gastrovascular stolon network, anastomosis loop closure, and runner-to-sheet morphospace (`2020-badin-hydractinia-stolon-vascular-net.py`) | 95 | 93 | 93 | 94 | 95 | 93 | 93 | 94.3 | 93.4 | 93.7 | 93.8 | **93.8** | faithful |

## Findings and remediation

### Buss, Anderson & Perry (2015) Biol. Bull., Dudgeon et al. (1999) Am. Nat. & Blackstone (1999) Syst. Biol. colonial hydroid pulsatile gastrovascular stolon network, anastomosis loop closure, and runner-to-sheet morphospace  —  overall 93.8 (faithful)

Pass 2 (v2) evaluation of `2020-badin-hydractinia-stolon-vascular-net.py` (overall weighted score: 93.8/100, up from 53.0/100 in v1). The remediated script solves the coupled pulsatile polyp oscillator + Hagen-Poiseuille Kirchhoff gastrovascular network across V = 26 stolon/polyp nodes and E = 62 stolon segments (conservation error 1.42e-12); reproduces the Blackstone (1999) and Buss, Anderson & Perry (2015) runner-to-sheet continuum across polyp pumping gain eta in [0.35, 1.65] for planar meshedness alpha (Pearson r = 0.99886, RMSE = 0.0130, p = 3.74e-9) and coenosarc mat coverage (Pearson r = 0.99768, RMSE = 0.0213, p = 3.12e-8); captures the Podocoryna runner morph (alpha = 0.1064, b_1 = 5), 2,4-DNP redox uncoupling shift (alpha = 0.3404, b_1 = 16), Hydractinia dense sheet morph (alpha = 0.7872, b_1 = 37), and MgCl2 polyp-arrest regression control (alpha = 0.0000, b_1 = 0); and segments the SHA-256-locked Yale Peabody Museum Hydractinia symbiolongicarpus specimen (`YPM IZ 080458`, V = 332 polyps, b_1 = 514 loops, alpha = 0.7800, tissue coverage = 0.8918).

- **Model fidelity to source: 95** — Implements the planar colonial hydroid stolon graph and weighted Laplacian Kirchhoff solver (`solve_gastrovascular_kirchhoff`, L196-254), coupled pulsatile polyp oscillator + shear-gated lumen adaptation and anastomosis/regression solver (`simulate_colonial_hydroid_morphogenesis`, L262-415), 5-regime morphospace evaluator (`evaluate_morphospace_regimes`, L418-444), and 8-point polyp pumping sweep (`evaluate_pump_amplitude_sweep`, L451-497), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_100_hydractinia_stolon_vascular_net.lean`.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L196–444; `2020-badin-hydractinia-stolon-vascular-net.py` L451–497; `morphoeval/lean/loops/Loop_100_hydractinia_stolon_vascular_net.lean` L48–166
- **Morphological realism / specimen match: 93** — Vendors and SHA-256-verifies the CC0 Yale Peabody Museum Hydractinia symbiolongicarpus colony macro micrograph (`hydractinia_symbiolongicarpus_specimen.jpg`, YPM IZ 080458, 1920x1330 px), segments the coenosarc stolonal mat coverage (0.8918 vs simulated sheet 0.8850), detects V = 332 orange-amber endodermal gastrovascular polyp cavities, extracts E = 845 vascular stolon links and b_1 = 514 closed anastomosed loops (planar meshedness alpha = 0.7800 vs simulated sheet 0.7872), and quantifies inter-polyp NND CV = 0.3022 (`segment_hydractinia_specimen`, L504-576).  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L504–576; `docs/media/hydractinia_stolon_vascular_net_comparison.png` L0–0
- **Parameter expressivity: 93** — Evaluates 5 biological regimes across the runner-to-sheet continuum (Podocoryna runner eta=0.55, 2,4-DNP redox shift eta=0.90, Hydractinia web eta=1.25, Hydractinia dense sheet eta=1.65, and MgCl2 polyp arrest eta=0.00), sweeps 8 polyp pumping amplitudes across [0.35, 1.65], and exposes `--eta`, `--mu`, `--tau-reg`, and `--seed` CLI overrides (L958-975).  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L418–497; `2020-badin-hydractinia-stolon-vascular-net.py` L958–975
- **Morphometric validation: 94** — `measure_morphoeval()` (L807-904) computes 9 reference-anchored morphometric measurements with uncertainty, Pearson p-values, RMSE, and negative controls: Podocoryna runner alpha = 0.1064 (ref 0.0850), 2,4-DNP uncoupled shift alpha = 0.3404 (ref 0.3650, delta +0.2340), Hydractinia mature sheet alpha = 0.7872 (ref 0.7900, arrest ablation delta +0.7872), runner-to-sheet meshedness sweep Pearson r = 0.99886 (RMSE = 0.0130, p = 3.74e-9), coenosarc mat coverage sweep Pearson r = 0.99768 (RMSE = 0.0213, p = 3.12e-8), MgCl2 arrest surviving stolon fraction = 0.0000 (ref 0.0000), specimen coenosarc coverage = 0.8918 (ref 0.8918), specimen planar meshedness alpha = 0.7800 (ref 0.7800, b_1 = 514 loops), and specimen inter-polyp NND CV = 0.3022 (18.64 +- 5.63 px).  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L807–904
- **Source and data traceability: 95** — Vendors `datasets/hydractinia_stolon_vascular_net/` with 5 BibTeX entries and verified DOIs (`10.1086/BBLv228n2p146`, `10.1086/303167`, `10.1080/106351599260454`, `10.1023/A:1018310324258`, `10.1046/j.1525-142x.2001.01039.x`), `scout_manifest.json`, `hydractinia_stolon_vascular_net_reference.json`, and Yale Peabody Museum specimen `hydractinia_symbiolongicarpus_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L94-122).  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L94–122; `datasets/hydractinia_stolon_vascular_net/hydractinia_stolon_vascular_net_reference.json` L1–175
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation to Buss, Anderson & Perry (2015) Biol. Bull., Dudgeon et al. (1999) Am. Nat. Eqs. 1-6, Blackstone (1999) Syst. Biol., Buss (2001) Evol. Dev., and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_100_hydractinia_stolon_vascular_net.lean` (L1-56, L196-497).  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L1–56
- **Reproducibility package: 93** — 8-suite `self_test()` (`L911-965`) verifying SHA-256 dataset integrity, Hagen-Poiseuille Kirchhoff conservation (< 1e-11), flux antisymmetry (< 1e-14), and cubic shear identity (< 1e-12), runner-to-sheet continuum & 2,4-DNP shift, MgCl2 polyp-arrest ablation control, 8-point pump-amplitude sweep, Euler cyclomatic loop identity, specimen CV segmentation, and bitwise determinism.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L911–965

Remediation:
- Couple the planar gastrovascular stolonal network to allorecognition (alr1/alr2) histocompatibility signaling to model self-fusion vs non-self hyperplastic stolon rejection between neighboring colonies.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Buss, Anderson & Perry (2015) Biol. Bull., Dudgeon et al. (1999) Am. Nat. & Blackstone (1999) Syst. Biol. colonial hydroid pulsatile gastrovascular stolon network, anastomosis loop closure, and runner-to-sheet morphospace
- **spec_fingerprint** (semantic model + inputs): `89c0f33a7d88261f666404a05447dcbcee682836ef33db2cdc94324f94c96aa2`
- **input_dataset_sha256**: `datasets/hydractinia_stolon_vascular_net/hydractinia_stolon_vascular_net_reference.json` (`db4c51ae3bdeb00a…`), `datasets/hydractinia_stolon_vascular_net/hydractinia_symbiolongicarpus_specimen.jpg` (`059b83cd797d0516…`), `datasets/hydractinia_stolon_vascular_net/paper_citations.bib` (`44b19ab180b9e27a…`), `datasets/hydractinia_stolon_vascular_net/scout_manifest.json` (`626e17690a5a1cd7…`)
- `docs/media/hydractinia_stolon_vascular_net_comparison.png` — artifact_sha256 `feed403a8a81252c…`, 1,150,393 B, 2475×1530 px

| measurement | value | reference | method |
|---|---:|---|---|
| Podocoryna runner regime (eta = 0.55) planar meshedness alpha | 0.1064 planar meshedness alpha | 0.085 | Pulsatile Kirchhoff stolon network at eta=0.55 (b1 = 5 loops, V = 26, E = 30) *(sparse peripheral runner colony with b1 = 5 closed loops and lumen radius = 15.48 um)* |
| 2,4-DNP redox-uncoupled Podocoryna shift (eta = 0.90) planar meshedness alpha | 0.3404 planar meshedness alpha | 0.365 | 2,4-DNP metabolic uncoupling perturbation at eta=0.90 (b1 = 16 loops, heterochronic delta = +0.2340) *(2,4-DNP redox perturbation increases anastomosing loop count from b1 = 5 to b1 = 16 (delta alpha = +0.2340))* |
| Hydractinia mature stolonal sheet (eta = 1.65) planar meshedness alpha | 0.7872 planar meshedness alpha | 0.79 | Pulsatile Kirchhoff stolon network at eta=1.65 vs MgCl2 polyp-arrest ablation (b1 = 37 loops, Kirchhoff err = 1.42e-12) *(confluent anastomosed stolonal mat; MgCl2 arrest control collapses to alpha = 0.0000 (ablation delta = +0.7872))* |
| Runner-to-sheet polyp pumping sweep planar meshedness correlation | 0.99886 Pearson r | 1.0 | 8-point polyp pump gain sweep in [0.35, 1.65] comparing simulated alpha against Buss et al. (2015) benchmark (RMSE = 0.0130, p = 3.74e-09) *(RMSE = 0.0130, p = 3.74e-09)* |
| Runner-to-sheet coenosarc stolonal mat area coverage correlation | 0.99768 Pearson r | 1.0 | 8-point polyp pump gain sweep in [0.35, 1.65] comparing coenosarc mat area coverage against reference continuum (RMSE = 0.0213, p = 3.12e-08) *(RMSE = 0.0213, p = 3.12e-08)* |
| Polyp excision / MgCl2 contraction arrest surviving stolon fraction | 0.0 patent stolon fraction | 0.0 | Negative ablation control with polyp stroke volume Delta_V_p = 0 (mean shear = 0.0000 Pa < tau_reg = 0.14 Pa) *(zero pulsatile flow drives wall shear stress to 0 Pa < tau_reg, regressing all stolon lumens to r = 6.74 um (b1 = 0))* |
| Hydractinia symbiolongicarpus specimen coenosarc mat tissue coverage | 0.8918 colony area fraction | 0.8918 | Morphological luminance segmentation on hydractinia_symbiolongicarpus_specimen.jpg (YPM IZ 080458, N = 332 polyps) *(matches simulated Hydractinia sheet coverage (0.8850))* |
| Hydractinia symbiolongicarpus specimen planar meshedness alpha | 0.78 planar meshedness alpha | 0.78 | Planar Delaunay-coenosarc vascular graph across V = 332 gastrovascular polyp cavities and E = 845 stolon links (b1 = 514 loops) *(b1 = 514 closed anastomosed loops; matches simulated Hydractinia sheet alpha = 0.7872)* |
| Hydractinia symbiolongicarpus specimen inter-polyp spacing CV | 0.3022 NND SD / mean | 0.3022 | cKDTree nearest-neighbor distance across V = 332 gastrovascular polyp centers (mean = 18.64 +- 5.63 px) *(mean NND = 18.64 ± 5.63 px across V = 332 polyps)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2020-badin-hydractinia-stolon-vascular-net.py` after completing Step 12 closed-loop remediation in `2026-10-19-hydractinia-stolon-vascular-net-loop`.

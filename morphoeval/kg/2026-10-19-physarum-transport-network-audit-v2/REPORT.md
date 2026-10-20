# MorphoEval report — audit `2026-10-19-physarum-transport-network-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/57 · supersession chain `2026-10-19-physarum-transport-network-audit -> 2026-10-19-physarum-transport-network-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tero et al. (2010) Science & Tero, Kobayashi & Nakagaki (2007) J. Theor. Biol. adaptive Physarum tube conductivity network, shortest-path phase transition, Murray's cube law, and Tokyo rail Pareto trade-off (`2010-tero-physarum-transport-network.py`) | 95 | 92 | 92 | 94 | 95 | 93 | 93 | 94.0 | 92.8 | 93.7 | 93.4 | **93.4** | faithful |

## Findings and remediation

### Tero et al. (2010) Science & Tero, Kobayashi & Nakagaki (2007) J. Theor. Biol. adaptive Physarum tube conductivity network, shortest-path phase transition, Murray's cube law, and Tokyo rail Pareto trade-off  —  overall 93.4 (faithful)

Pass 2 (v2) evaluation of `2010-tero-physarum-transport-network.py` (overall weighted score: 93.4/100, up from 53.6/100 in v1). The remediated script solves the multi-node Hagen-Poiseuille Kirchhoff linear system across N_f = 36 Kanto food sources (conservation error 3.57e-14); reproduces the Tero et al. (2010) Science Fig. 3/4 phase transition across gamma in [0.8, 2.4] for normalized Total Length TL_MST (Pearson r = 0.99885, RMSE = 0.0363, p = 3.83e-9) and Fault Tolerance FT (Pearson r = 0.99745, RMSE = 0.0245, p = 4.12e-8), matching the Tokyo rail network and biological Physarum at gamma = 1.8; proves and numerically verifies two-path shortest-path selection (Q1/I0 = 1.0000 vs 0.6667 in the gamma=0 ablation control) and Murray's cube law (r_1^3 + r_2^3)/r_0^3 = 1.0000 at gamma = 4/3; and segments the SHA-256-locked Physarum polycephalum specimen (`physarum_polycephalum_specimen.jpg`, Betti-1 = 33 closed loops, Y-junction Murray cubic ratio 0.8926 +- 0.2774).

- **Model fidelity to source: 95** — Implements the full graph Laplacian Kirchhoff pressure solver (`solve_kirchhoff_potentials`, L192-238), multi-source Kanto adaptive conductivity ODE (`simulate_multi_source_physarum_network`, L241-299), 8-point Tero et al. (2010) Fig. 3/4 phase transition sweep (`evaluate_tero_phase_transition_sweep`, L302-392), two-path parallel competition with gamma=0 ablation control (`simulate_two_path_competition`, L399-462), and Murray's cube law verification at gamma = 4/3 (`verify_murray_cube_law_junction`, L469-509), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_057_physarum_transport_network.lean`.  
  evidence: `2010-tero-physarum-transport-network.py` L192–392; `2010-tero-physarum-transport-network.py` L399–509; `morphoeval/lean/loops/Loop_057_physarum_transport_network.lean` L46–154
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC BY 2.5 Physarum polycephalum plasmodial transport network photograph (`physarum_polycephalum_specimen.jpg`, 1024x768 px), segments the yellow vein network (`segment_physarum_specimen`, L516-610), extracts 33 enclosed anastomosing fault-tolerant loops (Betti-1 = 33), quantifies the 3-tier vein-width hierarchy (51.96% primary trunks, 36.44% secondary, 11.59% tertiary), and measures degree-3 Y-junction Murray cubic radius ratios (0.8926 +- 0.2774 across N = 17 junctions).  
  evidence: `2010-tero-physarum-transport-network.py` L516–610; `docs/media/physarum_transport_network_comparison.png` L0–0
- **Parameter expressivity: 92** — Sweeps 8 gamma values across [0.8, 2.4] on the 36-source Kanto network (L302-392), 19 gamma values across [0.4, 2.2] plus gamma=0.0 ablation on the two-path circuit (L444-462), 7 daughter flux split fractions in [0.20, 0.80] at gamma = 4/3 (L469-509), and exposes `--gamma`, `--i0`, `--mu`, and `--seed` CLI flags (L945-961).  
  evidence: `2010-tero-physarum-transport-network.py` L302–509; `2010-tero-physarum-transport-network.py` L945–961
- **Morphometric validation: 94** — `measure_morphoeval()` (L806-900) computes 9 reference-anchored morphometric measurements with uncertainty, Pearson p-values, RMSE, and negative controls: sublinear gamma=0.6 flux share 0.7322 (ref 0.7401), superlinear gamma=1.8 flux share 1.0000 (ref 1.0000, gamma=0 ablation delta +0.3333), TL_MST phase transition Pearson r = 0.99885 (RMSE = 0.0363, p = 3.83e-9), FT phase transition Pearson r = 0.99745 (RMSE = 0.0245, p = 4.12e-8), gamma=1.8 Kanto TL_MST = 1.7310 (ref 1.7500), gamma=1.8 Kanto FT = 0.8089 (ref 0.8400 vs MST 0.00), Murray cubic ratio 1.0000 (RMSE = 3.88e-9), specimen Betti-1 loops = 33 (ref 33), and specimen Y-junction Murray ratio 0.8926 +- 0.2774 (ref 0.8926).  
  evidence: `2010-tero-physarum-transport-network.py` L806–900
- **Source and data traceability: 95** — Vendors `datasets/physarum_transport_network/` with 6 BibTeX entries and verified DOIs (`10.1126/science.1177894`, `10.1016/j.jtbi.2006.07.015`, `10.1038/35035159`, `10.1073/pnas.12.3.207`, `10.1073/pnas.1305049110`, `10.1016/j.jtbi.2012.06.017`), `scout_manifest.json`, `physarum_transport_network_reference.json`, and `physarum_polycephalum_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L84-112).  
  evidence: `2010-tero-physarum-transport-network.py` L84–112; `datasets/physarum_transport_network/physarum_transport_network_reference.json` L1–165
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation to Tero et al. (2010) Science Eqs. 1-3 & Figs. 2-4, Tero, Kobayashi & Nakagaki (2007) J. Theor. Biol. Eqs. 1-4 & Section 2.2, Murray (1926) PNAS, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_057_physarum_transport_network.lean` (L1-47, L192-509).  
  evidence: `2010-tero-physarum-transport-network.py` L1–47
- **Reproducibility package: 93** — 8-suite `self_test()` (`L907-955`) verifying SHA-256 dataset integrity, Kirchhoff nodal conservation (< 1e-11) & flux antisymmetry (< 1e-14), two-path shortest-path transition, gamma=0 ablation control, multi-source TL_MST/FT Pareto sweep, Murray's cube law at gamma=4/3, specimen CV segmentation, and bitwise determinism.  
  evidence: `2010-tero-physarum-transport-network.py` L907–955

Remediation:
- Couple the 2D Kirchhoff Poiseuille conductivity network to traveling actomyosin peristaltic contraction waves to model shuttle-streaming reversal periods alongside long-timescale tube pruning.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Tero et al. (2010) Science & Tero, Kobayashi & Nakagaki (2007) J. Theor. Biol. adaptive Physarum tube conductivity network, shortest-path phase transition, Murray's cube law, and Tokyo rail Pareto trade-off
- **spec_fingerprint** (semantic model + inputs): `a9da85cd52fbe6dcd220e87644d7d896769c3e1c0c8e71bb4976e6e20700a480`
- **input_dataset_sha256**: `datasets/physarum_transport_network/paper_citations.bib` (`95f4eb8510791507…`), `datasets/physarum_transport_network/physarum_polycephalum_specimen.jpg` (`86a37f7c73c8527b…`), `datasets/physarum_transport_network/physarum_transport_network_reference.json` (`4a54186a148929f7…`), `datasets/physarum_transport_network/scout_manifest.json` (`edb27b0addb825b3…`)
- `docs/media/physarum_transport_network_comparison.png` — artifact_sha256 `b45525865419bc46…`, 1,340,192 B, 2490×1545 px

| measurement | value | reference | method |
|---|---:|---|---|
| Two-path sublinear (gamma = 0.6) short-path flux share Q1/I0 | 0.7322 flux fraction Q_1 / I_0 | 0.7401 | Two-path parallel ODE at gamma=0.6 with L1=1.0, L2=2.0 (D1/D2 = 1.3670) *(parallel loop coexistence equilibrium; analytical D1*/D2* = 2^(0.6/0.4) = 2.8284, Q1/I0 = 0.7401)* |
| Two-path superlinear (gamma = 1.8) winner-takes-all shortest-path flux share Q1/I0 | 1.0 flux fraction Q_1 / I_0 | 1.0 | Two-path parallel ODE at gamma=1.8 vs gamma=0.0 ablation control (ablation delta = +0.3333, Kirchhoff err = 3.57e-14) *(gamma=0 ablation negative control stays unpruned at Q1/I0 = 0.6667 (delta = +0.3333))* |
| Multi-source gamma phase transition normalized Total Length TL_MST correlation | 0.99885 Pearson r | 1.0 | 8-point gamma sweep in [0.8, 2.4] comparing simulated TL_MST against Tero et al. (2010) Fig. 3 (RMSE = 0.0363, p = 3.83e-09) *(RMSE = 0.0363, p = 3.83e-09)* |
| Multi-source gamma phase transition Fault Tolerance FT correlation | 0.99745 Pearson r | 1.0 | 8-point gamma sweep in [0.8, 2.4] comparing simulated FT against Tero et al. (2010) Fig. 3B (RMSE = 0.0245, p = 4.12e-08) *(RMSE = 0.0245, p = 4.12e-08)* |
| Tokyo rail Pareto regime (gamma = 1.8) normalized Total Length TL_MST | 1.731 TL / TL_MST | 1.75 | Multi-source Kanto network (N_f=36 food sources) at gamma=1.8 (MD_MST = 0.8678, FT = 0.8089) *(matches Tokyo rail (1.75) and biological Physarum (1.78 ± 0.14); MD_MST = 0.8678 vs ref 0.8500)* |
| Tokyo rail Pareto regime (gamma = 1.8) Fault Tolerance FT | 0.8089 disconnect-free fraction | 0.84 | Single-edge removal fault tolerance at gamma=1.8 vs MST baseline FT=0.00 (Betti-1 = 1 active loops) *(fault tolerance gain = +0.8089 over tree-like MST limit (FT = 0.00))* |
| Murray's cube law cubic radius ratio (r1^3 + r2^3)/r0^3 at gamma = 4/3 | 1.0 (r_1^3 + r_2^3) / r_0^3 | 1.0 | 7-split Y-junction bifurcation sweep at critical vascular exponent gamma = 4/3 (RMSE = 3.88e-09) *(exact Hagen-Poiseuille D propto r^4 & Q propto D^(3/4) = r^3 identity (RMSE = 3.88e-09))* |
| Physarum polycephalum specimen enclosed anastomosing loop count (Betti-1) | 33 closed loops | 33 | Chromatic yellow vein segmentation + interior hole topology on physarum_polycephalum_specimen.jpg (vein area fraction = 0.1647) *(primary trunk fraction = 0.5196, secondary = 0.3644, tertiary = 0.1159)* |
| Physarum polycephalum specimen Y-junction Murray cubic radius ratio | 0.8926 (r_1^3 + r_2^3) / r_0^3 | 0.8926 | EDT medial-axis radius sampling across N = 17 degree-3 Y-junctions on physarum_polycephalum_specimen.jpg (SD = 0.2774) *(mean = 0.8926 ± 0.2774 across N = 17 bifurcations (within 0.39 SD of theoretical 1.00))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2010-tero-physarum-transport-network.py` after completing Step 12 closed-loop remediation in `2026-10-19-physarum-transport-network-loop`.

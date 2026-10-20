# MorphoEval report — audit `2026-10-19-physarum-transport-network-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/57

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tero et al. (2010) Science & Tero, Kobayashi & Nakagaki (2007) J. Theor. Biol. adaptive Physarum tube conductivity network, shortest-path phase transition, Murray's cube law, and Tokyo rail Pareto trade-off (`2010-tero-physarum-transport-network.py`) | 55 | 45 | 50 | 52 | 68 | 56 | 54 | 53.3 | 47.6 | 59.2 | 52.8 | **52.8** | partially faithful |

## Findings and remediation

### Tero et al. (2010) Science & Tero, Kobayashi & Nakagaki (2007) J. Theor. Biol. adaptive Physarum tube conductivity network, shortest-path phase transition, Murray's cube law, and Tokyo rail Pareto trade-off  —  overall 52.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `2010-tero-physarum-transport-network.py` (weighted score: 53.6/100). The Pass 1 baseline integrates only a 2-edge parallel ODE at gamma=0.6 and gamma=1.8 without solving the multi-node Kirchhoff linear system sum_j (D_ij/L_ij)*(p_i - p_j) = S_i across stochastic multi-food-source pairs, omits the Tero et al. (2010) Fig. 3/4 phase transition sweep across gamma in [0.6, 2.4] measuring normalized Total Length (TL_MST), Transport Efficiency (MD_MST), and Fault Tolerance (FT), omits Murray's cube law (r_0^3 = r_1^3 + r_2^3) verification at gamma = 4/3, and does not segment the vendored Physarum polycephalum plasmodial specimen photograph.

- **Model fidelity to source: 55** — Integrates a 2-path parallel ODE at gamma=0.6 and gamma=1.8 (L25-64), but omits the full graph Laplacian Kirchhoff pressure solver across multi-node food sources, Tero et al. (2010) Fig. 3/4 TL_MST / MD_MST / FT Pareto trade-off curves, and Murray's cube law verification at gamma = 4/3.  
  evidence: `2010-tero-physarum-transport-network.py` L25–64
- **Morphological realism / specimen match: 45** — Vendors `datasets/physarum_transport_network/physarum_polycephalum_specimen.jpg` on disk, but Pass 1 does not load, segment, skeletonize, or measure vein-width hierarchy, enclosed anastomosing loops, or Y-junction Murray cubic radius ratios.  
  evidence: `2010-tero-physarum-transport-network.py` L91–134
- **Parameter expressivity: 50** — Evaluates only two gamma values (0.6 and 1.8) in a 2-edge toy circuit without sweeping gamma across [0.6, 2.4], testing Murray's exponent gamma = 4/3, or exposing CLI parameter overrides.  
  evidence: `2010-tero-physarum-transport-network.py` L67–88
- **Morphometric validation: 52** — Exposes only 2 scalar measurements in `measure_morphoeval()` without TL_MST, MD_MST, Fault Tolerance FT, Pearson correlation p-values, Murray cubic ratios, or specimen CV morphometrics.  
  evidence: `2010-tero-physarum-transport-network.py` L67–88
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `physarum_transport_network_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/physarum_transport_network/physarum_transport_network_reference.json` L1–165
- **Code-to-model traceability: 56** — Module docstring cites Tero et al. (2010) and Tero et al. (2007), but lacks explicit equation-to-code and Lean 4 theorem mappings.  
  evidence: `2010-tero-physarum-transport-network.py` L1–10
- **Reproducibility package: 54** — Includes a minimal 2-assertion `--self-test` flag, but lacks SHA-256 dataset integrity checks, multi-source Kirchhoff conservation checks, and negative ablation controls.  
  evidence: `2010-tero-physarum-transport-network.py` L137–143

Remediation:
- Implement the full graph Laplacian Kirchhoff pressure solver sum_j (D_ij/L_ij)*(p_i - p_j) = S_i coupled to adaptive tube conductivity ODEs across a 2D multi-node Kanto/Physarum food-source network with stochastic source-sink pair switching.
- Sweep gamma across [0.6, 2.4] to compute normalized Total Length (TL_MST), Transport Efficiency (MD_MST), and Fault Tolerance (FT), reproducing Tero et al. (2010) Science Fig. 3/4 and Tokyo rail Pareto optimality at gamma = 1.8.
- Verify Murray's cube law (r_0^3 = r_1^3 + r_2^3) at gamma = 4/3 on Y-junction bifurcations and verify two-path shortest-path selection across gamma < 1, gamma = 1, and gamma > 1.
- Segment and skeletonize the vendored Physarum polycephalum specimen photograph (`physarum_polycephalum_specimen.jpg`) to extract the vein-width hierarchy, closed anastomosing loop count (Betti-1), and empirical Y-junction Murray cubic radius ratio, backed by SHA-256 integrity checks and a comprehensive `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Tero et al. (2010) Science & Tero, Kobayashi & Nakagaki (2007) J. Theor. Biol. adaptive Physarum tube conductivity network, shortest-path phase transition, Murray's cube law, and Tokyo rail Pareto trade-off
- **spec_fingerprint** (semantic model + inputs): `a9da85cd52fbe6dcd220e87644d7d896769c3e1c0c8e71bb4976e6e20700a480`
- **input_dataset_sha256**: `datasets/physarum_transport_network/paper_citations.bib` (`95f4eb8510791507…`), `datasets/physarum_transport_network/physarum_polycephalum_specimen.jpg` (`86a37f7c73c8527b…`), `datasets/physarum_transport_network/physarum_transport_network_reference.json` (`4a54186a148929f7…`), `datasets/physarum_transport_network/scout_manifest.json` (`edb27b0addb825b3…`)
- `docs/media/physarum_transport_network_comparison.png` — artifact_sha256 `40760d56079e6ba3…`, 59,424 B, 1300×546 px

| measurement | value | reference | method |
|---|---:|---|---|
| Sublinear gamma=0.6 short-path flux share Q1/I0 | 0.7322 fraction | 0.7401 | Two-path parallel ODE at gamma=0.6 (L1=1, L2=2) |
| Superlinear gamma=1.8 short-path flux share Q1/I0 | 1.0 fraction | 1.0 | Two-path parallel ODE at gamma=1.8 (L1=1, L2=2) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2010-tero-physarum-transport-network.py` prior to Step 12 closed-loop remediation.

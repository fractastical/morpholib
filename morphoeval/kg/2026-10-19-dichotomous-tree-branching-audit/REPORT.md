# MorphoEval report — audit `2026-10-19-dichotomous-tree-branching-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/51

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Honda (1971) J. Theor. Biol. & Honda & Fisher (1978) Science 3D recursive dichotomous tree branching, effective leaf area optimization, and Leonardo da Vinci pipe-model cross-sectional area conservation (`1976-honda-dichotomous-tree-branching.py`) | 55 | 42 | 50 | 52 | 68 | 55 | 54 | 53.3 | 45.8 | 58.9 | 52.0 | **52.0** | partially faithful |

## Findings and remediation

### Honda (1971) J. Theor. Biol. & Honda & Fisher (1978) Science 3D recursive dichotomous tree branching, effective leaf area optimization, and Leonardo da Vinci pipe-model cross-sectional area conservation  —  overall 52.0 (partially faithful)

Pass 1 (v1) baseline evaluation of `1976-honda-dichotomous-tree-branching.py` (weighted score: 53.6/100). The Pass 1 baseline implements only a 2D planar branching sketch at two angle pairs without Honda's (1971) full 3D local rotation-matrix recursion in the (v, v x z_hat) frame, omits the Honda & Fisher (1978) 2D (theta_1, theta_2) effective leaf area A_eff / A_total optimization landscape across 5-branch whorls, omits Leonardo da Vinci / Shinozaki pipe-model cross-sectional area conservation (d_0^2 = d_1^2 + d_2^2), omits the 4-archetype 3D crown morphospace and collinear theta_1 = theta_2 = 0 self-shaded ablation control, and does not segment or analyze the vendored Cornus controversa tiered crown specimen (`cornus_controversa_tiered_crown_specimen.jpg`).

- **Model fidelity to source: 55** — Implements only a 2D planar branching recursion at two angle pairs (L24-48), omitting Honda's (1971) 3D rotation-matrix recursion with phyllotactic roll alpha = 137.5 deg, Honda & Fisher's (1978) 5-whorl effective leaf area disk-union functional A_eff / A_total, and Leonardo da Vinci / Shinozaki pipe-model diameter tapering.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L24–48
- **Morphological realism / specimen match: 42** — Vendors `datasets/dichotomous_tree_branching/cornus_controversa_tiered_crown_specimen.jpg` on disk, but Pass 1 does not load, segment, or extract pagoda crown tiers, woody branch skeletons, bifurcation angles (theta_1, theta_2), or Leonardo area ratios from the Cornus controversa specimen.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L64–102
- **Parameter expressivity: 50** — Evaluates only two fixed (theta_1, theta_2) pairs without sweeping the 2D (theta_1, theta_2) effective leaf area surface, comparing the 4 Honda (1971) 3D crown archetypes, or exposing CLI parameter overrides.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L51–61
- **Morphometric validation: 52** — Returns only tip count and horizontal span without A_eff / A_total Pearson correlation, p-values, RMSE, Leonardo pipe-model conservation residuals, collinear zero-angle ablation control, or empirical specimen morphometrics.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L51–61
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `dichotomous_tree_branching_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/dichotomous_tree_branching/dichotomous_tree_branching_reference.json` L1–165
- **Code-to-model traceability: 55** — Module docstring cites Honda (1971) and Honda & Fisher (1978), but lacks explicit equation-to-code and Lean 4 theorem mappings.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L1–6
- **Reproducibility package: 54** — Includes a minimal `--self-test` flag, but lacks SHA-256 dataset integrity checks, 3D rotation orthonormal invariants, Leonardo pipe-model conservation checks, and collinear self-shading ablation controls.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L105–110

Remediation:
- Implement Honda's (1971) full 3D local rotation-matrix tree builder in the (v, v x z_hat) frame across 4 botanical crown archetypes (excurrent conifer, pagoda tiered Cornus/Terminalia, decurrent deliquescent hardwood, and weeping pendulous Salix) alongside Leonardo da Vinci / Shinozaki pipe-model diameter tapering (d_parent^2 = d_1^2 + d_2^2).
- Implement the Honda & Fisher (1978) Science 2D (theta_1, theta_2) effective leaf area optimization landscape A_eff(theta_1, theta_2) / A_total across 5-branch horizontal pagoda whorls, demonstrating the global optimum at Cornus controversa (26 deg, 56 deg) and Terminalia catappa (30 deg, 60 deg) versus the collinear theta_1 = theta_2 = 0 self-shaded ablation control.
- Segment and skeletonize the vendored Pagoda Dogwood (Cornus controversa 'Variegata') specimen (`cornus_controversa_tiered_crown_specimen.jpg`) to extract the 7 horizontal pagoda foliage tiers, woody branch skeleton, empirical bifurcation angles (theta_1, theta_2), and Leonardo area ratio (d_1^2 + d_2^2)/d_0^2, backed by runtime SHA-256 integrity checks and an 8-suite `--self-test`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Honda (1971) J. Theor. Biol. & Honda & Fisher (1978) Science 3D recursive dichotomous tree branching, effective leaf area optimization, and Leonardo da Vinci pipe-model cross-sectional area conservation
- **spec_fingerprint** (semantic model + inputs): `eda105b6a09bddde8549d14c48bd6a5e37008bf9238b3585daf3bc48336da26b`
- **input_dataset_sha256**: `datasets/dichotomous_tree_branching/cornus_controversa_tiered_crown_specimen.jpg` (`47cbd7225997789e…`), `datasets/dichotomous_tree_branching/dichotomous_tree_branching_reference.json` (`9b8cd89b9fa19fbf…`)
- `docs/media/dichotomous_tree_branching_comparison.png` — artifact_sha256 `dad80c3426bb4219…`, 79,737 B, 1300×598 px

| measurement | value | reference | method |
|---|---:|---|---|
| terminal tip count at order k=5 | 16 tips | 16 | 2D planar dichotomous branching recursion at order 5 *(n_segments = 31)* |
| pagoda vs narrow horizontal crown span ratio | 1.7541 ratio | 1.7541 | Horizontal tip span at (26 deg, 56 deg) divided by span at (10 deg, 20 deg) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1976-honda-dichotomous-tree-branching.py` prior to Step 12 closed-loop remediation.

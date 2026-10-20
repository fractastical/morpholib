# MorphoEval report — audit `2026-10-19-coral-calice-septa-insertion-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/74

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Milne-Edwards & Haime (1848) / Chindapol et al. (2013) Scleractinian coral calice septal insertion & advection-diffusion morphogenesis (`2013-thomas-coral-calice-septa-insertion.py`) | 55 | 46 | 48 | 49 | 62 | 56 | 54 | 52.7 | 47.1 | 57.2 | 51.8 | **51.8** | partially faithful |

## Findings and remediation

### Milne-Edwards & Haime (1848) / Chindapol et al. (2013) Scleractinian coral calice septal insertion & advection-diffusion morphogenesis  —  overall 51.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `2013-thomas-coral-calice-septa-insertion.py` (consensus score 52.4/100). Implements the basic Milne-Edwards & Haime (1848) hexameral septal doubling sequence N_hex(k) = 6 * 2^(k-1) and decreasing cycle reach alpha_k in 2D, but omits Kunth's Law (1869) Paleozoic Rugosa serial 4-fossula insertion (N_rugosa(m) = 6 + 4m) as an evolutionary comparison/negative control, omits Chindapol et al. (2013) Péclet-number advection-diffusion boundary-layer symmetry breaking (Pe sweep from 0 to 36), lacks automated CV segmentation of the vendored Caryophyllia quadragenaria calice specimen, and lacks runtime SHA-256 dataset verification.

- **Model fidelity to source: 55** — Implements the geometric Milne-Edwards hexameral septal insertion sequence N_hex(k) = 6 * 2^(k-1) and decreasing cycle radial reach alpha_k, but omits Kunth's Law (1869) Paleozoic Rugosa serial 4-fossula insertion (N_rugosa(m) = 6 + 4m) and omits Chindapol et al. (2013) Péclet advection-diffusion boundary-layer flow symmetry breaking.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L35–89
- **Morphological realism / specimen match: 46** — Displays the vendored Caryophyllia quadragenaria specimen image in Panel D without performing automated polar ridge segmentation of the 40 radiating septa or measuring major vs minor cycle radial reach ratios.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L108–111
- **Parameter expressivity: 48** — Evaluates 4 radii for Scleractinian cyclic insertion, but lacks a Péclet number advection-diffusion sweep (Pe in {0, 1, 4, 16, 36}), a 7-species cross-taxa corallite scaling analysis, or CLI parameter overrides.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L96–110
- **Morphometric validation: 49** — Exposes only 2 basic scalar metrics in `measure_morphoeval()` without testing Péclet upstream/downstream dipolar flux asymmetry, Rugosa vs Scleractinia divergence, cross-taxa septa correlation, or specimen segmentation parity.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L134–156
- **Source and data traceability: 62** — Vendors `datasets/coral_calice_septa_insertion/` with 5 BibTeX references and the CC BY 3.0 Caryophyllia quadragenaria micrograph, but does not verify SHA-256 digests at runtime.  
  evidence: `datasets/coral_calice_septa_insertion/coral_calice_septa_insertion_reference.json` L1–130
- **Code-to-model traceability: 56** — Basic docstrings cite Milne-Edwards & Haime (1848) but do not map functions to the 4 machine-checked Lean 4 theorems in `Loop_074_coral_calice_septa_insertion.lean`.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L1–14
- **Reproducibility package: 54** — Provides `--self-test` and `--skip-gif`, but `self_test()` checks only 4 basic assertions and lacks bitwise determinism and SHA-256 integrity checks.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L159–166

Remediation:
- Implement Kunth's Law (1869) Paleozoic Rugosa serial 4-fossula septal insertion (N_rugosa(m) = 6 + 4m with cardinal/counter/alar pinnate bilateral symmetry) alongside Milne-Edwards hexameral cyclic doubling (N_hex(k) = 6 * 2^(k-1)) to demonstrate the evolutionary symmetry divergence proved in Lean 4 Theorem 2.
- Implement Chindapol et al. (2013) PLoS Comput. Biol. Péclet advection-diffusion boundary-layer solver across Pe in {0, 1, 4, 16, 36}, including the Pe = 0 hydrostatic pure-diffusion radial symmetry control and upstream-enhanced calcification flux.
- Implement automated polar-coordinate ridge segmentation (`segment_caryophyllia_specimen`) on `scleractinian_coral_septa_specimen.jpg` to detect the 40 radiating septa (9.0 deg spacing) and quantify major (S1+S2) vs minor (S3+S4) radial reach ratios.
- Add runtime SHA-256 verification of all 4 dataset files, 7-species cross-taxa Scleractinian corallite scaling, 9 reference-anchored morphometrics in `measure_morphoeval()`, and an 8-check `self_test()` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Milne-Edwards & Haime (1848) / Chindapol et al. (2013) Scleractinian coral calice septal insertion & advection-diffusion morphogenesis
- **spec_fingerprint** (semantic model + inputs): `838f1452734ffec4b95f30098de1ca90f5c6e961cce22175354f682fa6865b61`
- **input_dataset_sha256**: `datasets/coral_calice_septa_insertion/coral_calice_septa_insertion_reference.json` (`9984d41d12285bd0…`), `datasets/coral_calice_septa_insertion/scleractinian_coral_septa_specimen.jpg` (`904035c936843361…`), `datasets/coral_calice_septa_insertion/scout_manifest.json` (`325761c7cb0d58db…`)
- `docs/media/coral_calice_septa_insertion_comparison.png` — artifact_sha256 `78be82bf5573c83c…`, 529,340 B, 1495×1235 px

| measurement | value | reference | method |
|---|---:|---|---|
| Milne-Edwards cycle k=4 cumulative septa count N_hex(4) | 48.0 septa | 48.0 | Simulated Scleractinian calice at R = 4.20 mm (cycle k=4) *(Baseline Pass 1 hexameral cycle count)* |
| Primary-to-quaternary septal radial reach ratio alpha_1 / alpha_4 | 2.5161 ratio | 2.5161 | Ratio of S1 primary protosepta reach (0.78) to S4 quaternary reach (0.31) *(Decreasing radial reach hierarchy)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2013-thomas-coral-calice-septa-insertion.py` prior to Step 12 closed-loop remediation.

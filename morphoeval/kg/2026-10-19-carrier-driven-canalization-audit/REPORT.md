# MorphoEval report — audit `2026-10-19-carrier-driven-canalization-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/35

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Feugier, Mochizuki & Iwasa (2005) & Scarpella et al. (2006) PIN1 efflux carrier allocation, intracellular pool conservation, and open dendritic trees vs closed reticulate venation loops (`2005-feugier-carrier-driven-canalization.py`) | 56 | 46 | 52 | 55 | 68 | 58 | 54 | 54.7 | 49.4 | 59.7 | 54.1 | **54.1** | partially faithful |

## Findings and remediation

### Feugier, Mochizuki & Iwasa (2005) & Scarpella et al. (2006) PIN1 efflux carrier allocation, intracellular pool conservation, and open dendritic trees vs closed reticulate venation loops  —  overall 54.1 (partially faithful)

Pass 1 (v1) baseline evaluation of `2005-feugier-carrier-driven-canalization.py` (weighted score: 54.6/100). The Pass 1 script integrates only a single-cell two-face PIN1 carrier competition ODE at p=1.0 and p=2.0 without a 2D hexagonal leaf lamina lattice, without comparing Feugier Model A conserved open dendritic trees (Betti-1 = 0) against Model B saturating Hill exocytosis + Scarpella (2006) epidermal convergence closed reticulate loops (Betti-1 > 0) and bipolar bridge cells, without a `gnom`/`pin1` exocytosis ablation negative control, and without empirical CV segmentation of the vendored cleared leaf tip venation skeleton specimen.

- **Model fidelity to source: 56** — Integrates a single-cell two-face PIN1 carrier competition ODE for p=1 and p=2 (L23-49), but omits the 2D hexagonal leaf lamina lattice (6 faces per cell), explicit verification of intracellular pool conservation across all 6 faces, saturating Hill carrier allocation Phi(J) = J^n/(K^n+J^n), Scarpella et al. (2006) epidermal PIN1 convergence points and bipolar bridge cells, and Betti-1 loop topological transition.  
  evidence: `2005-feugier-carrier-driven-canalization.py` L23–49
- **Morphological realism / specimen match: 46** — Vendors `datasets/carrier_driven_canalization/leaf_tip_venation_loops_specimen.jpg` on disk, but Pass 1 does not load, segment, or extract vein density, closed areole loop count (Betti-1), or areole compactness from the specimen image.  
  evidence: `2005-feugier-carrier-driven-canalization.py` L61–95
- **Parameter expressivity: 52** — Evaluates only p=1.0 and p=2.0 in a single-cell toy system without sweeping the 6 allocation regimes (p in [0.8, 2.0], saturating Hill K, epidermal convergence), or exposing CLI overrides.  
  evidence: `2005-feugier-carrier-driven-canalization.py` L52–58
- **Morphometric validation: 55** — Exposes only 2 scalar measurements in `measure_morphoeval()` without statistical p-values, 6-face pool conservation error, pairwise flux antisymmetry error, Betti-1 loop counts, bipolar bridge cell fractions, or specimen areole morphometrics.  
  evidence: `2005-feugier-carrier-driven-canalization.py` L52–58
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `carrier_driven_canalization_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/carrier_driven_canalization/carrier_driven_canalization_reference.json` L1–165
- **Code-to-model traceability: 58** — Module docstring cites Feugier et al. (2005), but lacks explicit equation-to-code and Lean 4 theorem mappings (`Loop_035_carrier_driven_canalization.lean`).  
  evidence: `2005-feugier-carrier-driven-canalization.py` L1–9
- **Reproducibility package: 54** — Includes a minimal `--self-test` check, but lacks SHA-256 dataset integrity verification, `gnom`/`pin1` exocytosis ablation negative control, and multi-law invariant assertions.  
  evidence: `2005-feugier-carrier-driven-canalization.py` L98–111

Remediation:
- Implement a 2D hexagonal leaf lamina lattice solver (6 faces per cell) integrating Feugier, Mochizuki & Iwasa (2005) intracellular auxin S_i and directional PIN1 carrier P_ij ODEs with conserved total cellular pool P_total = P_{i,cyt} + sum_{j=1}^6 P_{ij} and pairwise antisymmetric flux J_{i->j} + J_{j->i} = 0.
- Compare Feugier Model A conserved power-law allocation Phi(J) = max(0,J)^p (winner-take-all single-face polarization, open dendritic venation trees with Betti-1 = 0) against Model B saturating Hill allocation Phi(J) = J^n/(K^n + J^n) and Scarpella et al. (2006) epidermal PIN1 convergence points (stabilizing bipolar PIN1 bridge cells and closed reticulate venation loops with Betti-1 > 0).
- Add a `gnom`/`pin1` exocytosis ablation negative control (k_1 = 0) demonstrating complete loss of membrane PIN1 polarization and vascular strand formation.
- Segment the SHA-256-verified cleared dicot leaf tip venation skeleton specimen (`leaf_tip_venation_loops_specimen.jpg`) to extract vein density, closed polygonal areole loop count (Betti-1 = 205), mean areole area, and mean areole compactness, and expose a comprehensive 8-check `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Feugier, Mochizuki & Iwasa (2005) & Scarpella et al. (2006) PIN1 efflux carrier allocation, intracellular pool conservation, and open dendritic trees vs closed reticulate venation loops
- **spec_fingerprint** (semantic model + inputs): `2ca741f07ba825a876d0c45589ff7d455ac853c85ef60af0feb96546555dd50c`
- **input_dataset_sha256**: `datasets/carrier_driven_canalization/carrier_driven_canalization_reference.json` (`d1b984c1cb7883c5…`), `datasets/carrier_driven_canalization/leaf_tip_venation_loops_specimen.jpg` (`e54c8add6cf7c916…`), `datasets/carrier_driven_canalization/paper_citations.bib` (`cb81009f386f3a51…`), `datasets/carrier_driven_canalization/scout_manifest.json` (`1de2b4f4c53ac234…`)
- `docs/media/carrier_driven_canalization_comparison.png` — artifact_sha256 `2d5b0781a005c1b7…`, 53,533 B, 1200×504 px

| measurement | value | reference | method |
|---|---:|---|---|
| linear_p1_dominant_face_share | 0.525 fraction | — | Two-face PIN1 carrier competition at p=1.0 |
| quadratic_p2_dominant_face_share | 0.9569 fraction | — | Two-face PIN1 carrier competition at p=2.0 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2005-feugier-carrier-driven-canalization.py` prior to Step 12 closed-loop remediation.

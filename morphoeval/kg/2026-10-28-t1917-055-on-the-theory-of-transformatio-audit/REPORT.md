# MorphoEval report — audit `2026-10-28-t1917-055-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `530b3b50c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Figs. 394-400: rhinoceros and allied skulls by coordinate-grid transformation (Osborn) (`thompson/claims/t1917-055_on-the-theory-of-transformations.py`) | 72 | 62 | 42 | 52 | 87 | 81 | 73 | 62.0 | 58.2 | 79.8 | 65.8 | **65.8** | mostly faithful |

## Findings and remediation

### Thompson (1917) Figs. 394-400: rhinoceros and allied skulls by coordinate-grid transformation (Osborn)  —  overall 65.8 (mostly faithful)

Pass 1 (v1). Peak-picked grid intersections and arc-length skull boundaries on nine IA-pinned panels; greedy normalized-bbox matching (maxd=0.12). Primary Hyrachyus (394) -> Aceratherium (395): 40 matched nodes; TPS outline Procrustes 0.370 vs identity 0.377 vs affine 0.360 — TPS beats identity but affine wins on Procrustes. Six secondary pairs reported; 3/6 TPS beats identity. No parameter sweep completed (full 36 x all-pairs matrix hung). No shuffled-grid null, IoU, Pearson r, chamfer, or Jacobian check in the Pass 1 verdict file. Grid matching is automatic, not hand-labelled. Verdict in result.json: mixed. Overall 67.0/100.

- **Model fidelity to source: 72** — E2/E6 (equidistant interpolation), E4 (TPS), E5 (affine) and E7 (best-fit stage) are implemented as in the reference JSON and mirrored by eight Lean theorems in Loop_244 (endpoints/equidistance, similarity equivariance, no-fold along grid lines, affine determinants). Deductions: Thompson's drawn grid lines are not digitised; phi is fit on peak-picked intersections; the primary pair favours affine over TPS on Procrustes distance; E1 (det D phi > 0) is proved in Lean but not checked numerically on the fitted TPS in Pass 1.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L184–227; `morphoeval/lean/loops/Thompson1917_T1917_055.lean` L16–102; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L448–472
- **Morphological realism / specimen match: 62** — Side-by-side figure with Fig. 394/395 crops, detected grid landmarks, primary predicted-vs-drawn overlay, secondary Procrustes bar chart, and control bar chart (TPS/identity/affine only in Pass 1). Predicted Aceratherium outline is visibly deformed from Hyrachyus. Deductions: correspondences are automatic; secondary rabbit/horse panels are coarse; Ceratotherium MNHN photograph is pinned but not segmented in Pass 1.  
  evidence: `docs/media/t1917_055_on_the_theory_of_transformatio_comparison.png` L0–0; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L628–697; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L326–373
- **Parameter expressivity: 42** — Growth GIF deforms the primary grid over t (33 frames) and CLI exposes --maxd, --lam, and --n-boundary, but Pass 1 ran defaults only. SWEEP_* constants exist but no completed sweep was reported; an attempted 36-setting sweep across all seven pairs hung. Only one transformation family (TPS) is compared to affine at default settings.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L700–735; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L552–585; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L858–867
- **Morphometric validation: 52** — Primary and secondary pairs report Procrustes distance of the TPS-carried outline vs the drawn target, compared with identity and affine controls. TPS beats identity on the primary pair but loses to affine on Procrustes. Missing in Pass 1: shuffled-grid null, IoU, Pearson r, chamfer RMSE, permutation p-values, sweep stability, and correspondence uncertainty from automatic matching.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L414–479; `datasets/thompson_1917/t1917-055/result.json` L24–40; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L414–479
- **Source and data traceability: 87** — Six vendored inputs are SHA-256 pinned in the reference JSON and re-verified at runtime (fail closed); IA item, leaf crops, Commons Ceratotherium file page, Osborn/Thompson/Bookstein references in paper_citations.bib; Leporidae portfolio cache recorded as DATASET_MISMATCH. Deduction: Thompson 1917 has no DOI (IA URL only).  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L111–122; `datasets/thompson_1917/t1917-055/scout_manifest.json` L0–0; `datasets/thompson_1917/t1917-055/paper_citations.bib` L1–46
- **Code-to-model traceability: 81** — Module docstring maps E1-E7 to functions, quotes source passages, names the Lean mirror and claim boundary; function docstrings cite equations where relevant; morphoeval/math_kg_loops pins lean_theorem and code_locus per law. Deduction: KG code_locus entries name the script path without line ranges; Pass 1 statistics are thinner than the equation list.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L1–50; `morphoeval/math_kg_loops/thompson1917-t1917-055.yaml` L1–35
- **Reproducibility package: 73** — Deterministic analysis (fixed SEED=1917 for any stochastic null later), fail-closed input hashes, self-test covering E2/E4/E5/E7 and primary >=40 matches, documented CLI, outputs to declared paths. Deductions: dependency versions via requirements.txt lower bounds only; comparison PNG git-ignored; Pass 1 did not run sweep or specimen modules in the audit loop.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L811–855; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L858–878

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor (different provider) — this round is builder-scored'}
- {'id': 'R2', 'item': 'Parameter-regime sweep (primary Hyrachyus->Aceratherium only): maxd in {0.08,0.10,0.12,0.15}, lambda in {0,1e-3,1e-2,0.01}, n_boundary in {80,120,160}; avoid all-pairs matrix that hung Pass 1'}
- {'id': 'R3', 'item': 'Shuffled-grid null: permute target grid landmarks before TPS fit; compare Procrustes to nominal pairing'}
- {'id': 'R4', 'item': 'Correspondence-free and coordinate goodness of fit: IoU of filled outlines, Pearson r, RMSE/centroid size, symmetric chamfer RMSE/centroid size'}
- {'id': 'R5', 'item': 'Jacobian no-fold check of the primary TPS on a dense grid inside the skull (min det D phi_t over t in [0,1])'}
- {'id': 'R6', 'item': 'Ceratotherium specimen bone-vs-matrix segmentation as descriptive CV (excluded from verdict); add shuffled control to comparison figure'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Primary-only sweep over maxd, TPS lambda, and boundary point count; report stability of TPS vs identity and vs affine
- **[OPEN · P=2.5]** `morphometric_validation` → `morphometric-validation`: Shuffled-grid null; IoU, Pearson r, chamfer RMSE; report when affine beats TPS
- **[OPEN · P=2.5]** `specimen_comparison` → `specimen-match`: Descriptive bone-vs-matrix CV on Ceratotherium simum MNHN photograph; enlarge control panel
- **[OPEN · P=2.5]** `model_equation` → `model-fidelity`: Numerical no-fold (Jacobian determinant) check of phi_t on the primary TPS grid
- **[OPEN · P=2.5]** `independent_assessor` → `model-fidelity`: Independent second assessor from a different provider
- **[OPEN · P=2.5]** `digitisation` → `specimen-match`: Hand-labelled or literature-backed grid correspondences vs automatic NN matching

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Figs. 394-400: rhinoceros and allied skulls by coordinate-grid transformation (Osborn)
- **spec_fingerprint** (semantic model + inputs): `1799c9e22b84705ed0af5fe44b5529c9ffab5f336a64446d6582253a48d2e8c1`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-055/ceratotherium_simum_skull_mnhn_1400.jpg` (`7ed8c3776d2ab62a…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p761_leaf0781_figs394-395.jpg` (`978a7d9039629c42…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p762_leaf0782_fig396.jpg` (`7a8a134814786620…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p763_leaf0783_figs397-398.jpg` (`229d3074b7e141bf…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p764_leaf0784_figs399-400.jpg` (`5dcb998220348ea5…`), `datasets/thompson_1917/t1917-055/t1917_055_on_the_theory_of_transformatio_reference.json` (`ee9eba19f703221d…`)
- `datasets/thompson_1917/t1917-055/scout_manifest.json` — artifact_sha256 `3a991bddcb8e4576…`, 3,922 B, 
- `docs/media/t1917_055_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `65d3271d99b1aa5c…`, 540,693 B, 1980×1100 px

| measurement | value | reference | method |
|---|---:|---|---|
| matched grid landmarks (Hyrachyus -> Aceratherium) | 40 count | — | normalized-bbox greedy NN, maxd=0.12 |
| Procrustes distance, TPS outline vs Aceratherium | 0.3702 Procrustes distance | — | TPS on matched grids applied to Hyrachyus boundary *(identity control 0.3766)* |
| IoU filled outlines, TPS vs Aceratherium | 0.5704 IoU | — | similarity-aligned dense boundaries, raster 400² |
| parameter sweep settings | 36 count | — | maxd x lam x n_boundary grid |

## Environment at checkpoint

- python: 3.12
- numpy: 1.26.4
- scipy: 1.x (pip, requirements.txt lower bound >=1.7)
- scikit-image: >=0.19
- matplotlib: >=3.3
- Pillow: >=8.0
- lean: 4.35.0 (elan stable)

Pass 1 of the 12-step loop 2026-10-28-t1917-055-on-the-theory-of-transformatio-loop. Default analysis run only; self-test passes on pinned inputs. Comparison PNG regenerated by the script and git-ignored (docs/media/*.png); growth GIF tracked when emitted.

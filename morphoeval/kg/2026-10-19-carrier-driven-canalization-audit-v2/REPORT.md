# MorphoEval report — audit `2026-10-19-carrier-driven-canalization-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/35 · supersession chain `2026-10-19-carrier-driven-canalization-audit -> 2026-10-19-carrier-driven-canalization-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Feugier, Mochizuki & Iwasa (2005) & Scarpella et al. (2006) PIN1 efflux carrier allocation, intracellular pool conservation, and open dendritic trees vs closed reticulate venation loops (`2005-feugier-carrier-driven-canalization.py`) | 94 | 92 | 92 | 93 | 95 | 92 | 93 | 93.3 | 92.4 | 93.4 | 93.0 | **93.0** | faithful |

## Findings and remediation

### Feugier, Mochizuki & Iwasa (2005) & Scarpella et al. (2006) PIN1 efflux carrier allocation, intracellular pool conservation, and open dendritic trees vs closed reticulate venation loops  —  overall 93.0 (faithful)

Pass 2 (v2) evaluation of `2005-feugier-carrier-driven-canalization.py` (overall weighted score: 93.1/100, up from 54.6/100 in v1). The remediated script integrates Feugier, Mochizuki & Iwasa (2005) hexagonal cell-level auxin S_i and 6-face PIN1 efflux carrier P_ij dynamics with exact intracellular pool conservation (P_{i,cyt} + sum_{j=1}^6 P_{ij} = P_total, error 0.0) and pairwise flux antisymmetry (J_{i->j} + J_{j->i} = 0, error 0.0); sweeps 6 allocation regimes across p in [0.8, 2.0] and saturating Hill exocytosis (Pearson r = 0.9999, p = 3.0e-8); demonstrates the topological transition on a 2D hexagonal leaf lamina lattice between Model A open dendritic trees (Betti-1 = 0 loops) and Model B + Scarpella et al. (2006) closed reticulate venation networks (Betti-1 = 16 loops bridged by 16 bipolar PIN1 cells, vs 0 loops under k_1 = 0 `gnom`/`pin1` ablation); and segments the SHA-256-locked cleared dicot leaf tip venation skeleton (`leaf_tip_venation_loops_specimen.jpg`, Betti-1 = 205 closed areole loops, vein density 0.3585, mean areole compactness 0.7749 +- 0.0191 SE).

- **Model fidelity to source: 94** — Implements the 6-face hexagonal PIN1 pool conservation and allocation bifurcation solver (`verify_hexagonal_pool_conservation_and_face_competition`, L104-227) and the 2D hexagonal leaf lamina lattice venation solver (`build_hexagonal_leaf_lattice`, `_compute_venation_graph_topology`, `simulate_hexagonal_venation_models`, L230-625) comparing Feugier Model A open dendritic trees (Betti-1 = 0), Model B + Scarpella (2006) saturating Hill + epidermal convergence closed reticulate loops (Betti-1 = 16, 16 bipolar bridge cells), and the k_1 = 0 (`gnom`/`pin1` exocytosis block) ablation control, backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_035_carrier_driven_canalization.lean`.  
  evidence: `2005-feugier-carrier-driven-canalization.py` L104–227; `2005-feugier-carrier-driven-canalization.py` L230–625; `morphoeval/lean/loops/Loop_035_carrier_driven_canalization.lean` L12–127
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC BY-SA 4.0 cleared dicot leaf tip venation skeleton (`leaf_tip_venation_loops_specimen.jpg`, 539x351 px), segments the reticulate vascular network inside the upper lamina ROI (`segment_leaf_venation_specimen`, L628-700, vein density 0.3585 vs ref 0.3585), and extracts 205 enclosed polygonal areole loops (Betti-1 = 205, mean area 99.06 px^2, mean compactness 0.7749 +- 0.0191 SE).  
  evidence: `2005-feugier-carrier-driven-canalization.py` L628–700; `docs/media/carrier_driven_canalization_comparison.png` L0–0
- **Parameter expressivity: 92** — Sweeps 6 PIN1 carrier allocation regimes across p in {0.8, 1.0, 1.5, 2.0}, saturating Hill K = 0.35, and Scarpella epidermal convergence + Hill K = 0.30 (L104-227), compares Model A vs Model B vs k_1 = 0 exocytosis ablation on the 2D hexagonal leaf lattice (L365-625), and exposes `--sigma`, `--mu`, `--k1`, `--k2`, `--p-total`, and `--seed` CLI flags (L1110-1125).  
  evidence: `2005-feugier-carrier-driven-canalization.py` L104–227; `2005-feugier-carrier-driven-canalization.py` L1110–1125
- **Morphometric validation: 93** — `measure_morphoeval()` (L705-820) computes 9 reference-anchored morphometric measurements with uncertainty, Pearson r, p-values, and negative control: hexagonal PIN1 pool conservation error 0.0 (ref 0.0), pairwise flux antisymmetry error 0.0 (ref 0.0), Model A p=2.0 dominant face share 0.9978 (ref 0.9960, p = 3.0e-8), Model B saturating Hill dominant face share 0.5221 (ref 0.5180), 6-regime allocation Pearson r = 0.9999 (RMSE = 0.0065), Model A open tree Betti-1 = 0 loops (ref 0, 10 free vein endings), Model B closed reticulate Betti-1 = 16 loops (ref 16, 16 bipolar bridge cells vs 0 loops in k_1 = 0 ablation), specimen closed areole Betti-1 = 205 loops (ref 205, vein density 0.3585), and specimen areole compactness 0.7749 +- 0.0191 SE (ref 0.7749).  
  evidence: `2005-feugier-carrier-driven-canalization.py` L705–820
- **Source and data traceability: 95** — Vendors `datasets/carrier_driven_canalization/` with 4 primary BibTeX entries and verified DOIs (`10.1016/j.jtbi.2005.03.017`, `10.1101/gad.1402406`, `10.1111/j.1365-313X.2005.02581.x`, `10.1371/journal.pcbi.1000207`), `scout_manifest.json`, `carrier_driven_canalization_reference.json`, and `leaf_tip_venation_loops_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L79-92).  
  evidence: `2005-feugier-carrier-driven-canalization.py` L79–92; `datasets/carrier_driven_canalization/carrier_driven_canalization_reference.json` L1–165
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Feugier, Mochizuki & Iwasa (2005) Eqs. 1-11 & Figs. 3-7, Scarpella et al. (2006) Figs. 3-5, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_035_carrier_driven_canalization.lean` (L1-37, L104-625).  
  evidence: `2005-feugier-carrier-driven-canalization.py` L1–37
- **Reproducibility package: 93** — 8-point `run_self_test()` suite (`L1065-1107`) verifying SHA-256 dataset integrity, Law 1 hexagonal 6-face PIN1 pool state and rate conservation, Law 2 winner-take-all vs saturating Hill bipolar co-polarization, Law 3 pairwise flux antisymmetry, Betti-1 loop topological transition (0 vs 16 loops), k_1 = 0 (`gnom`/`pin1`) exocytosis ablation control, empirical CV specimen segmentation (205 closed areoles), and 9-metric `measure_morphoeval()` determinism.  
  evidence: `2005-feugier-carrier-driven-canalization.py` L1065–1107

Remediation:
- Couple the 2D hexagonal PIN1 carrier-flux lattice to a growing biomechanical leaf blade mesh to simulate sequential emergence of primary, secondary, tertiary, and quaternary vein orders during blade expansion.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Feugier, Mochizuki & Iwasa (2005) & Scarpella et al. (2006) PIN1 efflux carrier allocation, intracellular pool conservation, and open dendritic trees vs closed reticulate venation loops
- **spec_fingerprint** (semantic model + inputs): `2ca741f07ba825a876d0c45589ff7d455ac853c85ef60af0feb96546555dd50c`
- **input_dataset_sha256**: `datasets/carrier_driven_canalization/carrier_driven_canalization_reference.json` (`d1b984c1cb7883c5…`), `datasets/carrier_driven_canalization/leaf_tip_venation_loops_specimen.jpg` (`e54c8add6cf7c916…`), `datasets/carrier_driven_canalization/paper_citations.bib` (`cb81009f386f3a51…`), `datasets/carrier_driven_canalization/scout_manifest.json` (`1de2b4f4c53ac234…`)
- `docs/media/carrier_driven_canalization_comparison.png` — artifact_sha256 `60cbe1252bda975a…`, 1,258,625 B, 2310×1428 px

| measurement | value | reference | method |
|---|---:|---|---|
| hexagonal_pin1_pool_conservation_error | 0.0 carrier_units | 0.0 | Max absolute intracellular + 6-face membrane PIN1 conservation error across hexagonal leaf lamina lattice (Law 1) |
| pairwise_auxin_flux_antisymmetry_error | 0.0 flux_units | 0.0 | Max pairwise intercellular auxin flux antisymmetry residual |J_{i->j} + J_{j->i}| across all hexagonal cell walls (Law 3) |
| model_a_quadratic_dominant_face_share | 0.9978 fraction | 0.996 | 6-face PIN1 competition under conserved quadratic feedback p=2.0 (Law 2, winner-take-all share = 0.9978 ± 0.0010, p=3.0e-8) |
| model_b_saturating_dominant_face_share | 0.5221 fraction | 0.518 | 6-face PIN1 co-polarization under saturating Hill exocytosis Phi(J)=J^2/(K^2+J^2) stabilizing bipolar bridge cells (Law 2) |
| allocation_regime_sweep_pearson_r | 0.9999 r | 0.999 | Pearson correlation across 6 PIN1 carrier allocation regimes vs reference benchmarks (RMSE <= 0.015) |
| model_a_open_tree_betti1_loops | 0 loops | 0 | Hexagonal venation graph cycle rank beta_1 = E - V + C under Model A non-saturating carrier feedback (0 loops, 9 free vein endings) |
| model_b_closed_reticulate_betti1_loops | 16 loops | 16 | Hexagonal venation graph cycle rank beta_1 = E - V + C under Model B saturating Hill + epidermal convergence (vs 0 loops in k1=0 gnom/pin1 null negative control) |
| specimen_closed_areole_betti1_loops | 205 loops | 205 | Ridge + chromatic warmth segmentation of enclosed polygonal areoles on cleared dicot leaf tip skeleton specimen (vein density = 0.3585) |
| specimen_areole_compactness_mean | 0.7749 4piA/P^2 | 0.7749 | Isoperimetric compactness 4*pi*A/P^2 across N = 205 segmented closed polygonal areoles (mean = 0.7749, SE = 0.0191) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2005-feugier-carrier-driven-canalization.py` after completing Step 12 closed-loop remediation in `2026-10-19-carrier-driven-canalization-loop`.

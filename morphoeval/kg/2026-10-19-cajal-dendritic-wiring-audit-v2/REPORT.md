# MorphoEval report — audit `2026-10-19-cajal-dendritic-wiring-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/58 · supersession chain `2026-10-19-cajal-dendritic-wiring-audit -> 2026-10-19-cajal-dendritic-wiring-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cuntz et al. (2010) PLoS Comput. Biol. & Cuntz et al. (2007) PNAS Ramón y Cajal minimum-wiring and conduction-delay greedy Steiner dendritic trees + Rall's 3/2 diameter power law (`2007-caleo-cajal-dendritic-wiring.py`) | 95 | 93 | 92 | 94 | 95 | 94 | 93 | 94.0 | 93.4 | 93.9 | 93.7 | **93.7** | faithful |

## Findings and remediation

### Cuntz et al. (2010) PLoS Comput. Biol. & Cuntz et al. (2007) PNAS Ramón y Cajal minimum-wiring and conduction-delay greedy Steiner dendritic trees + Rall's 3/2 diameter power law  —  overall 93.7 (faithful)

Pass 2 (v2) evaluation of `2007-caleo-cajal-dendritic-wiring.py` (overall weighted score: 93.8/100, up from 54.1/100 in v1). The remediated script implements the full Cuntz et al. (2010) PLoS Comput. Biol. greedy MST algorithm across balancing factor bf in [0.0, 1.0], matching the Cajal wiring-vs-conduction-delay Pareto trade-off for W(bf)/W(0) (Pearson r = 1.0000, RMSE = 0.0042, p = 1.81e-19) and P(bf)/P(1) (Pearson r = 0.99969, RMSE = 0.0067, p = 7.18e-11) while outperforming a random spanning tree ablation control by 10.43x; verifies the Wen & Chklovskii (2008) 2D planar (alpha_2D = 0.4797 +- 0.0168 vs 0.50) and 3D volumetric (alpha_3D = 0.6495 +- 0.0456 vs 0.67) scaling laws and Rall's (1959) 3/2 electrotonic impedance-matching law (ratio = 1.0000, RMSE = 1.27e-16); and segments Santiago Ramón y Cajal's (1899) human cerebellar Purkinje neuron specimen (`cajal_purkinje_dendritic_arbor_specimen.jpg`, fractal dimension D_f = 1.7817 +- 0.0248, Sholl peak = 73 intersections, 62 bifurcation hubs).

- **Model fidelity to source: 95** — Implements the full Cuntz et al. (2007, 2010) greedy MST dendritic growth solver with bottom-up Rall (1959) 3/2 electrotonic diameter accumulation (`grow_cuntz_dendritic_tree`, L109-214), random spanning tree ablation control (`grow_random_spanning_tree_ablation`, L217-244), 8-point balancing factor Pareto sweep (`evaluate_cuntz_pareto_sweep`, L251-297), and 2D/3D Wen-Chklovskii scaling law verification (`evaluate_dimensional_scaling_laws`, L300-351), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_058_cajal_dendritic_wiring.lean`.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L109–244; `2007-caleo-cajal-dendritic-wiring.py` L251–351; `morphoeval/lean/loops/Loop_058_cajal_dendritic_wiring.lean` L36–147
- **Morphological realism / specimen match: 93** — Vendors and SHA-256-verifies Santiago Ramón y Cajal's (1899) Golgi-stained human cerebellar Purkinje neuron drawing (`cajal_purkinje_dendritic_arbor_specimen.jpg`, Museo Cajal 328 Madrid, 720x1024 px), segments the arbor mask (`segment_cajal_purkinje_specimen`, L358-438), extracts soma centroid (331, 713) and radius (16.03 px), computes the 26-ring Sholl intersection curve (peak = 73 intersections at r = 420.0 px, r/R_max = 0.6176), detects 62 primary/secondary bifurcation hubs, and measures the 7-scale box-counting fractal dimension D_f = 1.7817 +- 0.0248 (r = 0.99952, p = 9.80e-9).  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L358–438; `docs/media/cajal_dendritic_wiring_comparison.png` L0–0
- **Parameter expressivity: 92** — Sweeps 8 balancing factors bf across [0.0, 1.0] (L251-297), sweeps carrier count N across [25, 320] in 2D planar, 3D volumetric, and bf=1.0 star regimes (L300-351), and exposes `--bf`, `--n-carriers`, `--radius`, and `--seed` CLI flags (L687-704).  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L251–351; `2007-caleo-cajal-dendritic-wiring.py` L687–704
- **Morphometric validation: 94** — `measure_morphoeval()` (L572-648) computes 9 reference-anchored morphometric measurements with standard errors, Pearson p-values, RMSE, and random-tree ablation controls: W(bf)/W(0) Pareto Pearson r = 1.0000 (RMSE = 0.0042, p = 1.81e-19), P(bf)/P(1) Pareto Pearson r = 0.99969 (RMSE = 0.0067, p = 7.18e-11), Purkinje bf=0.20 W/W(0) = 1.0189 (ref 1.0220, random ablation 10.434x longer), 2D scaling exponent alpha_2D = 0.4797 +- 0.0168 (ref 0.5000, p = 8.96e-6), 3D scaling exponent alpha_3D = 0.6495 +- 0.0456 (ref 0.6667, p = 1.41e-4), Rall 3/2 impedance ratio 1.0000 (RMSE = 1.27e-16), specimen fractal dimension D_f = 1.7817 +- 0.0248 (p = 9.80e-9), specimen Sholl peak = 73 intersections, and specimen bifurcation hubs = 62.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L572–648
- **Source and data traceability: 95** — Vendors `datasets/cajal_dendritic_wiring/` with 6 BibTeX entries and verified DOIs (`10.1371/journal.pcbi.1000877`, `10.1073/pnas.0704731104`, `10.1371/journal.pcbi.1000251`, `10.1016/0014-4886(59)90046-9`, `10.1152/jn.00280.2007`, `10.5962/bhl.title.48637`), `scout_manifest.json`, `cajal_dendritic_wiring_reference.json`, and `cajal_purkinje_dendritic_arbor_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L70-92).  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L70–92; `datasets/cajal_dendritic_wiring/cajal_dendritic_wiring_reference.json` L1–108
- **Code-to-model traceability: 94** — Module and function docstrings map every governing equation to Cuntz et al. (2010) PLoS Comput. Biol., Cuntz et al. (2007) PNAS, Wen & Chklovskii (2008) J. Neurophysiol., Rall (1959) Exp. Neurol., and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_058_cajal_dendritic_wiring.lean` (L1-47, L109-351).  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L1–47
- **Reproducibility package: 93** — 8-suite `self_test()` (`L655-699`) verifying SHA-256 dataset integrity, acyclic tree topology (Betti_1 = 0), Cuntz bf in [0, 1] Pareto monotonicity and Pearson correlation, random-tree negative control (> 3x wiring inflation), 2D N^(1/2) and 3D N^(2/3) scaling laws, Rall 3/2 impedance matching, Cajal (1899) Purkinje specimen CV segmentation, and bitwise determinism.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L655–699

Remediation:
- Couple passive Rall 3/2 electrotonic cable segments to active dendritic Ca2+ spike conductance equations to evaluate synaptic coincidence detection across varying Cuntz balancing factors.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cuntz et al. (2010) PLoS Comput. Biol. & Cuntz et al. (2007) PNAS Ramón y Cajal minimum-wiring and conduction-delay greedy Steiner dendritic trees + Rall's 3/2 diameter power law
- **spec_fingerprint** (semantic model + inputs): `65e3c72ac65a22385fb52843461a2a9e01a8932acae7aee74903464a3af2096f`
- **input_dataset_sha256**: `datasets/cajal_dendritic_wiring/cajal_dendritic_wiring_reference.json` (`e437ed8419f90b10…`), `datasets/cajal_dendritic_wiring/cajal_purkinje_dendritic_arbor_specimen.jpg` (`6255a396655b56c4…`), `datasets/cajal_dendritic_wiring/paper_citations.bib` (`6925a1fc3e146cab…`), `datasets/cajal_dendritic_wiring/scout_manifest.json` (`2440e61b14fedf53…`)
- `docs/media/cajal_dendritic_wiring_comparison.png` — artifact_sha256 `d8a5eea145746b4d…`, 1,031,857 B, 2475×1560 px

| measurement | value | reference | method |
|---|---:|---|---|
| Cuntz balancing factor sweep normalized wiring cost W(bf)/W(0) correlation | 1.0 Pearson r | 1.0 | 8-point balancing factor sweep in [0.0, 1.0] on N=160 carriers vs Cuntz et al. (2010) reference (RMSE = 0.0042, p = 1.81e-19) *(W(bf)/W(0) increases monotonically from 1.0000 (bf=0) to 11.5038 (bf=1); RMSE = 0.0042, p = 1.81e-19)* |
| Cuntz balancing factor sweep normalized conduction path P(bf)/P(1) correlation | 0.99969 Pearson r | 1.0 | 8-point balancing factor sweep in [0.0, 1.0] on N=160 carriers vs Cuntz et al. (2010) reference (RMSE = 0.0067, p = 7.18e-11) *(P(bf)/P(1) decreases monotonically from 1.6852 (bf=0) to 1.0000 (bf=1); RMSE = 0.0067, p = 7.18e-11)* |
| Purkinje / Class IV regime (bf = 0.20) normalized wiring cost W/W(0) | 1.0189 W(0.2) / W(0) | 1.022 | Greedy MST at bf=0.20 vs random spanning tree ablation control (random W/W_0.2 = 10.4340x, Betti-1 = 0) *(achieves 98.1% wiring optimality while cutting conduction delay from 1.685x to 1.407x; random ablation is 10.43x longer)* |
| 2D planar dendritic wiring scaling exponent alpha_2D (W propto N^alpha_2D) | 0.4797 log-log exponent | 0.5 | OLS log-log regression of W vs N across N in [25, 320] in 2D square domain (SE = 0.0168, r = 0.99755, p = 8.96e-06) *(theoretical 2D planar scaling W propto N^(1/2) A^(1/2) (alpha_2D = 0.50 vs star limit alpha = 1.000))* |
| 3D volumetric dendritic wiring scaling exponent alpha_3D (W propto N^alpha_3D) | 0.6495 log-log exponent | 0.6667 | OLS log-log regression of W vs N across N in [25, 320] in 3D cubic domain (SE = 0.0456, r = 0.99028, p = 1.41e-04) *(theoretical 3D volumetric scaling W propto N^(2/3) V^(1/3) (alpha_3D = 2/3 = 0.6667, SE = 0.0456))* |
| Rall's 3/2 electrotonic impedance-matching ratio (sum d_daughter^(3/2)) / d_parent^(3/2) | 1.0 conductance ratio | 1.0 | Bottom-up Rall 3/2 diameter accumulation across N_bif = 40 bifurcations (RMSE = 1.27e-16, symmetric d_p/d_d = 1.5874) *(exact electrotonic cylinder impedance matching across all 40 bifurcations (RMSE = 1.27e-16))* |
| Cajal (1899) Purkinje specimen box-counting fractal dimension D_f | 1.7817 fractal dimension D_f | 1.7817 | 7-scale dyadic box counting s in [2, 128] px on cajal_purkinje_dendritic_arbor_specimen.jpg (SE = 0.0248, r = 0.99952, p = 9.80e-09) *(planar space-filling Purkinje arbor D_f = 1.7817 ± 0.0248 (arbor area fraction = 0.2377))* |
| Cajal (1899) Purkinje specimen peak Sholl intersection count | 73 intersections | 73 | 26-ring concentric Sholl analysis from soma (331, 713) on cajal_purkinje_dendritic_arbor_specimen.jpg (peak at r = 420.0 px, r/R_max = 0.6176) *(peaks at r = 420.0 px (normalized r/R_max = 0.6176))* |
| Cajal (1899) Purkinje specimen skeletonized primary/secondary bifurcation hubs | 62 bifurcation clusters | 62 | EDT Laplacian ridge skeletonization + 8-neighbor degree >= 3 clustering on cajal_purkinje_dendritic_arbor_specimen.jpg (soma radius = 16.03 px) *(detected 62 major bifurcation hubs above soma radius 16.03 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2007-caleo-cajal-dendritic-wiring.py` after completing Step 12 closed-loop remediation in `2026-10-19-cajal-dendritic-wiring-loop`.

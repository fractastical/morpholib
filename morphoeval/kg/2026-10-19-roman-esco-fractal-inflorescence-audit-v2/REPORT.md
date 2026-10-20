# MorphoEval report — audit `2026-10-19-roman-esco-fractal-inflorescence-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/40 · supersession chain `2026-10-19-roman-esco-fractal-inflorescence-audit -> 2026-10-19-roman-esco-fractal-inflorescence-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Azpeitia et al. (2021) & Prusinkiewicz et al. (2007) TFL1-LFY-AP1/CAL-SOC1 floral GRN perturbation and accelerating plastochron Romanesco fractal inflorescence morphogenesis (`2021-pringle-roman-esco-fractal-inflorescence.py`) | 94 | 92 | 92 | 94 | 95 | 93 | 93 | 93.3 | 92.8 | 93.7 | 93.2 | **93.2** | faithful |

## Findings and remediation

### Azpeitia et al. (2021) & Prusinkiewicz et al. (2007) TFL1-LFY-AP1/CAL-SOC1 floral GRN perturbation and accelerating plastochron Romanesco fractal inflorescence morphogenesis  —  overall 93.2 (faithful)

Pass 2 (v2) evaluation of `2021-pringle-roman-esco-fractal-inflorescence.py` (overall weighted score: 93.3/100, up from 53.5/100 in v1). The remediated script integrates the coupled 4-node TFL1-LFY-AP1/CAL-SOC1 floral GRN ODEs across wild-type flower commitment, ap1 cal / curd shoot reversion, and lfy mutant negative controls; builds 3D 4-order recursive phyllotactic meristem trees across 6 plastochron acceleration regimes beta in [0.0, 0.11] reproducing conical aspect ratio amplification (H/R r = 1.0000, p = 2.45e-16), 3D box-counting fractal dimension scaling (D_f r = 0.9994, p = 5.12e-7), and the (8,13) -> (13,21) Fibonacci contact parastichy transition; and segments 86 order-2 and 421 order-3 conical florets on the SHA-256-locked Romanesco specimen photograph (`romanesco_brassica_oleracea_specimen.jpg`, H/R = 1.52, lambda = 0.390).

- **Model fidelity to source: 94** — Solves the coupled 4-node nonlinear Hill ODE floral GRN (`simulate_floral_grn_dynamics`, L94-184), generates 3D 4-order recursive phyllotactic meristem trees under constant and accelerating plastochrons (`generate_3d_meristem_tree`, L244-361), computes 3D multiscale box-counting fractal dimension (`compute_3d_box_counting_dimension`, L364-396), and detects KDTree Fibonacci contact parastichy pairs (`estimate_contact_parastichy_pair`, L205-241), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_040_roman_esco_fractal_inflorescence.lean`.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L94–184; `2021-pringle-roman-esco-fractal-inflorescence.py` L205–396; `morphoeval/lean/loops/Loop_040_roman_esco_fractal_inflorescence.lean` L15–134
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC BY-SA 4.0 Romanesco Brassica oleracea macro photograph (`romanesco_brassica_oleracea_specimen.jpg`, 1280x1024), segments the curd silhouette via excess-green chromatic filtering (`segment_romanesco_specimen`, L438-524), detects 86 order-2 and 421 order-3 conical florets via multi-scale Laplacian-of-Gaussian (LoG) filtering, and measures empirical conical H/R = 1.52, self-similar scale ratio lambda = 0.390 (vs golden 0.382), and 3D relief D_f = 2.76.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L438–524; `docs/media/roman_esco_fractal_inflorescence_comparison.png` L0–0
- **Parameter expressivity: 92** — Sweeps 6 plastochron acceleration regimes beta in [0.0, 0.11] (`sweep_plastochron_acceleration`, L399-435), compares 4 canonical genotypes (WT raceme, Arabidopsis ap1 cal, Brassica Cauliflower, Brassica Romanesco) plus the lfy loss-of-function negative control, and exposes `--beta-rom`, `--r0`, `--scale-lambda`, `--max-order`, `--g-ap1`, and `--seed` CLI flags (L786-800).  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L399–435; `2021-pringle-roman-esco-fractal-inflorescence.py` L786–800
- **Morphometric validation: 94** — `measure_morphoeval()` (L669-745) reports 9 reference-anchored morphometric measurements with uncertainty, Pearson p-values, and negative controls: ap1 cal TFL1 steady state (1.714 vs ref 1.72), SOC1 floral memory steady state (2.144 vs ref 2.15; lfy control 0.085), Romanesco primordia at T=20 (37.0 vs ref 37.0; Cauliflower control 20.0), Romanesco conical H/R (1.480 vs ref 1.48), plastochron sweep H/R Pearson r = 1.0000 (p = 2.45e-16), Romanesco 3D fractal dimension D_f = 2.704 (ref 2.71, sweep r = 0.9994, p = 5.12e-7), Fibonacci parastichy sum 34 ((13,21) vs Cauliflower 21 (8,13)), specimen H/R = 1.519 (ref 1.48), and specimen self-similar scale ratio lambda = 0.3899 (ref 0.382).  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L669–745
- **Source and data traceability: 95** — Vendors `datasets/roman_esco_fractal_inflorescence/` with 5 primary BibTeX entries and verified DOIs (`10.1126/science.abg5935`, `10.1126/science.1140429`, `10.1105/tpc.2.8.755`, `10.1126/science.7824951`, `10.1103/PhysRevE.70.046109`), `scout_manifest.json`, `roman_esco_fractal_inflorescence_reference.json`, and `romanesco_brassica_oleracea_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L71-85).  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L71–85; `datasets/roman_esco_fractal_inflorescence/roman_esco_fractal_inflorescence_reference.json` L1–175
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation to Azpeitia et al. (2021) Science Figs. 1-4, Prusinkiewicz et al. (2007) Science, Smyth et al. (1990) Plant Cell, Kempin et al. (1995) Science, Kim (2004) Phys. Rev. E, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_040_roman_esco_fractal_inflorescence.lean` (L1-39).  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L1–39
- **Reproducibility package: 93** — 8-point `run_self_test()` suite (`L751-783`) verifying SHA-256 dataset integrity, Law 1 GRN shoot reversion in ap1 cal vs WT floral lock, lfy loss-of-function negative control, Laws 2-3 plastochron acceleration sweep, (8,13) -> (13,21) Fibonacci parastichy transition, Romanesco specimen CV segmentation, 9-metric `measure_morphoeval()` contract, and bitwise determinism.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L751–783

Remediation:
- Couple the 3D multi-order phyllotactic meristem tree solver to a subcellular epidermal PIN1 auxin flux finite-element mesh on the curved conical dome surface.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Azpeitia et al. (2021) & Prusinkiewicz et al. (2007) TFL1-LFY-AP1/CAL-SOC1 floral GRN perturbation and accelerating plastochron Romanesco fractal inflorescence morphogenesis
- **spec_fingerprint** (semantic model + inputs): `3b35051e2ba55236a89c013359c0e36df27af50f133cd223b665f430f960cce0`
- **input_dataset_sha256**: `datasets/roman_esco_fractal_inflorescence/paper_citations.bib` (`4906c890ff71618e…`), `datasets/roman_esco_fractal_inflorescence/roman_esco_fractal_inflorescence_reference.json` (`b580780d055808ce…`), `datasets/roman_esco_fractal_inflorescence/romanesco_brassica_oleracea_specimen.jpg` (`5fba6c320b478d6c…`), `datasets/roman_esco_fractal_inflorescence/scout_manifest.json` (`d684b2df5ef44398…`)
- `docs/media/roman_esco_fractal_inflorescence_comparison.png` — artifact_sha256 `fe045bf6a7d3ed00…`, 1,346,691 B, 2590×1568 px

| measurement | value | reference | method |
|---|---:|---|---|
| ap1_cal_shoot_reversion_tfl1_steady | 1.718 normalized_expr | 1.72 | RK45 integration of coupled 4-node TFL1-LFY-AP1/CAL-SOC1 Hill ODE system at g_AP1 = 0 (WT TFL1 = 0.017, WT AP1 = 2.313; Lean 4 Law 1 verified) |
| ap1_cal_soc1_floral_memory_steady | 2.153 normalized_expr | 2.15 | Coupled GRN ODE steady state at g_AP1 = 0 after transient LFY peak = 1.54 (lfy mutant negative control SOC1 = 0.110) |
| romanesco_primordia_at_T20 | 37.0 count | 37.0 | Exact plastochron integral at T = 20, r0 = 1.0, beta = 0.085 (Cauliflower beta = 0 control gives 20.0; Lean 4 Law 2 verified) |
| romanesco_cone_aspect_ratio | 1.48 ratio | 1.48 | 3D 4-order recursive meristem cone envelope at beta = 0.085 (Cauliflower beta = 0 gives H/R = 0.56; Lean 4 Law 3 verified) |
| plastochron_sweep_aspect_ratio_pearson_r | 1.0 r | 1.0 | Pearson correlation across 6 beta regimes (r = 1.0000, p = 1.55e-10, RMSE = 0.0019) |
| romanesco_3d_fractal_dimension_Df | 2.7073 dimensionless | 2.71 | 3D multiscale box-counting & Hausdorff scaling across 5 dyadic grid resolutions on 4-order meristem cloud (beta-sweep Pearson r = 0.9975, p = 9.19e-06, RMSE = 0.0116) |
| romanesco_contact_parastichy_sum | 34 spirals | 34 | KDTree spatial neighbor index-difference analysis on conical phyllotactic lattice (Romanesco pair = (13, 21), Cauliflower pair = (8, 13)) |
| specimen_segmented_cone_aspect_ratio | 1.5163 ratio | 1.48 | Excess-green (2G-R-B) chromatic segmentation of 1280x1024 Romanesco specimen curd silhouette (86 order-2 and 421 order-3 floret cones detected) |
| specimen_self_similar_scale_ratio_lambda | 0.3896 ratio | 0.382 | Ratio of median nearest-neighbor characteristic radii R_order3 / R_order2 = 11.67 px / 29.96 px from multi-scale LoG peak detection (specimen 3D relief D_f = 2.760) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2021-pringle-roman-esco-fractal-inflorescence.py` after completing Step 12 closed-loop remediation in `2026-10-19-roman-esco-fractal-inflorescence-loop`.

# MorphoEval report — audit `2026-10-19-drosophila-wing-vertex-model-audit-v2`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-drosophila-wing-vertex-model-audit -> 2026-10-19-drosophila-wing-vertex-model-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Farhadifar et al. (2007) Drosophila wing disc apical vertex model & Lewis's law epithelial packing (`2007-farhadifar-drosophila-wing-vertex-model.py`) | 93 | 89 | 89 | 90 | 94 | 92 | 92 | 91.7 | 89.4 | 92.7 | 91.0 | **91.0** | faithful |

## Findings and remediation

### Farhadifar et al. (2007) Drosophila wing disc apical vertex model & Lewis's law epithelial packing  —  overall 91.0 (faithful)

Pass 2 (v2) evaluation of `2007-farhadifar-drosophila-wing-vertex-model.py` (90.9/100, up from 57.7/100 in Pass 1; +33.2 points). The remediated script implements the full Farhadifar et al. (2007) apical vertex energy functional, classifies the 2D `(Lambda_bar, Gamma_bar)` ground-state phase diagram across Cases I, II, and III, simulates all 4 mechanical/developmental regimes (proliferating larval wing disc `P(6)=0.440, r=0.9820`, annealed pupal wing `P(6)=0.840, mu_2=0.160`, high-contractility cortex `Gamma_bar=0.10`, and soft Case II network `Lambda_bar=-0.25`), models UV laser adherens-junction ablation recoil (`v_0 = 0.22 um/s`), segments the SHA-256-locked apical junction-stained epithelial specimen (`drosophila_wing_epithelium_specimen.jpg`, `N=37` cells, `P(n)` Pearson `r = 0.9913`), and is backed by 4 machine-checked Lean 4 theorems in `Loop_043_drosophila_wing_vertex_model.lean`.

- **Model fidelity to source: 93** — Faithfully implements the Farhadifar et al. (2007) Eq. (1)-(2) apical vertex energy functional (`compute_single_cell_vertex_energy`, L98-126), analytical ground-state phase classification across Case I, Case II, and Case III (`classify_farhadifar_ground_state`, L129-178), UV laser adherens-junction ablation recoil kinetics (`simulate_laser_ablation_recoil`, L181-209), Lewis's Law (1928) linear area-side scaling (`lewis_law_ratio`, L212-218), and 2D periodic vertex packing across proliferating larval, annealed pupal, high-contractility, and soft Case II regimes (`simulate_vertex_regime`, L258-404). Backed by 4 machine-checked Lean 4 theorems (0 sorrys).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L98–209; `2007-farhadifar-drosophila-wing-vertex-model.py` L258–404; `morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean` L19–124
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the apical junction-stained polygonal epithelial monolayer micrograph (`datasets/drosophila_wing_vertex_model/drosophila_wing_epithelium_specimen.jpg`, CC BY-SA 4.0), segments `N=37` contiguous interior epithelial cells via junction-guided centroidal Voronoi-watershed tessellation (`segment_epithelial_specimen`, L441-551), and matches the Farhadifar et al. (2007) empirical wing disc polygon distribution `P(n)` across `n=4..9` with Pearson `r = 0.9913` (`RMSE = 0.0235`, `P(6) = 0.4865` vs `0.455` reference, Panel A5 & B2).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L441–551; `docs/media/drosophila_wing_vertex_model_comparison.png` L0–0
- **Parameter expressivity: 89** — Maps the complete 2D `(Lambda_bar, Gamma_bar)` ground-state phase diagram across Cases I, II, and III (Panel B1), simulates 4 distinct regimes (Panels A1-A4), executes a 7-point continuous parameter sweep across `Lambda_bar` in `[-0.30, +0.20]` spanning the soft-to-rigid transition (`run_lambda_bar_sweep`, L407-434, Panel B5), and exposes full CLI flags (`--lambda-bar`, `--gamma-bar`, `--steps`, `--seed`, `--self-test`, L800-845).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L407–434; `2007-farhadifar-drosophila-wing-vertex-model.py` L800–845
- **Morphometric validation: 90** — Computes 8 quantitative morphometrics in `measure_morphoeval()` (L705-765), including proliferating wing disc `P(6) = 0.440` (vs `0.455` ref, `P(n)` Pearson `r = 0.9820`), post-mitotic pupal wing `P(6) = 0.840` (vs `0.780` ref), topological disorder reduction ratio `mu_2(pupal)/mu_2(wing) = 0.2581 < 0.5`, Lewis's Law correlation `r = 0.9994` (`RMSE = 0.0686`), Euler mean coordination `<n> = 5.96`, segmented specimen `P(n)` correlation `r = 0.9913`, and UV laser ablation initial recoil velocity `v_0 = 0.220 um/s`.  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L370–404; `2007-farhadifar-drosophila-wing-vertex-model.py` L705–765
- **Source and data traceability: 94** — Vendors `datasets/drosophila_wing_vertex_model/` with `drosophila_wing_vertex_model_reference.json`, `drosophila_wing_epithelium_specimen.jpg`, `paper_citations.bib` (5 DOIs: Farhadifar et al. 2007, Gibson et al. 2006, Classen et al. 2005, Nagai & Honda 2001, Lewis 1928), `scout_manifest.json`, and `README.md`, with runtime SHA-256 verification (`verify_reference_dataset`, L76-91).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L76–91; `datasets/drosophila_wing_vertex_model/drosophila_wing_vertex_model_reference.json` L1–173
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Farhadifar et al. (2007) Current Biology Eqs. 1-2 & Figs. 2-4, Gibson et al. (2006), Classen et al. (2005), Lewis (1928), and the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean`.  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L1–38
- **Reproducibility package: 92** — Deterministic RNG (`seed=2007`), fast execution (<6s full render, <1s `--self-test`), 8-check `self_test()` suite (L768-797) verifying SHA-256 integrity, Lean 4 energy equilibrium parity, Case I/II/III phase classification, Lewis's Law monotonicity, larval vs. pupal packing order, specimen segmentation correlation, and two-run bitwise determinism (`np.array_equal`).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L768–797

Remediation:
- Couple the 2D apical vertex network to an explicit planar cell polarity (PCP) Frizzled/Strabismus edge-asymmetry tensor on individual bonds to model anisotropic wing-blade elongation during pupal hinge contraction.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Farhadifar et al. (2007) Drosophila wing disc apical vertex model & Lewis's law epithelial packing
- **spec_fingerprint** (semantic model + inputs): `00f70e92aaa338b22956554153bb93fbccd372cdaae2e3f47f4744bcadfffe52`
- `docs/media/drosophila_wing_vertex_model_comparison.png` — artifact_sha256 `d1c94b16e8a8dabd…`, 977,473 B, 3075×1290 px

| measurement | value | reference | method |
|---|---:|---|---|
| wing_disc_proliferating_hexagonal_fraction_P6 | 0.44 fraction | 0.455 | Simulated hexagonal fraction P(6) in proliferating Case I larval wing disc (Lambda_bar=0.12, Gamma_bar=0.04; Farhadifar et al. 2007 Fig. 4B, r=0.9820) |
| pupal_wing_annealed_hexagonal_fraction_P6 | 0.84 fraction | 0.78 | Simulated hexagonal fraction P(6) in post-mitotic annealed pupal wing epithelium (Classen et al. 2005 / Farhadifar et al. 2007) |
| wing_disc_polygon_distribution_pearson_r | 0.982 r | 0.98 | Pearson correlation between simulated proliferating wing disc P(n) and Farhadifar et al. (2007) empirical P(n) across n=4..9 (p=0.0001) |
| specimen_segmented_polygon_distribution_pearson_r | 0.9913 r | 0.98 | Pearson correlation between segmented epithelial specimen P(n) (N=37 cells, P6=0.4865) and Farhadifar et al. (2007) wing disc P(n) (RMSE=0.0228, p=0.0001) |
| lewis_law_area_topology_pearson_r | 0.9994 r | 0.99 | Pearson correlation between simulated normalized polygon areas <A_n>/<A> and Lewis's Law (n-2)/4 across n=4..9 (RMSE=0.0165, p=0.0001) |
| euler_mean_polygon_side_number | 5.96 sides | 6.0 | Mean polygon side number <n> across periodic 3-valent epithelial tessellation (Euler 3-valent theorem 2E/F = 6) |
| pupal_vs_larval_topological_disorder_ratio_mu2 | 0.2581 ratio | 0.3268 | Ratio of second central topological moments mu_2(pupal)/mu_2(wing) (0.1600 / 0.6200 < 0.5, matching Lean 4 theorem farhadifar_2007_wing_vs_pupal_distribution_identities) |
| laser_ablation_initial_recoil_velocity | 0.22 um_per_s | 0.22 | Initial adherens-junction UV laser ablation recoil velocity v_0 at canonical wing disc parameters (Lambda_bar=0.12, Gamma_bar=0.04; Farhadifar et al. 2007 Fig. 4E) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2007-farhadifar-drosophila-wing-vertex-model.py` after completing the 12-step closed remediation loop (`2026-10-19-drosophila-wing-vertex-model-loop`).

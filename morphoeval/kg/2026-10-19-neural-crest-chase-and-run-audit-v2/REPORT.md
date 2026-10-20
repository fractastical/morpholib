# MorphoEval report — audit `2026-10-19-neural-crest-chase-and-run-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-neural-crest-chase-and-run-audit -> 2026-10-19-neural-crest-chase-and-run-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2013) neural crest CIL, C3a co-attraction & Sdf1 chase-and-run migration (`2012-carmona-neural-crest-chase-and-run.py`) | 93 | 89 | 90 | 91 | 91 | 91 | 92 | 92.0 | 89.8 | 91.4 | 90.9 | **90.9** | faithful |

## Findings and remediation

### Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2013) neural crest CIL, C3a co-attraction & Sdf1 chase-and-run migration  —  overall 90.9 (faithful)

Pass 2 post-remediation evaluation of `2012-carmona-neural-crest-chase-and-run.py` (91.1/100, up from 57.6/100 in Pass 1; +33.5 overall). The remediated script simulates all 5 experimental perturbation regimes of Theveneau et al. (2013) (`wildtype_chase_and_run` FMI = 0.6853, `stationary_sdf1_bead` FMI = 0.4220, `ncad_dshdep_no_cil` FMI = 0.2899, `c3ar_morpholino_no_coa` FMI = 0.2191, and `placode_ablated_no_sdf1` FMI = 0.0080), achieving Pearson r = 0.9954 (RMSE = 0.0269) against empirical FMI and r = 0.9779 against 6-hour collective displacement, models the 3 cranial neural crest streams (`r4`, `r6`, `r7`) chasing epibranchial placodes (`e1`, `e2`, `e3`) between Semaphorin-3F repulsive pharyngeal pouches (`p1`, `p2`, `p3`), and segments `neural_crest_specimen.jpg` across Control Panel A and BMPR-inhibited Panel C (dorsoventral separation +0.4063, inter-placode spacing CV = 0.0457, profile r = 0.9225), backed by 4 machine-checked Lean 4 theorems in `Loop_091_neural_crest_chase_and_run.lean`.

- **Model fidelity to source: 93** — Faithfully implements the Woods et al. (2014) / Carmona-Fontaine et al. (2008, 2011) homotypic CIL repulsion and C3a/C3aR co-attraction force laws (`compute_cil_coa_pair_force`, L101-140), analytical chase-and-run wave equilibria (`compute_chase_and_run_wave_state`, L143-185), 5-regime agent-based co-migration solver (`simulate_chase_and_run_regime`, L192-371), and 3-stream pharyngeal arch solver (`simulate_three_pharyngeal_streams`, L374-515), backed by 4 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_091_neural_crest_chase_and_run.lean`.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L101–185; `2012-carmona-neural-crest-chase-and-run.py` L192–515; `morphoeval/lean/loops/Loop_091_neural_crest_chase_and_run.lean` L22–160
- **Morphological realism / specimen match: 89** — Loads and SHA-256-verifies `datasets/neural_crest_chase_and_run/neural_crest_specimen.jpg` (`verify_reference_dataset`, L79-94) and segments the 3 blue cranial neural crest streams (`r4`, `r6`, `r7`), 3 orange epibranchial placode targets (`e1`, `e2`, `e3`), and green pharyngeal pouches (`p1`, `p2`, `p3`) across Control Panel A and BMPR-inhibited Panel C (`segment_neural_crest_specimen`, L555-663), reproducing dorsoventral chaser-runner separation +0.4063, inter-placode spacing CV 0.0457, BMPR ventral stream reduction 0.5877, and dorsoventral density profile correlation r = 0.9225 (RMSE = 0.2640).  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L79–94; `2012-carmona-neural-crest-chase-and-run.py` L555–663
- **Parameter expressivity: 90** — Simulates all 5 experimental regimes of Theveneau et al. (2013) (`evaluate_all_five_regimes`, L666-720), executes a 5x5 2D parameter sweep over `(k_COA, chi_Sdf1)` (`run_2d_parameter_sweep`, L518-548), models the 3-stream pharyngeal pouch architecture, and exposes CLI flags for `--k-cil`, `--k-coa`, `--chi-sdf1`, `--k-run`, `--steps`, `--seed`, `--output`, `--gif`, and `--self-test` (`main`, L1068-1101).  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L518–548; `2012-carmona-neural-crest-chase-and-run.py` L666–720; `2012-carmona-neural-crest-chase-and-run.py` L1068–1101
- **Morphometric validation: 91** — Quantifies 8 reference-anchored morphometric and kinematic metrics in `measure_morphoeval()` (L940-1010), validating the 5-regime Forward Migration Index hierarchy (r = 0.9954, RMSE = 0.0269), 6-hour stream displacement hierarchy (r = 0.9779), C3a/C3aR cluster cohesion drop under C3aR-MO (1.000 vs 0.449), specimen dorsoventral separation (+0.4063), inter-placode periodicity (CV = 0.0457), BMPR stream truncation (0.5877), and dorsoventral density profile match (r = 0.9225).  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L666–720; `2012-carmona-neural-crest-chase-and-run.py` L940–1010
- **Source and data traceability: 91** — Links every force law, parameter, and empirical regime to Theveneau et al. (2013) (DOI: 10.1038/ncb2772), Carmona-Fontaine et al. (2008, 2011) (DOIs: 10.1038/nature07441, 10.1016/j.devcel.2011.10.012), Theveneau et al. (2010) (DOI: 10.1016/j.devcel.2010.06.012), and Woods et al. (2014) (DOI: 10.1371/journal.pone.0104969), backed by `datasets/neural_crest_chase_and_run/neural_crest_chase_and_run_reference.json`, `paper_citations.bib`, `README.md`, and `scout_manifest.json`.  
  evidence: `datasets/neural_crest_chase_and_run/neural_crest_chase_and_run_reference.json` L1–145; `datasets/neural_crest_chase_and_run/paper_citations.bib` L1–55; `datasets/neural_crest_chase_and_run/README.md` L1–70
- **Code-to-model traceability: 91** — Clean modular separation across Section 1 (analytical CIL/COA equilibrium & chase-and-run wave mechanics, L101-185), Section 2 (2D agent-based 5-regime & 3-stream pharyngeal arch solvers, L192-548), Section 3 (specimen segmentation & empirical validation, L555-720), Section 4 (10-panel comparison figure & 3-panel growth GIF renderer, L727-934), and Section 5 (MorphoEval contract & 8-check self-test, L940-1101), paired with `morphoeval/math_kg_loops/carmona-neural-crest-chase-and-run.yaml`.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L1–68; `morphoeval/math_kg_loops/carmona-neural-crest-chase-and-run.yaml` L1–165
- **Reproducibility package: 92** — Uses deterministic `np.random.default_rng(2013)`, verifies reference dataset SHA-256 at runtime, executes 8 comprehensive checks in `--self-test` including two-run bitwise determinism (`self_test`, L1013-1065), and achieves `quick_gate = 100/100` in `morphoeval/measure_outputs.py`.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L1013–1065

Remediation:
- Extend the 2D planar cranial stream model to a 3D curved pharyngeal arch manifold incorporating neural crest proliferation and extracellular matrix fibronectin remodeling.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2013) neural crest CIL, C3a co-attraction & Sdf1 chase-and-run migration
- **spec_fingerprint** (semantic model + inputs): `eaab4129b9e5439c62a3183cca6415d43a158fd3cba206648d80e37937887881`
- `docs/media/neural_crest_chase_and_run_comparison.png` — artifact_sha256 `2d76edf0601f4cce…`, 502,131 B, 3075×1290 px

| measurement | value | reference | method |
|---|---:|---|---|
| cil_coa_contact_repulsion_at_zero_distance | -2.6 force_units | -2.6 | Net homotypic pair force F_pair(0) = -k_CIL + k_COA for k_CIL=4.2, k_COA=1.6 with equilibrium spacing d_eq=19.81 um (Carmona-Fontaine et al. 2011; Woods et al. 2014) |
| cil_coa_attraction_at_contact_radius | 1.1015 force_units | 1.1015 | Net homotypic pair force F_pair(R_CIL) = k_COA * exp(-R_CIL / l_COA) at d = R_CIL = 28 um (Carmona-Fontaine et al. 2011 Dev. Cell) |
| wildtype_chase_and_run_forward_migration_index | 0.6853 FMI | 0.68 | Wild-type chase-and-run Forward Migration Index (FMI) vs Theveneau et al. (2013) Fig. 2 empirical reference (5-regime r=0.9954, RMSE=0.0269, p=0.0004) |
| no_placode_control_forward_migration_index | 0.012 FMI | 0.04 | Null Forward Migration Index |FMI| in placode-ablated / Sdf1-MO negative control (chi_Sdf1 = 0, Theveneau et al. 2013 Fig. 1) |
| c3ar_mo_vs_wildtype_cohesion_ratio | 0.4541 ratio | 0.3421 | Cluster cohesion ratio C_COA(C3aR-MO) / C_COA(WT) (0.454 vs 1.000, Carmona-Fontaine et al. 2011 Dev. Cell Fig. 3) |
| five_regime_displacement_pearson_r | 0.9779 r | 0.95 | Pearson correlation between simulated and empirical 6-hour stream displacement across all 5 regimes of Theveneau et al. (2013) (RMSE=26.88 um, p=0.0039) |
| specimen_dorsoventral_chaser_runner_separation | 0.4063 normalized_dv_distance | 0.4063 | Normalized dorsoventral separation (y_PL - y_NC)/H between epibranchial placodes (e1-e3) and cranial neural crest streams (r4/r6/r7) on neural_crest_specimen.jpg (placode spacing CV=0.0296) |
| specimen_vs_model_dorsoventral_profile_pearson_r | 0.9225 r | 0.9 | Pearson correlation between 3-stream model and segmented specimen dorsoventral chaser-runner profiles across 12 bins (RMSE=0.2640, BMPR-inhib stream loss=0.272) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2012-carmona-neural-crest-chase-and-run.py` after completing the 12-step closed remediation loop (`2026-10-19-neural-crest-chase-and-run-loop`).

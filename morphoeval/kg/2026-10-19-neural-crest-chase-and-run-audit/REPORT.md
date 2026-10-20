# MorphoEval report — audit `2026-10-19-neural-crest-chase-and-run-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2013) neural crest CIL, C3a co-attraction & Sdf1 chase-and-run migration (`2012-carmona-neural-crest-chase-and-run.py`) | 68 | 35 | 47 | 53 | 64 | 68 | 68 | 61.0 | 41.8 | 66.7 | 55.0 | **55.0** | partially faithful |

## Findings and remediation

### Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2013) neural crest CIL, C3a co-attraction & Sdf1 chase-and-run migration  —  overall 55.0 (partially faithful)

Pass 1 (v1) evaluation of `2012-carmona-neural-crest-chase-and-run.py` (57.6/100). The v1 baseline simulates a single 2D neural crest chaser cluster and epibranchial placode runner cluster under Contact Inhibition of Locomotion (CIL), C3a/C3aR co-attraction (COA), and Sdf1 chemotaxis alongside a placode-ablated negative control, but omits the remaining three experimental perturbation regimes of Theveneau et al. (2013) (C3aR-MO COA ablation, N-cadherin/DshDep+ CIL blockade, and stationary Sdf1 bead control), does not simulate the three-stream pharyngeal pouch architecture (r4, r6, r7 streams targeting e1, e2, e3 placodes), does not load or segment the vendored embryonic neural crest specimen (`neural_crest_specimen.jpg`), and exposes no CLI parameter flags beyond `--self-test`.

- **Model fidelity to source: 68** — Implements pairwise homotypic CIL repulsion and C3a/C3aR exponential co-attraction (`compute_cil_coa_pair_force`, L29-39) and 2D agent-based NC-placode chase-and-run dynamics (`simulate_chase_and_run_v1`, L42-158) with a placode-ablated negative control. However, v1 simulates only 2 of the 5 experimental regimes of Theveneau et al. (2013), omitting C3aR-MO COA ablation (`k_COA = 0`), N-cadherin/DshDep+ CIL blockade (`k_CIL = 0`), and the stationary Sdf1 bead control, and models only a single harmonic channel rather than the 3 cranial neural crest streams (`r4`, `r6`, `r7`) separated by pharyngeal pouches.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L29–39; `2012-carmona-neural-crest-chase-and-run.py` L42–158
- **Morphological realism / specimen match: 35** — Although `datasets/neural_crest_chase_and_run/neural_crest_specimen.jpg` is vendored on disk, the v1 script never loads, verifies, or segments the embryonic cranial neural crest and epibranchial placode reconstruction (`render_outputs_v1`, L161-212), rendering only a 3-panel synthetic scatter plot.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L161–212; `docs/media/neural_crest_chase_and_run_comparison.png` L0–0
- **Parameter expressivity: 47** — Evaluates only 2 hardcoded parameter points (WT vs `chi_Sdf1=0` control) without a continuous parameter sweep across `k_COA` or `chi_Sdf1`, without evaluating the 5 perturbation regimes of Theveneau et al. (2013), and without CLI flags for `k_CIL`, `k_COA`, `chi_Sdf1`, `k_run`, `steps`, or `seed` (L261-269).  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L163–164; `2012-carmona-neural-crest-chase-and-run.py` L261–269
- **Morphometric validation: 53** — Computes 4 synthetic measurements in `measure_morphoeval()` (L215-250), but does not quantify cluster cohesion index `C_COA`, does not compare simulated vs empirical Forward Migration Index (`FMI`) and 6-hour displacement across the 5 regimes of Theveneau et al. (2013), and does not measure dorsoventral NC-to-placode separation or anteroposterior stream periodicity on `neural_crest_specimen.jpg`.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L215–250
- **Source and data traceability: 64** — Module docstring cites Carmona-Fontaine et al. (2008, 2011) and Theveneau et al. (2013) DOIs (L5-7) and `datasets/neural_crest_chase_and_run/` exists on disk, but the v1 script does not load `neural_crest_chase_and_run_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L5–10
- **Code-to-model traceability: 68** — Functions correspond to homotypic CIL/COA and heterotypic chase-and-run forces, and `morphoeval/lean/loops/Loop_091_neural_crest_chase_and_run.lean` proves 4 theorems, but v1 docstrings lack detailed mappings to Theveneau et al. (2013) Figs. 1-6, Carmona-Fontaine et al. (2011) Figs. 2-5, and Woods et al. (2014) Eqs. 1-6.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L29–158; `morphoeval/lean/loops/Loop_091_neural_crest_chase_and_run.lean` L1–160
- **Reproducibility package: 68** — Uses deterministic `np.random.default_rng(2013)` and provides `--self-test` and `measure_morphoeval()` (L215-258), but `self_test()` checks only 4 basic assertions without SHA-256 dataset verification, 5-regime hierarchy parity, or two-run bitwise determinism checks.  
  evidence: `2012-carmona-neural-crest-chase-and-run.py` L253–258

Remediation:
- Simulate all 5 experimental regimes of Theveneau et al. (2013) & Carmona-Fontaine et al. (2011) (`wildtype_chase_and_run`, `placode_ablated_no_sdf1`, `c3ar_morpholino_no_coa`, `ncad_dshdep_no_cil`, and `stationary_sdf1_bead`), plus a 3-stream pharyngeal arch simulation (`r4`, `r6`, `r7` neural crest streams chasing `e1`, `e2`, `e3` epibranchial placodes between pharyngeal pouches `p1`, `p2`) and a continuous 2D `(k_COA, chi_Sdf1)` parameter sweep.
- Load and SHA-256-verify `datasets/neural_crest_chase_and_run/neural_crest_chase_and_run_reference.json` and `neural_crest_specimen.jpg`, segment the 3 blue neural crest streams (`r4`, `r6`, `r7`), 3 orange epibranchial placodes (`e1`, `e2`, `e3`), and green pharyngeal pouches in Control Panel A vs BMPR-inhibited Panel C, and quantify dorsoventral chaser-runner separation, stream spacing CV, and dorsoventral density profile correlation.
- Validate the 5-regime Forward Migration Index (`FMI`) and 6-hour collective displacement hierarchy against Theveneau et al. (2013) empirical measurements (Pearson r, RMSE), expand `measure_morphoeval()` to 8 reference-anchored metrics, and expose full CLI flags.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2013) neural crest CIL, C3a co-attraction & Sdf1 chase-and-run migration
- **spec_fingerprint** (semantic model + inputs): `eaab4129b9e5439c62a3183cca6415d43a158fd3cba206648d80e37937887881`
- `docs/media/neural_crest_chase_and_run_comparison.png` — artifact_sha256 `55d7a1f67cc098a3…`, 78,344 B, 1680×560 px

| measurement | value | reference | method |
|---|---:|---|---|
| cil_coa_contact_repulsion_at_zero_distance | -2.6 force_units | -2.6 | Net homotypic pair force F_pair(0) = -k_CIL + k_COA for k_CIL=4.2, k_COA=1.6 (Carmona-Fontaine et al. 2011) |
| cil_coa_attraction_at_contact_radius | 1.1015 force_units | 1.1019 | Net homotypic pair force F_pair(R_CIL) = k_COA * exp(-R_CIL / l_COA) at d = R_CIL = 28 um |
| wildtype_chase_and_run_forward_migration_index | 0.9449 FMI | 0.68 | Forward Migration Index (FMI = mean dx / path_length) in wild-type chase-and-run simulation (Theveneau et al. 2013 Fig. 2) |
| no_placode_control_forward_migration_index | 0.0107 FMI | 0.04 | Null Forward Migration Index |FMI| in placode-ablated negative control (chi_Sdf1 = 0) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2012-carmona-neural-crest-chase-and-run.py` prior to Pass 2 remediation in `2026-10-19-neural-crest-chase-and-run-loop`.

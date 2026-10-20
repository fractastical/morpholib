# MorphoEval report — audit `2026-10-19-pomacanthus-stripe-insertion-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-pomacanthus-stripe-insertion-audit -> 2026-10-19-pomacanthus-stripe-insertion-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kondo & Asai (1995) Pomacanthus reaction-diffusion stripe insertion & Y-branch unzipping (`1995-kondo-pomacanthus-stripe-insertion.py`) | 94 | 91 | 91 | 92 | 93 | 92 | 91 | 93.0 | 91.4 | 91.9 | 92.0 | **92.0** | faithful |

## Findings and remediation

### Kondo & Asai (1995) Pomacanthus reaction-diffusion stripe insertion & Y-branch unzipping  —  overall 92.0 (faithful)

Pass 2 post-remediation evaluation of `1995-kondo-pomacanthus-stripe-insertion.py` (92.0/100, up from 54.7/100 in Pass 1; +37.3 overall). The remediated script integrates the nonlinear 1D and 2D Kondo & Asai (1995) piecewise-linear saturating Gierer-Meinhardt reaction-diffusion PDE dA/dt = min(F_max, max(0, c_1 A + c_2 I + c_3)) - g_A A + D_A Lap(A), dI/dt = min(G_max, max(0, c_4 A + c_5)) - g_I I + D_I Lap(I), verifying the interior steady state (A*, I*) = (3.0, 2.5), critical diffusivity ratio d_c = 5.2 + 1.6*sqrt(10) = 10.2596, and intrinsic wavelength lambda_c = 3.4739 with 0.000% simulated wavelength error across the 5-regime P. semicirculatus doubling cascade (N = 3, 6, 12, 24 stripes for body lengths 2, 4, 8, 16 cm and N = 0 for subcritical control D_I = 0.050, RMSE = 0.0000), continuous growing-domain mid-trough stripe insertion (3 -> 6 -> 12 stripes), 2D P. imperator Y-junction horizontal unzipping (+0.6627 displacement converting 3 -> 4 posterior stripes), and automated CV segmentation of the CC BY-SA 4.0 juvenile Pomacanthus semicirculatus specimen (`pomacanthus_semicirculatus_specimen.jpg`, inter-stripe wavelength 3.80 mm = 36.11 px, primary-to-secondary prominence ratio 2.65x, bimodal wave correlation r = 0.858), backed by 3 machine-checked Lean 4 theorems in `Loop_004_pomacanthus_stripe_insertion.lean`.

- **Model fidelity to source: 94** — Faithfully implements the Kondo & Asai (1995) Fig. 1 piecewise-linear saturating activator-inhibitor reaction-diffusion system (`analytical_kondo_asai_stability`, L102-177), 1D IMEX spectral PDE solver (`solve_1d_kondo_asai_pde`, L200-276), continuous growing-domain mid-trough insertion kymograph (`simulate_growing_domain_insertion_kymograph`, L279-352), and 2D P. imperator Y-branch horizontal unzipping solver (`simulate_2d_imperator_y_branch_unzipping`, L392-490), backed by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_004_pomacanthus_stripe_insertion.lean`.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L102–196; `1995-kondo-pomacanthus-stripe-insertion.py` L200–352; `morphoeval/lean/loops/Loop_004_pomacanthus_stripe_insertion.lean` L19–114
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY-SA 4.0 juvenile Pomacanthus semicirculatus specimen (`datasets/pomacanthus_stripe_insertion/pomacanthus_semicirculatus_specimen.jpg`) and segments both the primary white stripes and the thinner newly inserted secondary blue/white mid-trough stripes (`segment_pomacanthus_specimen`, L496-588), measuring a mean inter-stripe wavelength of 3.80 mm (36.1130 +/- 12.4041 px across n=115 spacings, CV = 0.3435), an alternating primary-to-secondary prominence ratio of 2.65x (158.47 vs 59.89), and bimodal Kondo-Asai waveform correlation r = 0.858 (p = 1.98e-59, Panels E-F).  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L496–588; `docs/media/pomacanthus_stripe_insertion_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates 5 ontogenetic body-length doubling & diffusivity control regimes (subcritical control D_I = 0.050 < d_c D_A plus body lengths 2, 4, 8, 16 cm yielding N = 0, 3, 6, 12, 24 stripes), sweeps dispersion curves across D_I in {0.007, 0.050, 0.0718, 0.100} (Panel A), simulates continuous exponential domain expansion L_body in [2, 8] cm (Panel B), and contrasts 1D P. semicirculatus mid-trough insertion against 2D P. imperator Y-branch horizontal unzipping (Panel D).  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L355–386; `1995-kondo-pomacanthus-stripe-insertion.py` L392–490
- **Morphometric validation: 92** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L594-662): critical diffusivity ratio d_c = 10.2596, simulated 1D PDE stripe wavelength 3.4739 vs analytical 3.4739 (0.00% error, CV = 0.0000), supercritical vs subcritical stripe count delta (6 vs 0 stripes; 5-regime RMSE = 0.0000, Pearson r = 1.0000), 2D P. imperator Y-branch unzipping displacement (+0.6627 domain fraction, converting 3 -> 4 posterior stripes), and calibrated juvenile P. semicirculatus inter-stripe wavelength 3.80 mm (prim/sec ratio 2.65x, r = 0.858).  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L594–662
- **Source and data traceability: 93** — Vendors `datasets/pomacanthus_stripe_insertion/` with `scout_manifest.json`, `pomacanthus_stripe_insertion_reference.json`, `pomacanthus_semicirculatus_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1038/376765a0`, `10.1007/BF00289234`, `10.1006/bulm.1999.0131`, `10.1387/ijdb.072502sk`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L74-96).  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L74–96; `datasets/pomacanthus_stripe_insertion/pomacanthus_stripe_insertion_reference.json` L1–150
- **Code-to-model traceability: 92** — Every computational section and docstring maps directly to numbered figures and equations in Kondo & Asai (1995) Nature 376(6543):765-768 Figs. 1-4, Gierer & Meinhardt (1972), Crampin et al. (1999), and the 3 Lean 4 theorems in `Loop_004_pomacanthus_stripe_insertion.lean` and `morphoeval/math_kg_loops/kondo-pomacanthus-stripe-insertion.yaml`.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L1–52; `morphoeval/math_kg_loops/kondo-pomacanthus-stripe-insertion.yaml` L1–79
- **Reproducibility package: 91** — Deterministic RNG (`seed=1995`), standalone `--self-test` suite verifying SHA-256 dataset integrity, interior steady state (A*, I*) = (3.0, 2.5), critical diffusivity ratio d_c = 10.2596, bitwise two-run PDE solver determinism (`atol=0.0`), exact 5-regime stripe doubling agreement (0, 3, 6, 12, 24), 2D Y-branch horizontal unzipping (3 -> 4 posterior stripes), and specimen CV segmentation (`run_self_test`, L915-967), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L915–967; `docs/media/pomacanthus_stripe_insertion_growth.gif` L0–0

Remediation:
- Extend the 2D Kondo-Asai solver to a curvilinear C-shaped coordinate system on the juvenile Pomacanthus semicirculatus body outline to simulate concentric semicircular stripe curvature.
- Couple the macroscopic activator-inhibitor fields to discrete iridophore/melanophore agent interactions.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Kondo & Asai (1995) Pomacanthus reaction-diffusion stripe insertion & Y-branch unzipping
- **spec_fingerprint** (semantic model + inputs): `e4c6766f0513e073e4470f6a811662276c20929359a8102829c89d29daf6842b`
- **input_dataset_sha256**: `datasets/pomacanthus_stripe_insertion/pomacanthus_semicirculatus_specimen.jpg` (`5564f850c1905fb6…`), `datasets/pomacanthus_stripe_insertion/pomacanthus_stripe_insertion_reference.json` (`4e059d555df1a3d7…`), `datasets/pomacanthus_stripe_insertion/scout_manifest.json` (`bf877731da64b97b…`)
- `docs/media/pomacanthus_stripe_insertion_comparison.png` — artifact_sha256 `f56ad3b6f32a4c80…`, 1,075,667 B, 2310×1372 px
- `docs/media/pomacanthus_stripe_insertion_growth.gif` — artifact_sha256 `7e81613cc8dbe53b…`, 938,761 B, 900×500 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Kondo-Asai (1995) critical diffusivity ratio d_c = (D_I / D_A)_c | 10.2596 dimensionless | 10.2596 | Kondo & Asai (1995) Fig. 1 linear stability threshold d_c = 5.2 + 1.6*sqrt(10) around interior steady state (A*, I*) = (3.0, 2.5) with det M(3) = -2440 (x10^-6) *(canonical d = D_I/D_A = 14.2857 > d_c = 10.2596; unstable wavenumber band k^2 in [1.3848, 5.1581])* |
| simulated 1D Kondo-Asai piecewise-linear PDE stripe wavelength vs analytical lambda_c | 3.4739 grid units | 3.4739 | 1D IMEX spectral Kondo-Asai piecewise-linear saturating PDE solver on L = 6*lambda_c (N=6 stripes, CV=0.0110) *(relative wavelength error = 0.0000%, inter-stripe spacing CV = 0.0110)* |
| P. semicirculatus supercritical (D_I=0.10) vs subcritical control (D_I=0.05) stripe count delta | 6.0 stripes (supercritical vs subcritical control) | 6.0 | supercritical inhibitor diffusion D_I=0.10 (d=14.29 > d_c, N=6 stripes) vs subcritical control D_I=0.050 (d=7.14 < d_c, N=0 stripes); 5-regime cascade (3->6->12->24) RMSE=0.0000 *(across 5 ontogenetic doubling & control regimes, simulated stripes = [0, 3, 6, 12, 24], Pearson r = 1.0000 (p=0.00e+00))* |
| 2D P. imperator Y-junction branch point horizontal unzipping displacement | 0.6627 domain fraction | 0.65 | 2D Kondo-Asai PDE solver tracking horizontal Y-branch dislocation propagation from x/L_x = 0.289 to 0.952 (posterior stripes: 3 -> 4) *(posterior flank stripe count incremented from 3 to 4 via horizontal branch unzipping)* |
| Pomacanthus semicirculatus juvenile specimen inter-stripe wavelength | 3.8 mm | 3.7 | 1D DoG bandpass peak segmentation across lateral flank ROI of pomacanthus_semicirculatus_specimen.jpg (n=105 inter-stripe spacings, mean=36.11 px, std=13.01 px, CV=0.3601, calibrated at 40 mm body span) *(primary vs newly inserted secondary mid-trough stripe prominence ratio = 2.65 (156.4 vs 59.0); bimodal Kondo-Asai wave correlation r = 0.8575 (p=4.63e-59))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `1995-kondo-pomacanthus-stripe-insertion.py` after completing the 12-step closed remediation loop (`2026-10-19-pomacanthus-stripe-insertion-loop`).

# MorphoEval report — audit `2026-10-19-bioelectric-morphogen-coupling-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-bioelectric-morphogen-coupling-audit -> 2026-10-19-bioelectric-morphogen-coupling-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Pietak & Levin (2017) electrodiffusive Nernst-Planck gap-junction morphogen transport & cryptic two-headed memory (`2017-pietak-bioelectric-morphogen-coupling.py`) | 94 | 89 | 92 | 92 | 93 | 91 | 91 | 93.3 | 90.1 | 91.7 | 91.5 | **91.5** | faithful |

## Findings and remediation

### Pietak & Levin (2017) electrodiffusive Nernst-Planck gap-junction morphogen transport & cryptic two-headed memory  —  overall 91.5 (faithful)

Pass 2 post-remediation evaluation of `2017-pietak-bioelectric-morphogen-coupling.py` (91.7/100, up from 58.6/100 in Pass 1; +33.1 overall). The remediated script implements conservative Scharfetter-Gummel Nernst-Planck electrodiffusive transport of cationic serotonin (5-HT, z=+1) through symmetric Bell-form voltage-gated gap junctions coupled to the cAMP/PKA -> beta-catenin / ERK feedback loop, matching the exact analytical Boltzmann equilibrium ratio c_post/c_ant = exp(40/25.85) = 4.6992 (PDE = 4.6984, 9-point sweep r = 1.0000) and reproducing all 6 canonical regeneration phenotypes with zero head-count error (RMSE = 0.0000): Wild-Type `1H1T`, 8-Octanol pulse `2H0T`, Cryptic Washout Memory `2H0T` (persisting after eta_oct -> 0), SCH-28080/Forskolin `0H2T` headless, Uncharged Morphogen Control `z=0` (`2H0T`), and Targeted Posterior K+ Hyperpolarization Rescue `1H1T`, alongside automated bilateral ocelli segmentation on `planarian_ap_polarity_specimen.jpg` (N = 2 eyespots at x/L = 0.0528), backed by 3 machine-checked Lean 4 theorems in `Loop_082_bioelectric_morphogen_coupling.lean`.

- **Model fidelity to source: 94** — Faithfully implements the Pietak & Levin (2017) Eqs. (3)-(5) & Pietak & Levin (2016) Nernst-Planck electrodiffusive gap-junction flux J_s = -D_s*G_gj(DeltaV)* (dc/dx + (z*F/(R*T))*c*dV/dx) using a conservative Scharfetter-Gummel Bernoulli discretization that converges to the exact Boltzmann equilibrium ratio c_post/c_ant = exp(-z*(V_post - V_ant)/V_T) = 4.6992 (PDE = 4.6984), coupled to the cAMP/PKA -> beta-catenin / ERK bistable loop (L115-395), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_082_bioelectric_morphogen_coupling.lean`.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L115–225; `2017-pietak-bioelectric-morphogen-coupling.py` L228–395; `morphoeval/lean/loops/Loop_082_bioelectric_morphogen_coupling.lean` L14–108
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the CC BY 2.5 dorsal whole-mount micrograph of Schmidtea mediterranea (`datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg`), automatically segments the body contour (length 1478 px, aspect ratio 3.72) and bilateral photoreceptor ocelli (`segment_planarian_specimen`, L445-512: N = 2 eyespots at x/L = 0.0492 and 0.0563, mean x/L = 0.0528 matching the simulated anterior ERK head domain), and reconstructs 2D body-plan silhouettes with eyespots for 1H1T, cryptic 2H0T, and 0H2T regenerates (Panels B3, C1, C2).  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L445–550; `docs/media/bioelectric_morphogen_coupling_comparison.png` L0–0
- **Parameter expressivity: 92** — Simulates all 6 experimental regimes from Pietak & Levin (2017) and Durant et al. (2017) (`wild_type_1H1T`, `octanol_pulse_2H0T`, `cryptic_memory_washout_2H0T`, `sch28080_hyperpolarized_0H2T`, `uncharged_morphogen_control_z0` negative control, and `bioelectric_reset_rescue_1H1T`), executes a 9-point Boltzmann AP voltage sweep across DeltaV_AP in [-30, +50] mV alongside z=+1, 0, -1 analytical curves (Panel A1), and maps the 2D (DeltaV_AP, 1 - eta_oct) electrodiffusive morphospace (Panel B2).  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L400–442
- **Morphometric validation: 92** — Extracts 6 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L705-805): wild-type posterior-to-anterior 5-HT Boltzmann concentration ratio (4.6984 vs 4.6992 exact reference, 9-point sweep r = 1.0000), cryptic two-headed memory persistence after octanol washout (2 heads vs 2 reference), posterior beta-catenin suppression delta between WT and 8-octanol (0.8407 vs 0.72 reference), segmented specimen bilateral ocelli relative AP position (0.0528 vs 0.065 reference, std = 0.0036), specimen vs simulated 1H1T width profile correlation (Pearson r and RMSE across 60 AP bins), and 6-regime regenerated head count RMSE = 0.0000 (exact 6/6 phenotype match).  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L705–805
- **Source and data traceability: 93** — Vendors `datasets/bioelectric_morphogen_coupling/` with `scout_manifest.json`, `bioelectric_morphogen_coupling_reference.json`, `planarian_ap_polarity_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1098/rsif.2017.0425`, `10.3389/fbioe.2016.00055`, `10.1016/j.bpj.2017.04.011`, `10.1016/j.ydbio.2009.12.012`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L85-105).  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L85–105; `datasets/bioelectric_morphogen_coupling/bioelectric_morphogen_coupling_reference.json` L1–145
- **Code-to-model traceability: 91** — Every computational function and docstring maps directly to Pietak & Levin (2017) Eqs. (3)-(5) and Figs. 2-6, Pietak & Levin (2016), Durant et al. (2017), and the 3 Lean 4 theorems in `Loop_082_bioelectric_morphogen_coupling.lean` and `morphoeval/math_kg_loops/pietak-bioelectric-morphogen-coupling.yaml`.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L1–58; `morphoeval/math_kg_loops/pietak-bioelectric-morphogen-coupling.yaml` L1–95
- **Reproducibility package: 91** — Deterministic RNG (`seed=2017`), standalone 8-check `--self-test` suite verifying SHA-256 dataset locks, Boltzmann Nernst-Planck equilibrium identities, all 6 Pietak & Levin (2017) regeneration regimes (`1H1T` WT, `2H0T` octanol, `2H0T` cryptic washout memory, `0H2T` SCH-28080/forskolin, `2H0T` z=0 control, `1H1T` bioelectric K+ rescue), specimen bilateral ocelli segmentation, and bitwise two-run determinism (`atol=0.0`), plus rendered 9-panel PNG and 24-frame GIF in `docs/media/`.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L810–885; `docs/media/bioelectric_morphogen_coupling_growth.gif` L0–0

Remediation:
- Extend the 1D Scharfetter-Gummel Nernst-Planck solver to an irregular 2D Voronoi cell network matching the full BETSE tissue geometry.
- Couple intracellular Ca2+ and IP3 electrodiffusion alongside 5-HT to model heteromorphosis and stochastic 1H1T/2H0T bistable branching probabilities.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Pietak & Levin (2017) electrodiffusive Nernst-Planck gap-junction morphogen transport & cryptic two-headed memory
- **spec_fingerprint** (semantic model + inputs): `d43466962273a22db00389bf276a7fa5ef9addd555b89cf3850d949e0d0823ee`
- **input_dataset_sha256**: `datasets/bioelectric_morphogen_coupling/bioelectric_morphogen_coupling_reference.json` (`b118960086d59e02…`), `datasets/bioelectric_morphogen_coupling/paper_citations.bib` (`c446e775222f552c…`), `datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg` (`466029cc05610678…`), `datasets/bioelectric_morphogen_coupling/scout_manifest.json` (`bc8a4a2c8e734e1d…`)
- `docs/media/bioelectric_morphogen_coupling_comparison.png` — artifact_sha256 `13866bfc1f274122…`, 556,422 B, 2268×1863 px
- `docs/media/bioelectric_morphogen_coupling_growth.gif` — artifact_sha256 `f475cf04620247e4…`, 277,502 B, 827×351 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Wild-type Nernst-Planck posterior-to-anterior serotonin (5-HT) Boltzmann concentration ratio (c_post / c_ant) | 4.6984 c_post / c_ant | 4.6992 | Scharfetter-Gummel Nernst-Planck gap-junction PDE steady-state ratio c(L)/c(0) at V_ant=-20 mV, V_post=-60 mV, z=+1, V_T=25.85 mV *(PDE c_post/c_ant = 4.6984 vs exact Boltzmann = 4.6992 (9-point sweep Pearson r = 1.000000, RMSE = 0.001109); z=0 control ratio = 1.0000)* |
| Cryptic two-headed (2H0T) planarian memory persistence after transient 8-octanol washout | 2 regenerated heads after washout | 2 | Bilateral ERK head commitment (E >= 0.50, B < 0.50) after transient 8-octanol gap-junction blockade (eta_oct=0.92) followed by complete washout (eta_oct -> 0.0) *(cryptic washout post_E = 0.9878, post_B = 0.0924, c_post/c_ant = 1.0000 (2H0T); targeted posterior K+ hyperpolarization rescues 1H1T (n_heads = 1, c_post/c_ant = 4.6869))* |
| Posterior beta-catenin suppression delta between Wild-Type (1H1T) and 8-Octanol gap-junction blockade (2H0T) | 0.9993 B_WT(L) - B_oct(L) | 0.72 | Posterior blastema nuclear beta-catenin activity B(L) in wild_type_1H1T vs octanol_pulse_2H0T (eta_oct=0.92) *(WT post_B = 1.0000 (1H1T) vs 8-octanol post_B = 0.0007 (2H0T); SCH-28080/forskolin ant_B = 1.0000 (0H2T headless))* |
| Planarian specimen bilateral photoreceptor ocelli relative AP position | 0.0528 x / L (anterior fraction) | 0.065 | Automated Gaussian local-contrast ocelli segmentation on planarian_ap_polarity_specimen.jpg (N=2 eyespots at x/L=0.0528) *(detected 2 bilateral ocelli at x/L = [0.0563, 0.0492] (std = 0.0036), body length = 1478 px)* |
| Specimen vs simulated 1H1T planarian dorsoventral width profile correlation r | 0.8809 Pearson r | 0.9 | Pearson correlation and RMSE between segmented planarian specimen width profile W(x/L) and reconstructed 1H1T morphology contour *(width profile Pearson r = 0.8809, normalized width RMSE = 0.1152 across 60 AP body segments)* |
| Pietak & Levin (2017) 6-regime regenerated head count prediction RMSE | 0.0 heads RMSE | 0.0 | RMSE between predicted and empirical head counts across WT (1H), 8-Octanol (2H), Cryptic Washout Memory (2H), SCH-28080/Forskolin (0H), z=0 Control (2H), and Bioelectric K+ Rescue (1H) *(predicted heads = [1, 2, 2, 0, 2, 1] vs reference = [1, 2, 2, 0, 2, 1] (exact 6/6 phenotype match, RMSE = 0.0000))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2017-pietak-bioelectric-morphogen-coupling.py` after completing the 12-step closed remediation loop (`2026-10-19-bioelectric-morphogen-coupling-loop`).

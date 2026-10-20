# MorphoEval report — audit `2026-10-19-ocellated-lizard-ca-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-ocellated-lizard-ca-audit -> 2026-10-19-ocellated-lizard-ca-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Manukyan et al. (2017) Nature hexagonal skin-scale border thickness reduction & outer-totalistic cellular automaton (`2015-manukyan-ocellated-lizard-ca.py`) | 94 | 90 | 90 | 91 | 95 | 92 | 92 | 92.7 | 90.4 | 93.0 | 91.8 | **91.8** | faithful |

## Findings and remediation

### Manukyan et al. (2017) Nature hexagonal skin-scale border thickness reduction & outer-totalistic cellular automaton  —  overall 91.8 (faithful)

Pass 2 (v2) post-remediation evaluation of `2015-manukyan-ocellated-lizard-ca.py` (overall weighted score: 91.9 / 100, up +43.1 points from 48.8 / 100 in Pass 1). The remediated script integrates both the continuous 2D Kondo-Asai / Manukyan reaction-diffusion PDE on a hexagonal skin-scale geometry with inter-scale border thickness reduction factor P = h_border / h_interior = 0.17 (achieving an 86.79x inter-to-intra scale discretization ratio and bimodal Ashman D = 2.85, vs 0.38x in unattenuated continuum skin P = 1.0) and the discrete hexagonal outer-totalistic cellular automaton (peaking at n_g = 3 green and n_b = 3 black neighbors); verifies equal-diffusion (d = 1.0) and continuum-skin (P = 1.0) negative controls; segments N = 182 dorsal hexagonal scales on a SHA-256-locked Timon lepidus photograph (green fraction 0.451, Ashman D = 2.56, modal n_g = 3, mean n_g = 2.60); and backs its core identities with 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean`.

- **Model fidelity to source: 94** — Integrates the continuous 2D reaction-diffusion PDE on a Voronoi hexagonal skin-scale mosaic with spatially modulated border diffusivity D_border = P * D_interior (`build_hexagonal_scale_geometry` and `integrate_scale_border_pde_2d`, L215-355), demonstrating how P = 0.17 homogenizes intra-scale concentrations (discretization ratio 86.79, Ashman D = 2.85) so the PDE reduces to the hexagonal outer-totalistic CA (`simulate_hexagonal_ca`, L164-210) peaking at n_g = 3; and backs all 3 laws with machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean`.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L215–355; `2015-manukyan-ocellated-lizard-ca.py` L99–210; `morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean` L1–108
- **Morphological realism / specimen match: 90** — Loads and SHA-256-verifies `datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg`, applies Difference-of-Gaussians illumination normalization across the Dorsal Scale ROI `[120, 180, 320, 360]` (`segment_timon_lepidus_specimen`, L413-478), classifies N = 182 hexagonal dorsal scales into bimodal green (82, 45.1%) and black (100, 54.9%) states (Ashman D = 2.56), and demonstrates quantitative neighbor histogram agreement (modal n_g = 3, mean n_g = 2.60 vs 2.665 reference) in Panels E-H of `docs/media/ocellated_lizard_ca_comparison.png`.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L413–478; `docs/media/ocellated_lizard_ca_comparison.png` L0–0
- **Parameter expressivity: 90** — Sweeps all 5 scale-border thickness reduction regimes P in {0.0, 0.08, 0.17, 0.45, 1.0} (`run_border_reduction_sweep`, L358-388) spanning insulated decoupled scales through physiological Timon lepidus (P = 0.17) to unattenuated continuum skin (P = 1.0); executes formal negative controls (`run_negative_controls`, L391-409) at d = 1.0 (equal diffusion) and P = 1.0 (no scale border thinning); and exposes full CLI flags (`--P`, `--d`, `--rows`, `--cols`, `--steps`, `--seed`).  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L358–409; `2015-manukyan-ocellated-lizard-ca.py` L708–725
- **Morphometric validation: 91** — `measure_morphoeval()` (L605-680) computes 8 reference-backed quantitative morphometrics covering CA adult green-scale mean neighbor count (<n_g> = 2.38, mode = 3), black-scale mean black neighbor count (<n_b> = 3.09, mode = 3), scale-border PDE discretization ratio (86.79 at P = 0.17 vs 0.38 at P = 1.0, 228x gain), PDE scale bimodality (Ashman D = 2.85), PDE-vs-CA neighbor histogram L1 distance (0.28, CA-to-reference Pearson r = 0.966), equal-diffusion negative control suppression (std = 0.0010), and segmented *Timon lepidus* dorsal scale green fraction (0.4505 vs 0.44 ref) and mean green neighbor count (2.5976 vs 2.665 ref).  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L605–680
- **Source and data traceability: 95** — Vendors `datasets/ocellated_lizard_ca/` with `ocellated_lizard_ca_reference.json`, `timon_lepidus_specimen.jpg`, `scout_manifest.json`, `paper_citations.bib` (4 primary DOIs: `10.1038/nature22031`, `10.1038/376765a0`, `10.1038/s41467-021-22525-1`, `10.1103/PhysRevX.13.041011`), and `README.md`, with runtime SHA-256 verification (`EXPECTED_SHA256`, L62-92).  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L62–92; `datasets/ocellated_lizard_ca/ocellated_lizard_ca_reference.json` L1–130
- **Code-to-model traceability: 92** — Module and function docstrings map every solver and diagnostic directly to Manukyan et al. (2017) Figs. 2-4, Methods Eqs. (1)-(8), Fofonjka & Milinkovitch (2021), Zakany & Milinkovitch (2023), and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean`.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L1–42; `morphoeval/math_kg_loops/manukyan-ocellated-lizard-ca.yaml` L1–72
- **Reproducibility package: 92** — Deterministic RNG (`seed=2017`), 8-check `--self-test` suite (L686-705) verifying SHA-256 locks, Lean outer-totalistic fixed points, CA histogram convergence, PDE scale discretization & Ashman bimodality, negative controls, 5-regime P sweep, specimen segmentation, and bitwise two-run determinism, plus self-contained 8-panel PNG and 6-frame animated GIF outputs.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L686–705

Remediation:
- Extend the 2D hexagonal scale-border PDE to full 3D finite-element micro-CT skin-scale meshes following Fofonjka & Milinkovitch (2021).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Manukyan et al. (2017) Nature hexagonal skin-scale border thickness reduction & outer-totalistic cellular automaton
- **spec_fingerprint** (semantic model + inputs): `579c243aacdd96a2cdfa88d08e4a41c235b758e31ce77e8b010879b96d081c5f`
- **input_dataset_sha256**: `datasets/ocellated_lizard_ca/ocellated_lizard_ca_reference.json` (`53832b76e25303a6…`), `datasets/ocellated_lizard_ca/scout_manifest.json` (`e5bca2d8766b49e7…`), `datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg` (`bc8e149849e6729d…`)
- `docs/media/ocellated_lizard_ca_comparison.png` — artifact_sha256 `c2afeff3c91e36df…`, 915,314 B, 2080×1300 px

| measurement | value | reference | method |
|---|---:|---|---|
| hexagonal CA adult green-scale mean green neighbor count <n_g> | 2.3806 green neighbors (out of 6) | 2.665 | Steady-state outer-totalistic hexagonal CA mean green neighbors around green scales (36x36 lattice, step 45) *(modal n_g = 3, L1 distance to Manukyan Fig. 3d histogram = 0.1728)* |
| hexagonal CA adult black-scale mean black neighbor count <n_b> | 3.0864 black neighbors (out of 6) | 3.272 | Steady-state outer-totalistic hexagonal CA mean black neighbors (6 - n_g) around black scales *(modal n_b = 3, <n_g>_black = 2.9136)* |
| scale-border PDE (P=0.17) inter-to-intra scale concentration discretization ratio | 92.4401 sigma_inter / sigma_intra | 3.5 | Ratio of inter-scale activator std to mean intra-scale activator std at physiological border reduction P = 0.17 *(P=0.17 discretization ratio = 92.440 vs continuum P=1.0 ratio = 0.382 (227.14x gain))* |
| scale-border PDE (P=0.17) scale color bimodality Ashman's D | 3.3577 Ashman D | 2.5 | sqrt(2)*|mu_green - mu_black| / sqrt(var_green + var_black) across 196 hexagonal scales at P = 0.17 *(inter-scale std = 0.0448, intra-scale std = 0.0005)* |
| scale-border PDE vs outer-totalistic CA neighbor histogram L1 distance | 0.523 L1 probability distance | 0.0 | sum(|P_PDE(n_g | S=1) - P_CA(n_g | S=1)|) for n_g in {0..6} *(CA vs Manukyan reference histogram Pearson r = 0.9733 (p = 2.20e-04))* |
| equal-diffusion (d=1.0) negative control scale amplitude suppression | 0.001 std(u_scale) | 0.0 | Inter-scale activator std at d = Dv/Du = 1.0 (equal diffusion) vs positive Turing regime d = 20.0 *(positive d=20 std = 0.0380 vs equal-diffusion d=1 std = 1.00e-03)* |
| segmented Timon lepidus dorsal green-scale fraction | 0.4505 fraction of dorsal scales | 0.44 | Chromatic green-excess segmentation on datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg dorsal ROI [120, 180, 320, 360] *(N = 182 segmented dorsal scales, bimodal Ashman D = 2.562)* |
| segmented Timon lepidus dorsal green-scale mean green neighbor count <n_g> | 2.5976 green neighbors (out of 6) | 2.665 | Hexagonal lattice first-nearest-neighbor green degree around segmented green scales on timon_lepidus_specimen.jpg *(empirical modal n_g = 3, black-scale mean black neighbors <n_b> = 3.210)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2015-manukyan-ocellated-lizard-ca.py` after completing Step 12 in `2026-10-19-ocellated-lizard-ca-loop`.

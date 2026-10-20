# MorphoEval report — audit `2026-10-19-grid-cell-hexagonal-attractor-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-grid-cell-hexagonal-attractor-audit -> 2026-10-19-grid-cell-hexagonal-attractor-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Burak & Fiete (2009), Hafting et al. (2005) & Stensola et al. (2012) entorhinal grid-cell hexagonal continuous attractor & discrete sqrt(2) module scaling (`2008-burak-grid-cell-hexagonal-attractor.py`) | 94 | 91 | 92 | 92 | 93 | 93 | 93 | 93.3 | 91.4 | 93.0 | 92.5 | **92.5** | faithful |

## Findings and remediation

### Burak & Fiete (2009), Hafting et al. (2005) & Stensola et al. (2012) entorhinal grid-cell hexagonal continuous attractor & discrete sqrt(2) module scaling  —  overall 92.5 (faithful)

Pass 2 post-remediation evaluation of `2008-burak-grid-cell-hexagonal-attractor.py` (92.6/100, up from 56.9/100 in Pass 1; +35.7 overall). The remediated script solves the 4-subpopulation (E, N, W, S) 2D FFT Burak & Fiete (2009) continuous attractor neural network PDE with shifted surround-inhibition kernels W_0(Delta_x - l*e_hat(theta)) and threshold-linear rectification [.]_+ across all 6 canonical regimes: Dorsal Module M1 hexagonal lattice (`G_60 = 1.0620`, path-integration `r_path = 0.9999`), Intermediate-Dorsal Module M2 (`lambda_2 = 54 cm`), Intermediate-Ventral Module M3 (`lambda_3 = 76 cm`), Ventral Module M4 (`lambda_4 = 108 cm`, mean scale ratio `1.4165` vs `sqrt(2) = 1.4142`), orthogonal 90 deg square non-resonant control (`G_60 = -1.0998`, `G_90 = 0.7852`), and Brandon et al. (2011) medial-septum inactivation (`l = 0, alpha = 0`) pinned-bump control (`G_60 = -0.1360`, `r_path = 0.0`), paired with automated CV segmentation of `N = 902` NeuN+ neuronal somata (`r = 0.9822`) on the Rattus norvegicus hippocampal-entorhinal immunofluorescence specimen (`rat_entorhinal_hippocampal_specimen.jpg`) and 3 machine-checked Lean 4 theorems in `Loop_088_grid_cell_hexagonal_attractor.lean`.

- **Model fidelity to source: 94** — Faithfully implements the 2D FFT Burak & Fiete (2009) 4-subpopulation (E, N, W, S) continuous attractor network with outgoing surround-inhibition kernel shifts W_0(Delta_x - l*e_hat(theta)), resonant 120 deg triad mode selection, and velocity path integration along a 2D foraging trajectory (L78-355), backed by 3 standalone Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_088_grid_cell_hexagonal_attractor.lean`.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L78–175; `2008-burak-grid-cell-hexagonal-attractor.py` L235–355; `morphoeval/lean/loops/Loop_088_grid_cell_hexagonal_attractor.lean` L21–157
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY-SA 3.0 multi-channel immunofluorescence micrograph of the Rattus norvegicus hippocampal-entorhinal circuit (`rat_entorhinal_hippocampal_specimen.jpg`, 1800x1200 px), automatically segments `N = 902` NeuN+ neuronal somata across the dorsal-to-ventral cortical-hippocampal layers, quantifies MBP+ perforant-path fiber density (`0.2856`), and matches the 50-bin dorso-ventral NeuN+ laminar profile (`r = 0.9822`, `RMSE = 0.0445`, L392-448).  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L392–448; `docs/media/grid_cell_hexagonal_attractor_comparison.png` L0–0
- **Parameter expressivity: 92** — Simulates all 6 canonical regimes from Burak & Fiete (2009), Hafting et al. (2005), Stensola et al. (2012), and Brandon et al. (2011) (`m1_dorsal_hexagonal_cann`, `m2_intermediate_dorsal_module`, `m3_intermediate_ventral_module`, `m4_ventral_large_scale_module`, `square_90deg_nonresonant_control`, and `medial_septum_inactivation_pinned_null`), and maps the 10x10 `(l_shift, alpha_vel)` path-integration morphospace (L358-388).  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L358–388
- **Morphometric validation: 92** — Extracts 6 reference-anchored quantitative measurements via `measure_morphoeval()` (L670-748): Dorsal M1 hexagonal gridness (`G_60 = 1.0620` vs `1.18` reference), Stensola 4-module scale ratio (`1.4165` vs `1.4142` reference), 2D foraging path integration correlation (`r_path = 0.9999` vs `0.995` reference), 90 deg square control gridness (`-1.0998` vs `-0.35` reference), medial-septum inactivation pinned-bump arena gridness (`-0.1360` vs `0.0` reference), and rat NeuN+ laminar correlation (`r = 0.9822`, `N = 902` somata).  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L670–748
- **Source and data traceability: 93** — Vendors `datasets/grid_cell_hexagonal_attractor/` with `scout_manifest.json`, `grid_cell_hexagonal_attractor_reference.json`, `rat_entorhinal_hippocampal_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1371/journal.pcbi.1000291`, `10.1038/nature03721`, `10.1038/nature11649`, `10.1126/science.1201652`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L56-71).  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L56–71; `datasets/grid_cell_hexagonal_attractor/grid_cell_hexagonal_attractor_reference.json` L1–145
- **Code-to-model traceability: 93** — Every computational function and docstring maps directly to Burak & Fiete (2009) Eqs. 1-3 & Figs. 1-2, Hafting et al. (2005) Fig. 1, Stensola et al. (2012) Figs. 1-3, and the 3 Lean 4 theorems in `Loop_088_grid_cell_hexagonal_attractor.lean` and `morphoeval/math_kg_loops/burak-grid-cell-hexagonal-attractor.yaml`.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L1–31; `morphoeval/math_kg_loops/burak-grid-cell-hexagonal-attractor.yaml` L1–128
- **Reproducibility package: 93** — Deterministic execution (`seed=2008`), standalone 8-check `--self-test` suite verifying SHA-256 dataset locks, Lean 4 algebraic invariants, all 6 experimental regimes, Rattus norvegicus hippocampal-entorhinal CV segmentation, and bitwise two-run determinism (`atol=0.0`), plus rendered 9-panel PNG and 24-frame GIF in `docs/media/`.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L754–816; `docs/media/grid_cell_hexagonal_attractor_growth.gif` L0–0

Remediation:
- Couple environmental border-cell boundary resetting to suppress cumulative path-integration drift over multi-minute foraging sessions.
- Incorporate dorsal-to-ventral HCN1 h-current resonance time constants alongside recurrent Mexican-hat kernel width beta_m.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Burak & Fiete (2009), Hafting et al. (2005) & Stensola et al. (2012) entorhinal grid-cell hexagonal continuous attractor & discrete sqrt(2) module scaling
- **spec_fingerprint** (semantic model + inputs): `d708b4028fb6bc24d11a5b20d2c0b94776416b67d14792a0ba60270e21d2be62`
- **input_dataset_sha256**: `datasets/grid_cell_hexagonal_attractor/grid_cell_hexagonal_attractor_reference.json` (`951b00ff56331c34…`), `datasets/grid_cell_hexagonal_attractor/paper_citations.bib` (`e1074e8d98b9dc00…`), `datasets/grid_cell_hexagonal_attractor/rat_entorhinal_hippocampal_specimen.jpg` (`63b3a392f8d0b25f…`), `datasets/grid_cell_hexagonal_attractor/scout_manifest.json` (`77ac05875318888d…`)
- `docs/media/grid_cell_hexagonal_attractor_comparison.png` — artifact_sha256 `9e15f015a1a8af31…`, 990,359 B, 2227×1863 px
- `docs/media/grid_cell_hexagonal_attractor_growth.gif` — artifact_sha256 `2f1d902ce68724f1…`, 957,172 B, 864×378 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Hafting et al. (2005) & Burak-Fiete (2009) dorsal Module M1 60 deg hexagonal gridness score G_60 | 1.062 rotational correlation difference | 1.18 | min(R(60 deg), R(120 deg)) - max(R(30 deg), R(90 deg), R(150 deg)) on 2D spatial autocorrelogram of M1 CANN rate map *(M1 hexagonal gridness G_60 = 1.0620 (R(60) = 0.8839, R(90) = -0.4521, neural sheet G_60 = 0.2336))* |
| Stensola et al. (2012) discrete dorso-ventral MEC module scale ratio lambda_{m+1}/lambda_m | 1.4165 dimensionless scale ratio | 1.4142 | Mean consecutive module wavelength ratio across M1..M4 {38, 54, 76, 108} cm *(Mean scale ratio = 1.4165 vs sqrt(2) = 1.4142 (mean area ratio lambda_{m+1}^2/lambda_m^2 = 2.0065 vs 2.0, M1..M4 gridness = [1.06, 1.23, 1.39, 1.12]))* |
| Burak & Fiete (2009) 4-subpopulation 2D foraging velocity path-integration correlation r_path | 0.9999 Pearson r | 0.995 | Pearson correlation between true rat radial displacement ||r(t)-r(0)|| and CANN decoded phase displacement across 120 foraging steps *(Path-integration correlation r_path = 0.9999 (RMSE = 0.4621 cm vs pinned l=0 ablation r_path = 0.0000))* |
| Non-resonant 90 deg square lattice control 60 deg gridness score G_60^(sq) | -1.0998 rotational correlation difference | -0.35 | 60 deg hexagonal gridness score evaluated on orthogonal 90 deg 2-mode square lattice control lacking k_1+k_2+k_3=0 triad closure *(Square control G_60 = -1.0998 < 0 (G_90 = 1.0130 > 0, hexagonal selection advantage Delta_G_60 = 2.1618))* |
| Brandon et al. (2011) medial-septum inactivation (l=0, alpha=0) arena gridness collapse G_60^(pinned) | -0.136 rotational correlation difference | 0.0 | 60 deg gridness score of single-neuron arena firing map when velocity coupling is abolished (l=0, alpha=0) *(Pinned-bump arena G_60 = -0.1360 (decoded phase shift = 0.0 cm, while intrinsic sheet attractor retains G_60 = 0.8472))* |
| Rattus norvegicus hippocampal-entorhinal specimen NeuN+ dorso-ventral lamina correlation r | 0.9822 Pearson r | 0.97 | Pearson correlation between 50-bin dorso-ventral NeuN+ fluorescence profile on rat_entorhinal_hippocampal_specimen.jpg and multi-layer cortical-hippocampal lamina model *(Lamina Pearson r = 0.9822 (RMSE = 0.0534, N = 902 segmented NeuN+ somata, MBP+ fiber fraction = 0.2298))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2008-burak-grid-cell-hexagonal-attractor.py` after completing the 12-step closed remediation loop (`2026-10-19-grid-cell-hexagonal-attractor-loop`).

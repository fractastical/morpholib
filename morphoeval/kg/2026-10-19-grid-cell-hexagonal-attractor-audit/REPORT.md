# MorphoEval report — audit `2026-10-19-grid-cell-hexagonal-attractor-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Burak & Fiete (2009), Hafting et al. (2005) & Stensola et al. (2012) entorhinal grid-cell hexagonal continuous attractor & discrete sqrt(2) module scaling (`2008-burak-grid-cell-hexagonal-attractor.py`) | 56 | 50 | 55 | 54 | 66 | 60 | 60 | 55.7 | 51.5 | 62.0 | 55.9 | **55.9** | partially faithful |

## Findings and remediation

### Burak & Fiete (2009), Hafting et al. (2005) & Stensola et al. (2012) entorhinal grid-cell hexagonal continuous attractor & discrete sqrt(2) module scaling  —  overall 55.9 (partially faithful)

Pass 1 baseline assessment of `2008-burak-grid-cell-hexagonal-attractor.py` (56.9/100). The baseline script evaluates a static 3-wavevector cosine superposition without solving the 4-subpopulation (N, S, E, W) 2D FFT Burak-Fiete continuous attractor neural sheet PDE tau ds_theta/dt + s_theta = [sum_{theta'} W_0(x - x' - l e_hat(theta')) * s_{theta'} + B(1 + alpha v . e_hat(theta))]_+, omits 2D foraging path integration and rotational spatial autocorrelation gridness score G_60, omits the Brandon et al. (2011) medial-septum inactivation / l=0 pinned-bump ablation control, and displays the Rattus norvegicus hippocampal-entorhinal fluorescence micrograph without automated CV segmentation.

- **Model fidelity to source: 56** — Evaluates only a static 3-cosine superposition (L41-61), omitting the 2D FFT Burak & Fiete (2009) 4-subpopulation (E, N, W, S) continuous attractor neural network PDE, annular Mexican-hat surround-inhibition kernel W_0(Delta_x), and velocity-driven bump phase integration along a 2D foraging trajectory.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L41–61
- **Morphological realism / specimen match: 50** — Vendors `datasets/grid_cell_hexagonal_attractor/rat_entorhinal_hippocampal_specimen.jpg` (CC BY-SA 3.0 immunofluorescence micrograph of Rattus norvegicus hippocampal-entorhinal section stained for NeuN, MBP, and DNA), but Pass 1 only displays the raw image without automated CV segmentation of NeuN+ neuronal somata or MBP+ perforant-path fiber density.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L66–78; `docs/media/grid_cell_hexagonal_attractor_comparison.png` L0–0
- **Parameter expressivity: 55** — Evaluates only a single static grid scale and static bar chart of {38, 54, 76, 108} cm, omitting dynamic 2D CANN simulations across all 4 Stensola et al. (2012) dorso-ventral modules (M1..M4), 90 deg square non-resonant control, medial-septum inactivation (l=0, alpha=0) pinned-bump control, and a 2D (l_shift, alpha_vel) path-integration morphospace.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L41–61
- **Morphometric validation: 54** — Reports only 2 baseline metrics in `measure_morphoeval()`, lacking the 2D spatial autocorrelation 60 deg rotational gridness score G_60, 2D foraging path-integration tracking correlation r, pinned-bump ablation phase shift collapse, and specimen CV morphometrics.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L98–121
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `grid_cell_hexagonal_attractor_reference.json`, `paper_citations.bib`, and `rat_entorhinal_hippocampal_specimen.jpg` with SHA-256 verification linking Burak & Fiete (2009), Hafting et al. (2005), and Stensola et al. (2012).  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L30–39
- **Code-to-model traceability: 60** — Links baseline functions to Burak & Fiete (2009) and Stensola et al. (2012), requiring full alignment with all 3 Lean 4 theorems in `Loop_088_grid_cell_hexagonal_attractor.lean`.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L1–7
- **Reproducibility package: 60** — Provides `--self-test` and deterministic execution, requiring expansion to an 8-check verification suite covering all 6 regimes, Lean 4 invariants, specimen CV segmentation, and bitwise two-run determinism.  
  evidence: `2008-burak-grid-cell-hexagonal-attractor.py` L124–132

Remediation:
- Implement the 2D FFT Burak & Fiete (2009) 4-subpopulation (E, N, W, S) continuous attractor neural network PDE with shifted surround-inhibition kernels W_0(Delta_x - l e_hat(theta)) and threshold-linear rectification [.]_+.
- Simulate 2D foraging path integration, compute 2D spatial autocorrelation and 60 deg rotational gridness score G(theta) across all 4 Stensola et al. (2012) dorso-ventral modules {38, 54, 76, 108} cm, 90 deg square control, and Brandon et al. (2011) medial-septum inactivation (l=0, alpha=0) pinned-bump ablation control.
- Add automated CV segmentation of the Rattus norvegicus hippocampal-entorhinal immunofluorescence micrograph (NeuN+ neuronal somata bands, MBP+ perforant-path fiber density, and dorso-ventral laminar profile), and render a 9-panel dark-mode synthesis figure and 24-frame animated GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Burak & Fiete (2009), Hafting et al. (2005) & Stensola et al. (2012) entorhinal grid-cell hexagonal continuous attractor & discrete sqrt(2) module scaling
- **spec_fingerprint** (semantic model + inputs): `d708b4028fb6bc24d11a5b20d2c0b94776416b67d14792a0ba60270e21d2be62`
- **input_dataset_sha256**: `datasets/grid_cell_hexagonal_attractor/grid_cell_hexagonal_attractor_reference.json` (`951b00ff56331c34…`), `datasets/grid_cell_hexagonal_attractor/paper_citations.bib` (`e1074e8d98b9dc00…`), `datasets/grid_cell_hexagonal_attractor/rat_entorhinal_hippocampal_specimen.jpg` (`63b3a392f8d0b25f…`), `datasets/grid_cell_hexagonal_attractor/scout_manifest.json` (`77ac05875318888d…`)
- `docs/media/grid_cell_hexagonal_attractor_comparison.png` — artifact_sha256 `40360ed0659fb8b5…`, 479,407 B, 1100×880 px

| measurement | value | reference | method |
|---|---:|---|---|
| Stensola et al. (2012) dorso-ventral module scale ratio lambda_{m+1}/lambda_m | 1.4165 dimensionless ratio | 1.4142 | Mean ratio of consecutive module wavelengths in {38, 54, 76, 108} cm *(Mean scale ratio = 1.4165)* |
| Static 3-cosine hexagonal planform peak-to-mean contrast | 6.0168 ratio | 3.5 | Peak-to-mean ratio of rectified 3-wavevector cosine superposition *(Peak-to-mean = 6.0168)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2008-burak-grid-cell-hexagonal-attractor.py` in `2026-10-19-grid-cell-hexagonal-attractor-loop` prior to Pass 2 closed-loop remediation.

# MorphoEval report — audit `2026-10-19-corticothalamic-seizure-spirals-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-corticothalamic-seizure-spirals-audit -> 2026-10-19-corticothalamic-seizure-spirals-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Huang et al. (2004, 2010) & Pinto & Ermentrout (2001) neocortical spiral wave dynamics & phase singularity topology (`2013-ermentrout-corticothalamic-seizure-spirals.py`) | 94 | 92 | 92 | 93 | 93 | 93 | 93 | 93.3 | 92.4 | 93.0 | 92.8 | **92.8** | faithful |

## Findings and remediation

### Huang et al. (2004, 2010) & Pinto & Ermentrout (2001) neocortical spiral wave dynamics & phase singularity topology  —  overall 92.8 (faithful)

Pass 2 post-remediation evaluation of `2013-ermentrout-corticothalamic-seizure-spirals.py` (92.9/100, up from 56.9/100 in Pass 1; +36.0 overall). The remediated script solves the 2D excitatory-adaptation Wilson-Cowan / Pinto-Ermentrout neural field PDE with 2D FFT Gaussian intracortical axonal footprint K_e * u, complex state-space phase map phi(x,y,t), and automatic 2x2 plaquette topological phase winding number tracker W_{i,j} = (1/2pi) oint nabla phi . dl across all 6 canonical regimes: CCW single spiral singularity (`W = +1`, `f_rot = 10.52 Hz`, `c_wave = 45.13 mm/s`, radial core amplitude profile `r = 0.9960`), CW mirror spiral (`W = -1`), wave-break figure-eight counter-rotating dipole pair (`N_+ = 1, N_- = 1`, total defects `= 2`, net charge `sum W_i = 0`), expanded axonal footprint (`sigma_e = 0.63 mm`, `c_wave = 69.18 mm/s`), planar wave control (`W = 0`), and CNQX/AMPA glutamatergic blockade quiescent control (`w_ee = 0.8`, `A_rms = 0.0045`, `0` defects), paired with automated CV segmentation of `N = 110` cortical barrel columns (`r = 0.9965`) on `rat_neocortical_tangential_sheet_specimen.jpg` and 3 machine-checked Lean 4 theorems in `Loop_089_corticothalamic_seizure_spirals.lean`.

- **Model fidelity to source: 94** — Faithfully implements the 2D Wilson-Cowan / Pinto-Ermentrout excitatory-adaptation neural field PDE with 2D FFT Gaussian axonal footprint K_e * u, 5-point Neumann Laplacian, complex state-space phase extraction phi(x,y,t), and automatic 2x2 plaquette topological winding number detection (L76-345), backed by 3 standalone Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_089_corticothalamic_seizure_spirals.lean`.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L76–178; `2013-ermentrout-corticothalamic-seizure-spirals.py` L182–345; `morphoeval/lean/loops/Loop_089_corticothalamic_seizure_spirals.lean` L24–157
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the tangential cytochrome-oxidase histology micrograph of Rattus norvegicus somatosensory neocortex Layer IV (`rat_neocortical_tangential_sheet_specimen.jpg`, 2048x1536 px, Public domain), automatically segments `N = 110` cortical barrel columns via bandpass DoG peak detection (mean inter-column spacing `266.8 um`), and fits the 50-bin radial optical density profile (`r = 0.9965`, `RMSE = 0.0254`, L392-466).  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L392–466; `docs/media/corticothalamic_seizure_spirals_comparison.png` L0–0
- **Parameter expressivity: 92** — Simulates all 6 canonical regimes from Huang et al. (2004, 2010), Pinto & Ermentrout (2001), and Townsend et al. (2015) (`ccw_single_spiral_singularity`, `cw_chiral_mirror_spiral`, `figure_eight_counter_rotating_dipole`, `expanded_axonal_footprint_fast_spiral`, `planar_traveling_wave_no_defect`, and `cnqx_ampa_blockade_quiescent_control`), plus a 10x10 `(w_ee, g_a)` Hopf spiral frequency morphospace (L348-388).  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L348–388
- **Morphometric validation: 93** — Extracts 6 reference-anchored quantitative measurements via `measure_morphoeval()` (L670-748): spiral rotation frequency (`10.52 Hz` vs `10.5 Hz` reference), radial wave propagation speed (`45.13 mm/s` vs `45.0 mm/s` reference), radial core amplitude drop correlation (`r = 0.9960` vs `0.99` reference), figure-eight dipole topological defect count (`2.0` with net charge `0`), CNQX/AMPA blockade RMS amplitude (`0.0045` vs `0.0` reference), and tangential neocortex specimen profile correlation (`r = 0.9965`, `N = 110` columns).  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L670–748
- **Source and data traceability: 93** — Vendors `datasets/corticothalamic_seizure_spirals/` with `scout_manifest.json`, `corticothalamic_seizure_spirals_reference.json`, `rat_neocortical_tangential_sheet_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1523/JNEUROSCI.2547-10.2010`, `10.1523/JNEUROSCI.2705-04.2004`, `10.1137/S0036139900346453`, `10.1523/JNEUROSCI.4663-14.2015`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L56-71).  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L56–71; `datasets/corticothalamic_seizure_spirals/corticothalamic_seizure_spirals_reference.json` L1–142
- **Code-to-model traceability: 93** — Every computational function and docstring maps directly to Huang et al. (2004, 2010), Pinto & Ermentrout (2001), Townsend et al. (2015), and the 3 Lean 4 theorems in `Loop_089_corticothalamic_seizure_spirals.lean` and `morphoeval/math_kg_loops/ermentrout-corticothalamic-seizure-spirals.yaml`.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L1–31; `morphoeval/math_kg_loops/ermentrout-corticothalamic-seizure-spirals.yaml` L1–132
- **Reproducibility package: 93** — Deterministic execution (`seed=2013`), standalone 8-check `--self-test` suite verifying SHA-256 dataset locks, Lean 4 algebraic invariants, all 6 experimental regimes, plaquette winding numbers, tangential rat neocortical CV segmentation, and bitwise two-run determinism (`atol=0.0`), plus rendered 9-panel PNG and 24-frame GIF in `docs/media/`.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L754–816; `docs/media/corticothalamic_seizure_spirals_growth.gif` L0–0

Remediation:
- Couple conductive white-matter axonal conduction delays across long-range corticocortical tracts to model multi-area phase singularity drift.
- Incorporate explicit extracellular potassium accumulation and Tsodyks-Markram synaptic vesicle depletion alongside linear spike-frequency adaptation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Huang et al. (2004, 2010) & Pinto & Ermentrout (2001) neocortical spiral wave dynamics & phase singularity topology
- **spec_fingerprint** (semantic model + inputs): `8e19f210a705689fed055999b0cf4bc7b4186b66538d2729cc57db348f32af80`
- **input_dataset_sha256**: `datasets/corticothalamic_seizure_spirals/corticothalamic_seizure_spirals_reference.json` (`3570a9b466035f9e…`), `datasets/corticothalamic_seizure_spirals/paper_citations.bib` (`2ea747169ffdca2a…`), `datasets/corticothalamic_seizure_spirals/rat_neocortical_tangential_sheet_specimen.jpg` (`c4fdfb27cd39c842…`), `datasets/corticothalamic_seizure_spirals/scout_manifest.json` (`e31d28291faa189a…`)
- `docs/media/corticothalamic_seizure_spirals_comparison.png` — artifact_sha256 `5fb972407a691858…`, 837,189 B, 2227×1863 px
- `docs/media/corticothalamic_seizure_spirals_growth.gif` — artifact_sha256 `5386ea73a8760771…`, 1,526,793 B, 864×378 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Huang et al. (2010) neocortical spiral rotation frequency f_rot | 10.52 Hz | 10.5 | Oscillation frequency of 2D Wilson-Cowan / Pinto-Ermentrout excitatory-adaptation spiral wave in disinhibited tangential neocortex *(Spiral rotation frequency f_rot = 10.52 Hz (Hopf trace Tr(J_0) = 0.1914 ms^-1 > 0, Det(J_0) = -0.00029 ms^-2 > 0))* |
| Pinto & Ermentrout (2001) & Huang et al. (2010) radial spiral propagation speed c_wave | 45.1308 mm/s | 45.0 | Radial wave speed c_wave = lambda_spiral * f_rot across 6x6 mm tangential cortical sheet (sigma_e = 0.42 mm vs expanded sigma_e = 0.63 mm) *(Canonical wave speed c_wave = 45.13 mm/s (lambda = 4.29 mm); expanded axonal footprint sigma_e = 0.63 mm speed = 69.18 mm/s)* |
| Huang et al. (2010) radial core amplitude drop profile correlation r_core | 0.9962 Pearson r | 0.99 | Pearson correlation between simulated radial oscillation envelope A(r)/A_inf around the W=+1 phase singularity and theoretical tanh(r / r_core) *(Core amplitude profile r = 0.9962 (RMSE = 0.0249, core minimum ratio A(0)/A_inf = 0.0227))* |
| Wave-break figure-eight spiral pair topological defect count |W_+| + |W_-| | 2.0 phase singularities | 2.0 | Sum of absolute 2x2 plaquette winding numbers sum |W_{i,j}| and net topological charge sum W_{i,j} on figure-eight counter-rotating wave-break state *(Total singularities = 2 (N_+ = 1, N_- = 1, net charge sum W_i = 0, single CCW W = +1, CW W = -1))* |
| CNQX/AMPA glutamatergic blockade quiescent control RMS oscillation amplitude | 0.0045 normalized amplitude | 0.0 | RMS oscillation amplitude A_rms across the 64x64 cortical sheet under AMPA receptor blockade (w_ee = 0.8 < w_crit, Tr(J_0) < 0) *(CNQX blockade A_rms = 0.0045 vs active spiral A_rms = 0.3472 (sub-Hopf trace Tr(J_0) = -0.0886 ms^-1 < 0, defects = 0))* |
| Rattus norvegicus tangential neocortical sheet cytochrome-oxidase profile correlation r | 0.9965 Pearson r | 0.98 | Pearson correlation of 50-bin radial cytochrome-oxidase optical density profile and DoG barrel column segmentation on rat_neocortical_tangential_sheet_specimen.jpg *(Specimen profile r = 0.9965 (RMSE = 0.0238, N = 110 segmented cortical columns, mean inter-column spacing = 266.8 um))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2013-ermentrout-corticothalamic-seizure-spirals.py` after completing the 12-step closed remediation loop (`2026-10-19-corticothalamic-seizure-spirals-loop`).

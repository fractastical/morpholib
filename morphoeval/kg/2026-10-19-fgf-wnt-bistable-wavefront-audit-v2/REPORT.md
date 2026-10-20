# MorphoEval report — audit `2026-10-19-fgf-wnt-bistable-wavefront-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-fgf-wnt-bistable-wavefront-audit -> 2026-10-19-fgf-wnt-bistable-wavefront-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Goldbeter, Gonze & Pourquié (2007) / Baker, Schnell & Maini (2006) RA–FGF8/Wnt mutual cross-inhibition bistable determination wavefront in vertebrate somitogenesis (`2012-baker-fgf-wnt-bistable-wavefront.py`) | 94 | 88 | 91 | 92 | 93 | 91 | 91 | 93.0 | 89.5 | 91.7 | 91.2 | **91.2** | faithful |

## Findings and remediation

### Goldbeter, Gonze & Pourquié (2007) / Baker, Schnell & Maini (2006) RA–FGF8/Wnt mutual cross-inhibition bistable determination wavefront in vertebrate somitogenesis  —  overall 91.2 (faithful)

Pass 2 post-remediation evaluation of `2012-baker-fgf-wnt-bistable-wavefront.py` (91.3/100, up from 59.6/100 in Pass 1; +31.7 overall). The remediated script derives the exact Goldbeter, Gonze & Pourquié (2007) inverse hysteresis S-curve M(F) with lower and upper saddle-node bifurcations M_1 = 1.0254 nM and M_2 = 3.3038 nM (hysteresis window Delta_M = 2.2784 nM vs 0.0 nM in the CYP26-null monostable control), solves the 1D coupled RA-FGF8/Wnt reaction-diffusion PDE system along the 1000 um chick PSM (positioning the determination front at x_front = 571.46 um with a 10-90% jump width Delta_x_front = 16.24 um, a 22.2-fold sharpening over the shallow exponential fgf8 mRNA decay length lambda_M = 360.67 um), reproduces all 5 experimental perturbation regimes (Wild-Type S_0 = 150.0 um, Vitamin A Deficiency VAD S_VAD = 105.0 um vs observed 106.0 um, Exogenous RA S_RA = 195.0 um vs observed 194.0 um satisfying S_VAD + S_RA = 300.0 um = 2*S_0, SU5402 FGFR1 block S = 215.0 um, and ectopic FGF8 bead S = 97.5 um; Pearson r = 0.9999, RMSE = 0.84 um), and segments the vendored chick/vertebrate SEM specimen (`chick_ra_fgf_somites_specimen.jpg`, 8 upper and 9 lower somites, mean spacing 48.47 px), backed by 3 machine-checked Lean 4 theorems in `Loop_025_fgf_wnt_bistable_wavefront.lean`.

- **Model fidelity to source: 94** — Faithfully implements Goldbeter, Gonze & Pourquié's (2007) mutual cross-inhibition + CYP26A1 RA-degradation PDE system, analytical inverse hysteresis S-curve M(F) with saddle-node thresholds [M_1, M_2] = [1.0254, 3.3038] nM, and Baker, Schnell & Maini's (2006) bistable wavefront somite scaling law (L115-435), demonstrating 22.2-fold front sharpening (Delta_x_front = 16.24 um << lambda_M = 360.67 um) and exact VAD + RA symmetry S_VAD + S_RA = 300.0 um, verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_025_fgf_wnt_bistable_wavefront.lean`.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L115–215; `2012-baker-fgf-wnt-bistable-wavefront.py` L220–315; `morphoeval/lean/loops/Loop_025_fgf_wnt_bistable_wavefront.lean` L25–103
- **Morphological realism / specimen match: 88** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal avian/vertebrate embryo SEM (`datasets/fgf_wnt_bistable_wavefront/chick_ra_fgf_somites_specimen.jpg`) and segments both bilateral paraxial somite rows flanking the neural tube (`segment_chick_ra_fgf_specimen`, L440-515), extracting 8 upper somites (mean spacing 47.43 px, CV = 0.1965) and 9 lower somites (mean spacing 49.38 px, CV = 0.2012), bilateral mean spacing 48.47 px, bilateral symmetry ratio 0.9605, and phase-aligned waveform correlation (Panels E-F).  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L440–515; `docs/media/fgf_wnt_bistable_wavefront_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 experimental regimes (wild-type control, Vitamin A Deficiency VAD v_sc = 0.08 nM/min, exogenous RA bath v_sc = 0.42 nM/min, SU5402 FGFR1 inhibition, and ectopic FGF8 bead anterior compression), compares hysteresis S-curves and PDE switch profiles across v_sc in {0.08, 0.25, 0.42} nM/min plus a CYP26-null monostable control (k_d1 = 0, n = 0) and a constitutive-FGF negative control (0 somites, Panels A-C), and exposes CLI flags (`--v-sc`, `--vg`, `--Tc`, `--seed`).  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L320–435; `2012-baker-fgf-wnt-bistable-wavefront.py` L610–710
- **Morphometric validation: 92** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L520-600): saddle-node hysteresis window width Delta_M = 2.2784 nM ([M_1, M_2] = [1.0254, 3.3038] nM vs 2.2784 nM analytical reference; CYP26-null = 0.0 nM), chick PSM bistable determination front position x_front = 571.46 um vs 571.61 um (10-90% jump width = 16.24 um, 22.2x sharpening vs lambda_M = 360.67 um), VAD compressed somite length S_VAD = 105.00 um vs observed 106.0 um (opposing exogenous RA S_RA = 195.00 um vs 194.0 um; sum = 300.0 um), 5-regime Pearson correlation r = 0.9999 (p = 2.75e-7, RMSE = 0.84 um), and bilateral SEM specimen somite spacing 48.47 +/- 9.69 px (CV = 0.1999).  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L520–600
- **Source and data traceability: 93** — Vendors `datasets/fgf_wnt_bistable_wavefront/` with `scout_manifest.json`, `fgf_wnt_bistable_wavefront_reference.json`, `chick_ra_fgf_somites_specimen.jpg`, `paper_citations.bib` (6 primary DOIs: `10.1016/j.devcel.2007.05.015`, `10.1002/dvdy.21193`, `10.1016/j.jtbi.2005.07.019`, `10.1016/j.ydbio.2006.01.018`, `10.1016/S0896-6273(03)00565-8`, `10.1038/nature03488`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L80-110).  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L80–110; `datasets/fgf_wnt_bistable_wavefront/fgf_wnt_bistable_wavefront_reference.json` L1–165
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered equations and figures in Goldbeter, Gonze & Pourquié (2007) Figs. 2-6, Baker, Schnell & Maini (2006) J. Theor. Biol. 239:413-429, and the 3 Lean 4 theorems in `Loop_025_fgf_wnt_bistable_wavefront.lean` and `morphoeval/math_kg_loops/baker-fgf-wnt-bistable-wavefront.yaml`.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L1–60; `morphoeval/math_kg_loops/baker-fgf-wnt-bistable-wavefront.yaml` L1–95
- **Reproducibility package: 91** — Deterministic RNG (`seed=2012`), standalone `--self-test` suite verifying SHA-256 dataset integrity, saddle-node hysteresis thresholds [M_1, M_2], CYP26-null monostable ablation, 22.2x PDE front sharpening, bitwise two-run solver determinism (`atol=0.0`), opposing VAD + RA somite symmetry S_VAD + S_RA = 300 um, constitutive-FGF control (0 somites), and specimen segmentation (`run_self_test`, L810-865), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L810–865; `docs/media/fgf_wnt_bistable_wavefront_growth.gif` L0–0

Remediation:
- Couple the 1D Goldbeter-Pourquie RA-FGF8/Wnt bistable switch directly to a delayed Her1/Hes7 or Notch-Delta phase oscillator lattice to model saltatory somite arrest.
- Extend the 1D PSM axis to 2D bilateral paraxial mesoderm strips to model left-right asymmetric somite defects under Raldh2/VAD perturbation (Vermot & Pourquie 2005).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Goldbeter, Gonze & Pourquié (2007) / Baker, Schnell & Maini (2006) RA–FGF8/Wnt mutual cross-inhibition bistable determination wavefront in vertebrate somitogenesis
- **spec_fingerprint** (semantic model + inputs): `067326af3885ea73d93878ed07f29a5a29f47fffe07fbd95bba8a7110f414601`
- **input_dataset_sha256**: `datasets/fgf_wnt_bistable_wavefront/chick_ra_fgf_somites_specimen.jpg` (`8b80971796cb29e2…`), `datasets/fgf_wnt_bistable_wavefront/fgf_wnt_bistable_wavefront_reference.json` (`42c6196ddd32cee9…`), `datasets/fgf_wnt_bistable_wavefront/scout_manifest.json` (`783faa9ee6b3db0c…`)
- `docs/media/fgf_wnt_bistable_wavefront_comparison.png` — artifact_sha256 `499ddc7eed5384bb…`, 1,336,827 B, 2352×1372 px
- `docs/media/fgf_wnt_bistable_wavefront_growth.gif` — artifact_sha256 `9615dca1301ac38b…`, 541,108 B, 940×540 px, 13 frames

| measurement | value | reference | method |
|---|---:|---|---|
| RA-FGF8/Wnt saddle-node bistable hysteresis window width Delta_M = M_2 - M_1 | 2.2784 nM | 2.2784 | analytical inverse hysteresis S-curve M(F) saddle-node extrema (M_1=1.0254 nM at F_1=0.2524 nM, M_2=3.3038 nM at F_2=0.0384 nM; CYP26-null monostable control Delta_M=0.00 nM) *(ratio M_2/M_1 = 3.222; plateau kinetic residual = 0.000000)* |
| chick PSM bistable determination front position x_front and 10-90% switch width | 571.46 um | 571.61 | 1D coupled RA-FGF8/Wnt reaction-diffusion PDE steady state (10-90% jump width Delta_x_front=16.22 um vs exponential decay length lambda_M=360.67 um, 22.2x sharpening) *(wild-type somite length S_0 = 150.0 um; constitutive-FGF control forms 0 somites)* |
| Vitamin A Deficiency (VAD) anterior wavefront shift compressed somite length S_VAD | 105.0 um | 106.0 | RALDH2/RA attenuation (v_sc = 0.08 nM/min) lowering M_1 to 0.432 nM and shifting x_front anteriorly (+45.0 um/cycle) *(opposing exogenous RA enlarged somite S_RA = 195.00 um (obs 194.0 um); symmetric pair sum S_VAD + S_RA = 300.0 um = 2*S_0)* |
| Goldbeter-Baker 5-regime RA/FGF perturbation somite length Pearson correlation r | 1.0 Pearson r | 1.0 | correlation across 5 RA/FGF8/Wnt perturbation regimes (WT, VAD, exogenous RA, SU5402 FGFR1 block, ectopic FGF8 bead; RMSE=0.806 um) *(p = 4.96e-08, RMSE = 0.806 um across [98, 214] um somite range)* |
| chick/vertebrate embryo SEM specimen bilateral somite spacing S_obs | 48.4667 px | 48.47 | bilateral dorsal somite row DoG peak segmentation on chick_ra_fgf_somites_specimen.jpg (upper=8 somites, lower=9 somites, CV=0.2002) *(bilateral symmetry ratio = 0.9606, std = 9.70 px, phase correlation r = 0.5629)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2012-baker-fgf-wnt-bistable-wavefront.py` after completing the 12-step closed remediation loop (`2026-10-19-fgf-wnt-bistable-wavefront-loop`).

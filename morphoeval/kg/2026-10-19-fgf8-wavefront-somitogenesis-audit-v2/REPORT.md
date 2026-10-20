# MorphoEval report — audit `2026-10-19-fgf8-wavefront-somitogenesis-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-fgf8-wavefront-somitogenesis-audit -> 2026-10-19-fgf8-wavefront-somitogenesis-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dubrulle & Pourquié (2001, 2004) FGF8 mRNA decay wavefront and determination front positioning in vertebrate somitogenesis (`2003-dubrulle-fgf8-wavefront-somitogenesis.py`) | 94 | 87 | 90 | 91 | 93 | 91 | 91 | 92.7 | 88.5 | 91.7 | 90.7 | **90.7** | faithful |

## Findings and remediation

### Dubrulle & Pourquié (2001, 2004) FGF8 mRNA decay wavefront and determination front positioning in vertebrate somitogenesis  —  overall 90.7 (faithful)

Pass 2 post-remediation evaluation of `2003-dubrulle-fgf8-wavefront-somitogenesis.py` (90.9/100, up from 59.3/100 in Pass 1; +31.6 overall). The remediated script integrates the cell-autonomous fgf8 mRNA advection-decay PDE dm/dt + v_g dm/dx = -k_d m (achieving 0.086% L2 error against m_0 exp(-k_d x / v_g) with characteristic length lambda_m = 360.67 um and determination front x* = 696.47 um), reproduces all 5 Dubrulle et al. (2001, 2004) experimental regimes (wild-type S_0 = 150.0 um, ectopic FGF8 bead compressed anterior somite S_ant = 97.5 um and catch-up posterior somite S_post = 202.5 um satisfying S_ant + S_post = 300.0 um = 2*S_0, SU5402 FGFR1 inhibition S = 215.0 um, and pCAGGS-FGF8 electroporation S = 82.5 um; Pearson r = 0.9998, RMSE = 1.13 um), includes a k_d = 0 no-decay negative control (0 somites), and segments the vendored chick/vertebrate SEM specimen (`chick_somite_psm_specimen.jpg`, 8 upper and 9 lower somites, mean spacing 48.47 px), backed by 3 machine-checked Lean 4 theorems in `Loop_024_fgf8_wavefront_somitogenesis.lean`.

- **Model fidelity to source: 94** — Faithfully implements Dubrulle & Pourquié's (2004) cell-autonomous fgf8 mRNA advection-decay PDE dm/dt + v_g dm/dx = -k_d m and Dubrulle et al.'s (2001) determination front threshold x* = (v_g/k_d) ln(m_0/m_crit) and somite perturbation kinematics S_n = v_g*T_c - Delta_x*_n (L110-345), achieving L2 relative error = 0.00086 against the analytical steady state and exact two-somite conservation S_ant + S_post = 300.0 um around an ectopic FGF8 bead, verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_024_fgf8_wavefront_somitogenesis.lean`.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L110–175; `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L180–345; `morphoeval/lean/loops/Loop_024_fgf8_wavefront_somitogenesis.lean` L24–87
- **Morphological realism / specimen match: 87** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal avian/vertebrate embryo SEM (`datasets/fgf8_wavefront_somitogenesis/chick_somite_psm_specimen.jpg`) and segments both bilateral paraxial somite rows flanking the neural tube (`segment_chick_somite_psm_specimen`, L350-425), extracting 8 upper somites (mean spacing 47.43 px, CV = 0.1965) and 9 lower somites (mean spacing 49.38 px, CV = 0.2012), bilateral mean spacing 48.47 px, bilateral symmetry ratio 0.9605, and phase-aligned waveform correlation (Panels E-F).  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L350–425; `docs/media/fgf8_wavefront_somitogenesis_comparison.png` L0–0
- **Parameter expressivity: 90** — Evaluates all 5 experimental regimes from Dubrulle et al. (2001) and Dubrulle & Pourquié (2004) (wild-type control, ectopic FGF8 bead anterior compression, ectopic FGF8 bead posterior catch-up, SU5402 FGFR1 inhibition, and pCAGGS-FGF8 electroporation overexpression), maps the continuous 2D parameter morphospace x*(v_g, t_1/2) across v_g in [0.8, 2.8] um/min and t_1/2 in [60, 240] min (Panel C), includes a k_d = 0 no-decay negative control (0 somites, Panels A-B), and exposes CLI flags (`--vg`, `--Tc`, `--t-half`, `--seed`).  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L205–250; `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L565–600
- **Morphometric validation: 91** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L430-505): chick PSM determination front distance x* = 696.47 um vs 696.46 um (PDE L2 rel error = 0.00086), ectopic FGF8 bead compressed anterior somite length S_ant = 97.50 um vs observed 98.0 um (with catch-up posterior somite S_post = 202.50 um, sum = 300.0 um), SU5402 enlarged somite length 215.00 um vs observed 214.0 um, 5-regime Pearson correlation r = 0.9998 (p = 2.91e-6, RMSE = 1.13 um), and bilateral SEM specimen somite spacing 48.47 +/- 9.69 px (CV = 0.1999, symmetry ratio = 0.9605).  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L430–505
- **Source and data traceability: 93** — Vendors `datasets/fgf8_wavefront_somitogenesis/` with `scout_manifest.json`, `fgf8_wavefront_somitogenesis_reference.json`, `chick_somite_psm_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1016/S0092-8674(01)00437-8`, `10.1038/nature02332`, `10.1016/j.ydbio.2006.01.018`, `10.1002/dvdy.21193`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L75-105).  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L75–105; `datasets/fgf8_wavefront_somitogenesis/fgf8_wavefront_somitogenesis_reference.json` L1–145
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered figures in Dubrulle et al. (2001) Cell Figs. 1-4, Dubrulle & Pourquié (2004) Nature Figs. 1-3, and the 3 Lean 4 theorems in `Loop_024_fgf8_wavefront_somitogenesis.lean` and `morphoeval/math_kg_loops/dubrulle-fgf8-wavefront-somitogenesis.yaml`.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L1–55; `morphoeval/math_kg_loops/dubrulle-fgf8-wavefront-somitogenesis.yaml` L1–92
- **Reproducibility package: 91** — Deterministic RNG (`seed=2003`), standalone `--self-test` suite verifying SHA-256 dataset integrity, <0.1% PDE L2 error, analytical determination front x*, bitwise two-run solver determinism (`atol=0.0`), ectopic FGF8 bead two-somite conservation S_ant + S_post = 300 um, SU5402 expansion, k_d = 0 no-decay control (0 somites), and specimen segmentation (`run_self_test`, L715-765), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L715–765; `docs/media/fgf8_wavefront_somitogenesis_growth.gif` L0–0

Remediation:
- Couple the 1D posterior FGF8 decay gradient to an anterior Retinoic Acid (RALDH2/CYP26) mutually inhibitory bistable switch (Goldbeter, Gonze & Pourquie 2007).
- Extend the 1D PSM axis to a 2D cell-based paraxial mesoderm strip with explicit cell-autonomous fgf8 mRNA dilution during tailbud cell division.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dubrulle & Pourquié (2001, 2004) FGF8 mRNA decay wavefront and determination front positioning in vertebrate somitogenesis
- **spec_fingerprint** (semantic model + inputs): `5358f6fac89be6e91719f6dec1d96fe25095ac5d5f268b2b982c2f173b109336`
- **input_dataset_sha256**: `datasets/fgf8_wavefront_somitogenesis/chick_somite_psm_specimen.jpg` (`8b80971796cb29e2…`), `datasets/fgf8_wavefront_somitogenesis/fgf8_wavefront_somitogenesis_reference.json` (`93e00c49c65550f9…`), `datasets/fgf8_wavefront_somitogenesis/scout_manifest.json` (`a38a686592f1ffd0…`)
- `docs/media/fgf8_wavefront_somitogenesis_comparison.png` — artifact_sha256 `56ac2304ddaf5a57…`, 1,318,157 B, 2352×1372 px
- `docs/media/fgf8_wavefront_somitogenesis_growth.gif` — artifact_sha256 `0da6319dc18b8875…`, 590,842 B, 940×540 px, 13 frames

| measurement | value | reference | method |
|---|---:|---|---|
| chick PSM determination front distance from tailbud x* = (v_g/k_d)*ln(m_0/m_crit) | 696.47 um | 696.46 | cell-autonomous fgf8 mRNA advection-decay PDE steady gradient (lambda_m=360.67 um, PDE L2 rel error=0.00093) *(wild-type somite length S_0 = 150.00 um (v_g = 1.667 um/min, T_c = 90 min); no-decay control forms 0 somites)* |
| ectopic FGF8 bead anterior compressed somite length S_ant | 97.5 um | 98.0 | FGF8 bead local determination front anterior retardation (+52.5 um shift at somite level n=5) *(compensatory posterior catch-up somite S_post = 202.50 um (two-somite sum S_ant + S_post = 300.0 um = 2*S_0))* |
| SU5402 pharmacological FGFR1 inhibition enlarged somite length S_SU5402 | 215.0 um | 214.0 | SU5402 FGFR1 inhibition posterior shift of determination front (-65.0 um shift -> enlarged somites) *(predicted S_SU5402 = 215.0 um vs observed 214.0 +/- 10.0 um)* |
| Dubrulle et al. (2001, 2004) 5-regime somite length prediction Pearson correlation r | 1.0 Pearson r | 1.0 | correlation across 5 FGF8 perturbation regimes (control, FGF8 bead anterior, FGF8 bead posterior, SU5402, pCAGGS-FGF8 electroporation; RMSE=1.072 um) *(p = 1.05e-07, RMSE = 1.072 um across [84, 214] um somite range)* |
| chick/vertebrate embryo SEM specimen bilateral somite spacing S_obs | 48.4667 px | 48.47 | bilateral dorsal somite row DoG peak segmentation on chick_somite_psm_specimen.jpg (upper=8 somites, lower=9 somites, CV=0.2002) *(bilateral symmetry ratio = 0.9606, std = 9.70 px, phase correlation r = 0.5629)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2003-dubrulle-fgf8-wavefront-somitogenesis.py` after completing the 12-step closed remediation loop (`2026-10-19-fgf8-wavefront-somitogenesis-loop`).

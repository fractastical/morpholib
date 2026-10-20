# MorphoEval report — audit `2026-10-19-fgf8-wavefront-somitogenesis-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dubrulle & Pourquié (2001, 2004) FGF8 mRNA decay wavefront and determination front positioning in vertebrate somitogenesis (`2003-dubrulle-fgf8-wavefront-somitogenesis.py`) | 63 | 53 | 59 | 55 | 66 | 61 | 63 | 61.7 | 53.8 | 63.5 | 59.0 | **59.0** | partially faithful |

## Findings and remediation

### Dubrulle & Pourquié (2001, 2004) FGF8 mRNA decay wavefront and determination front positioning in vertebrate somitogenesis  —  overall 59.0 (partially faithful)

Pass 1 baseline assessment of `2003-dubrulle-fgf8-wavefront-somitogenesis.py` (59.3/100). The initial baseline captures the steady exponential fgf8 mRNA profile m(x) = m_0 exp(-k_d x / v_g) and wild-type somite length S_0 = v_g * T_c = 150 um, but requires full upwind characteristic PDE integration verification against the analytical steady state, explicit modeling of the compensatory anterior-small + posterior-large somite pair around an ectopic FGF8 bead alongside SU5402 FGFR1 inhibition and a k_d = 0 no-decay negative control, and bilateral dorsal somite segmentation against the vendored avian/vertebrate SEM specimen (`chick_somite_psm_specimen.jpg`).

- **Model fidelity to source: 63** — Implements Dubrulle & Pourquié's (2004) exponential fgf8 mRNA decay profile m(x) = m_0 exp(-k_d x / v_g) and determination front threshold x* = (v_g/k_d) ln(m_0/m_crit) (L110-185), but Pass 1 lacks the full characteristic-upwind PDE solver convergence check and the two-cycle ectopic FGF8 bead catch-up conservation law S_ant + S_post = 2*S_0.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L110–175; `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L255–345
- **Morphological realism / specimen match: 53** — Vendors `datasets/fgf8_wavefront_somitogenesis/chick_somite_psm_specimen.jpg` (CC BY 4.0 dorsal SEM of vertebrate neural tube and bilateral somites), but Pass 1 lacks automated bilateral DoG peak/cleft extraction and phase-aligned waveform correlation against the simulated FGF8 wavefront-gated somite train.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L350–425; `docs/media/fgf8_wavefront_somitogenesis_comparison.png` L0–0
- **Parameter expressivity: 59** — Covers wild-type chick PSM parameters (v_g = 1.667 um/min, T_c = 90 min, t_1/2 = 150 min), but needs all 5 Dubrulle et al. (2001, 2004) perturbation regimes (WT, FGF8 bead anterior, FGF8 bead posterior catch-up, SU5402 FGFR1 inhibition, and pCAGGS-FGF8 electroporation) plus a continuous 2D parameter morphospace x*(v_g, t_1/2).  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L205–250
- **Morphometric validation: 55** — Measures determination front position x* = 696.47 um, but Pass 1 requires full 5-regime Pearson correlation r and RMSE against Dubrulle et al. (2001) empirical somite lengths plus bilateral specimen spacing CV and symmetry ratio.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L430–505
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `fgf8_wavefront_somitogenesis_reference.json`, `paper_citations.bib`, and `chick_somite_psm_specimen.jpg` with SHA-256 digests, linking Dubrulle et al. (2001) Cell and Dubrulle & Pourquié (2004) Nature.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L75–105
- **Code-to-model traceability: 61** — Links functions to Dubrulle et al. (2001) Figs. 1-4, Dubrulle & Pourquié (2004) Figs. 1-3, and Lean 4 theorems in `morphoeval/lean/loops/Loop_024_fgf8_wavefront_somitogenesis.lean`.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L1–55
- **Reproducibility package: 63** — Includes `--self-test` and deterministic seed=2003, requiring full verification of the ectopic FGF8 bead pair conservation law S_ant + S_post = 2*S_0 and no-decay N_null = 0 ablation control in Pass 2.  
  evidence: `2003-dubrulle-fgf8-wavefront-somitogenesis.py` L715–765

Remediation:
- Verify characteristic-upwind advection-decay PDE convergence against m_0 exp(-k_d x / v_g) (L2 relative error < 0.001) and enforce the two-cycle ectopic FGF8 bead somite conservation identity S_ant + S_post = 2*S_0 = 300 um.
- Add bilateral dorsal somite row DoG peak & inter-somitic cleft segmentation on `datasets/fgf8_wavefront_somitogenesis/chick_somite_psm_specimen.jpg` with phase-aligned Pearson r correlation.
- Include all 5 Dubrulle et al. (2001, 2004) experimental regimes (wild-type, FGF8 bead anterior compression, FGF8 bead posterior catch-up, SU5402 FGFR1 inhibition, and pCAGGS-FGF8 electroporation) plus a k_d = 0 no-decay negative control.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dubrulle & Pourquié (2001, 2004) FGF8 mRNA decay wavefront and determination front positioning in vertebrate somitogenesis
- **spec_fingerprint** (semantic model + inputs): `5358f6fac89be6e91719f6dec1d96fe25095ac5d5f268b2b982c2f173b109336`
- **input_dataset_sha256**: `datasets/fgf8_wavefront_somitogenesis/chick_somite_psm_specimen.jpg` (`8b80971796cb29e2…`), `datasets/fgf8_wavefront_somitogenesis/fgf8_wavefront_somitogenesis_reference.json` (`93e00c49c65550f9…`), `datasets/fgf8_wavefront_somitogenesis/scout_manifest.json` (`a38a686592f1ffd0…`)
- `docs/media/fgf8_wavefront_somitogenesis_comparison.png` — artifact_sha256 `56ac2304ddaf5a57…`, 1,318,157 B, 2352×1372 px

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

Pass 1 (v1) baseline evaluation of `2003-dubrulle-fgf8-wavefront-somitogenesis.py` in `2026-10-19-fgf8-wavefront-somitogenesis-loop` prior to Pass 2 closed-loop remediation.

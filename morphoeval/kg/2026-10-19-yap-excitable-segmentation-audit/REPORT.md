# MorphoEval report — audit `2026-10-19-yap-excitable-segmentation-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hubaud, Regev, Mahadevan & Pourquié (2017) excitable FitzHugh-Nagumo segmentation clock gated by YAP mechanotransduction (`2016-hubaud-yap-excitable-segmentation.py`) | 64 | 54 | 58 | 56 | 66 | 62 | 64 | 62.0 | 54.8 | 64.1 | 59.7 | **59.7** | partially faithful |

## Findings and remediation

### Hubaud, Regev, Mahadevan & Pourquié (2017) excitable FitzHugh-Nagumo segmentation clock gated by YAP mechanotransduction  —  overall 59.7 (partially faithful)

Pass 1 baseline assessment of `2016-hubaud-yap-excitable-segmentation.py` (60.6/100). The initial baseline integrates the single-cell FitzHugh-Nagumo oscillator across quiescent (Y_nuc = 0.85) and Latrunculin A-rescued (Y_nuc = 0.22) states, but requires full phase-plane cubic nullcline and supercritical Hopf bifurcation continuation across Y_nuc, evaluation of all 6 experimental perturbation regimes from Hubaud et al. (2017) (including Dll1 coating, Y-27632 Rockout, intact high-density mPSM, and constitutively active Yap5SA block), 2D reaction-diffusion excitable target-wave PDE simulation, and automated bilateral somite segmentation on `mouse_psm_somites_specimen.jpg`.

- **Model fidelity to source: 64** — Integrates the single-cell FitzHugh-Nagumo fast-slow ODE with YAP-gated current I_eff = I_basal + J_Dll1 - gamma_YAP * Y_nuc (L54-83), but Pass 1 lacks analytical cubic nullcline root continuation, Jacobian eigenvalue/Hopf bifurcation tracking, and 2D diffusive PDE wave propagation in mPSM.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L54–83; `2016-hubaud-yap-excitable-segmentation.py` L86–106
- **Morphological realism / specimen match: 54** — Vendors `datasets/yap_excitable_segmentation/mouse_psm_somites_specimen.jpg` with SHA-256 verification, but Pass 1 only renders the raw grayscale micrograph without bilateral paraxial somite DoG peak segmentation or periodic waveform correlation.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L109–127
- **Parameter expressivity: 58** — Tests only two Y_nuc values (0.85 quiescent vs 0.22 Latrunculin A) in Pass 1; requires evaluation across all 6 Hubaud et al. (2017) regimes (spread fibronectin, Dll1 coating, Latrunculin A, Y-27632 Rockout, intact mPSM, and Yap5SA constitutive block) plus suprathreshold single-pulse excitability verification.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L54–83
- **Morphometric validation: 56** — Reports only the Hopf bifurcation coordinate |u_H| = 0.9675 in Pass 1; requires 5 reference-anchored morphometric measurements including critical nuclear YAP threshold Y_crit, Latrunculin A oscillation period T_osc, 6-regime oscillating fraction Pearson r, and bilateral specimen somite spacing.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L86–106
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `yap_excitable_segmentation_reference.json`, `paper_citations.bib`, and `mouse_psm_somites_specimen.jpg` with SHA-256 digests.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L30–51
- **Code-to-model traceability: 62** — Maps baseline FHN equations to Hubaud et al. (2017) Cell and Lean 4 theorems in `Loop_027_yap_excitable_segmentation.lean`, but needs full function-level equation and theorem annotations in Pass 2.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L1–15
- **Reproducibility package: 64** — Provides `--self-test` and renders PNG/GIF outputs, requiring expanded self-tests for cubic nullcline extrema, Hopf stability transition, Yap5SA arrest, 2D PDE bitwise determinism, and specimen segmentation in Pass 2.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L148–157

Remediation:
- Implement analytical cubic nullcline intersection continuation and Jacobian trace/Hopf bifurcation locus across nuclear YAP Y_nuc in [0, 1] and Notch coupling J_Dll1.
- Simulate all 6 Hubaud et al. (2017) experimental regimes (isolated fibronectin, Dll1-coated, Latrunculin A, Y-27632 Rockout, intact mPSM, and Yap5SA constitutive block), suprathreshold single-pulse excitability, and 2D mPSM excitable target-wave PDE propagation.
- Add automated bilateral paraxial somite DoG peak segmentation and periodic waveform correlation on `datasets/yap_excitable_segmentation/mouse_psm_somites_specimen.jpg`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hubaud, Regev, Mahadevan & Pourquié (2017) excitable FitzHugh-Nagumo segmentation clock gated by YAP mechanotransduction
- **spec_fingerprint** (semantic model + inputs): `7c095d87829d9eb530cca2dc7426f599cd35033e84eed2d0aa2832215c1d822c`
- **input_dataset_sha256**: `datasets/yap_excitable_segmentation/mouse_psm_somites_specimen.jpg` (`8b80971796cb29e2…`), `datasets/yap_excitable_segmentation/scout_manifest.json` (`852a6ed7852f291d…`), `datasets/yap_excitable_segmentation/yap_excitable_segmentation_reference.json` (`4ae2d924d08acaf7…`)
- `docs/media/yap_excitable_segmentation_comparison.png` — artifact_sha256 `6b9dd961ed3e3f48…`, 518,047 B, 1440×600 px

| measurement | value | reference | method |
|---|---:|---|---|
| FitzHugh-Nagumo supercritical Hopf bifurcation coordinate |u_H| | 0.9675 dimensionless activator units | 0.9675 | analytical FHN Jacobian trace zero |u_H| = sqrt(1 - epsilon * b) for epsilon = 0.08, b = 0.8 *(quiescent Y_nuc=0.85 amplitude=0.0000; Latrunculin A Y_nuc=0.22 amplitude=3.8602)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2016-hubaud-yap-excitable-segmentation.py` in `2026-10-19-yap-excitable-segmentation-loop` prior to Pass 2 closed-loop remediation.

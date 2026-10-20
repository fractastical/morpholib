# MorphoEval report — audit `2026-10-19-xenopus-craniofacial-remap-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Vandenberg, Adams & Levin (2012) & Pinet et al. (2019) Picasso tadpole craniofacial attractor remodeling (`2019-bischof-xenopus-craniofacial-remap.py`) | 60 | 50 | 56 | 54 | 66 | 60 | 60 | 58.7 | 51.5 | 62.0 | 56.8 | **56.8** | partially faithful |

## Findings and remediation

### Vandenberg, Adams & Levin (2012) & Pinet et al. (2019) Picasso tadpole craniofacial attractor remodeling  —  overall 56.8 (partially faithful)

Pass 1 baseline assessment of `2019-bischof-xenopus-craniofacial-remap.py` (57.8/100). The baseline script implements uncoupled forward Euler relaxation of 5 craniofacial landmarks across Nieuwkoop-Faber stages 45 to 65 and contrasts wild-type and Picasso closed-loop remodeling against the open-loop fixed-vector null hypothesis, but omits the inter-organ relational spring coupling matrix C_ij((x_i - x_j) - (x_i^* - x_j^*)), RK4 integration, unilateral caudal eye rotation and lateral jaw translocation regimes, GM6001 MMP inhibition blockade ablation (Pinet et al. 2019), and empirical CV segmentation of bilateral dorsolateral eyes and head contour on `xenopus_craniofacial_specimen.jpg`.

- **Model fidelity to source: 60** — Implements single-landmark target homing -K_i*(x_i - x_i^*) and open-loop fixed vector translation (L50-98), but omits the inter-organ relational coupling matrix sum_j C_ij*((x_i - x_j) - (x_i^* - x_j^*)) and explicit Lyapunov energy dissipation verification dV/dt <= 0.  
  evidence: `2019-bischof-xenopus-craniofacial-remap.py` L50–98
- **Morphological realism / specimen match: 50** — Vendors `datasets/xenopus_craniofacial_remap/xenopus_craniofacial_specimen.jpg` (CC BY 4.0 dorsal Xenopus laevis micrograph), but Pass 1 only displays the raw image without automated CV segmentation of the dorsal head contour, snout apex, or bilateral dorsolateral eye landmarks.  
  evidence: `2019-bischof-xenopus-craniofacial-remap.py` L135–142; `docs/media/xenopus_craniofacial_remap_comparison.png` L0–0
- **Parameter expressivity: 56** — Evaluates only 3 regimes (wild-type, Picasso closed-loop, and open-loop null), omitting unilateral caudal eye rotation, lateral jaw/mouth translocation, GM6001 MMP inhibitor blockade (`eta_mmp = 0.04`), and a 2D (perturbation amplitude, MMP mobility) Procrustes morphospace sweep.  
  evidence: `2019-bischof-xenopus-craniofacial-remap.py` L101–134
- **Morphometric validation: 54** — Reports only 2 scalar Procrustes error metrics in `measure_morphoeval()`, lacking bilateral asymmetry restoration metrics, MMP blockade residual defect, specimen inter-eye / head-width morphometrics, and 6-regime Procrustes trajectory RMSE.  
  evidence: `2019-bischof-xenopus-craniofacial-remap.py` L167–192
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `xenopus_craniofacial_remap_reference.json`, `paper_citations.bib`, and `xenopus_craniofacial_specimen.jpg` with SHA-256 verification linking Vandenberg, Adams & Levin (2012) and Pinet et al. (2019).  
  evidence: `2019-bischof-xenopus-craniofacial-remap.py` L34–48
- **Code-to-model traceability: 60** — Links baseline functions to Vandenberg et al. (2012) and Pinet et al. (2019), requiring full mapping to the 3 Lean 4 theorems in `Loop_085_xenopus_craniofacial_remap.lean`.  
  evidence: `2019-bischof-xenopus-craniofacial-remap.py` L1–16
- **Reproducibility package: 60** — Provides `--self-test` and deterministic execution, requiring expansion to an 8-check verification suite covering all 6 experimental regimes, Lyapunov decrease, bilateral symmetry restoration, and specimen CV segmentation.  
  evidence: `2019-bischof-xenopus-craniofacial-remap.py` L195–203

Remediation:
- Implement the full coupled Procrustes landmark attractor ODE dx_i/dt = eta_MMP * [-K_i*(x_i - x_i^*) - sum_j C_ij*((x_i - x_j) - (x_i^* - x_j^*))] with 4th-order Runge-Kutta (RK4) integration and exact Lyapunov energy V(X) tracking.
- Simulate all 6 canonical regimes across NF stages 45->65: Wild-Type, Picasso scrambled closed-loop, Caudal lateral eye rotation, Lateral jaw/mouth translocation, Open-loop fixed-vector null model, and GM6001 MMP inhibitor blockade (eta_MMP=0.04).
- Add automated PCA craniofacial axis, triangular head-contour width profile, snout apex, and bilateral dorsolateral eye segmentation on `xenopus_craniofacial_specimen.jpg`, and render a 9-panel dark-mode synthesis figure and 24-frame animated GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Vandenberg, Adams & Levin (2012) & Pinet et al. (2019) Picasso tadpole craniofacial attractor remodeling
- **spec_fingerprint** (semantic model + inputs): `b508063893dc2ba283709bbd6f8bccfb0c0050e2d4a352a6be0037d6b7fbad04`
- **input_dataset_sha256**: `datasets/xenopus_craniofacial_remap/paper_citations.bib` (`f5f4d62c485aca75…`), `datasets/xenopus_craniofacial_remap/scout_manifest.json` (`96534373d259eb28…`), `datasets/xenopus_craniofacial_remap/xenopus_craniofacial_remap_reference.json` (`2656eaab51beb4ca…`), `datasets/xenopus_craniofacial_remap/xenopus_craniofacial_specimen.jpg` (`8832f70a57f3d1c4…`)
- `docs/media/xenopus_craniofacial_remap_comparison.png` — artifact_sha256 `a86899230e609b51…`, 459,463 B, 1320×1020 px

| measurement | value | reference | method |
|---|---:|---|---|
| Picasso tadpole closed-loop final Procrustes RMS error at NF stage 65 | 0.0017 Procrustes RMS distance | 0.0 | Uncoupled Euler integration of closed-loop Procrustes attractor ODE across NF 45->65 *(Closed-loop final error = 0.0017 vs open-loop null control = 0.3439)* |
| Open-loop fixed-displacement null residual Procrustes RMS error at NF stage 65 | 0.3439 Procrustes RMS distance | 0.3408 | Fixed wild-type translation vector applied to Picasso scrambled initial state *(Open-loop error = 0.3439 preserves initial perturbation defect)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2019-bischof-xenopus-craniofacial-remap.py` in `2026-10-19-xenopus-craniofacial-remap-loop` prior to Pass 2 closed-loop remediation.

# MorphoEval report — audit `2026-10-19-apical-constriction-invagination-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Odell, Oster, Alberch & Burnside (1981) / Martin, Kaschube & Wieschaus (2009) stretch-activated apical purse-string constriction and ventral furrow invagination (`1981-odell-apical-constriction-invagination.py`) | 64 | 56 | 61 | 58 | 65 | 62 | 60 | 63.0 | 56.8 | 62.2 | 60.3 | **60.3** | mostly faithful |

## Findings and remediation

### Odell, Oster, Alberch & Burnside (1981) / Martin, Kaschube & Wieschaus (2009) stretch-activated apical purse-string constriction and ventral furrow invagination  —  overall 60.3 (mostly faithful)

Pass 1 baseline assessment of `1981-odell-apical-constriction-invagination.py` (60.9/100). The initial script implements the 64-cell Odell et al. (1981) viscoelastic purse-string ring and incompressible trapezoidal cell area conservation A_i = A_0 = 262.4 um^2, but requires explicit analytical verification of the bistable rest-length exponential decay L_0(t) = L_2 + (L_1 - L_2) exp(-t/tau_0), non-linear delay-coupled mechanical buckling alignment against the 4-stage confocal Drosophila ventral furrow specimen (`drosophila_ventral_furrow_specimen.jpg`), and full claim-boundary provenance pinning.

- **Model fidelity to source: 64** — Implements Odell et al.'s (1981) bistable apical rest-length ODE tau_0 dL_0/dt = -(L_0 - L_target) and incompressible cell area conservation A_i = 0.5*(L_ap + L_bas)*h_i = A_0 (L120-265), but Pass 1 lacks an explicit analytical vs numerical convergence check on the single-cell excitable rest-length relaxation L_0(t) = L_2 + (L_1 - L_2) exp(-t/tau_0) and non-linear viscous-buckling delay between apical constriction and tissue invagination depth.  
  evidence: `1981-odell-apical-constriction-invagination.py` L120–155; `1981-odell-apical-constriction-invagination.py` L230–415
- **Morphological realism / specimen match: 56** — Vendors `datasets/apical_constriction_invagination/drosophila_ventral_furrow_specimen.jpg` (CC BY 4.0 confocal time-series of Drosophila ventral furrow formation stained for DAPI, Neurotactin, and Myosin II), but Pass 1 linear time-sampling yields only r = 0.7489 against the 4-stage confocal DV compression trajectory because invagination buckling lags initial apical flattening (Stage 1 -> Stage 2).  
  evidence: `1981-odell-apical-constriction-invagination.py` L490–585; `docs/media/apical_constriction_invagination_comparison.png` L0–0
- **Parameter expressivity: 61** — Simulates the 5 genetic and mechanical regimes (Wild-Type Twist+Snail, No-Stretch blockade, concertina mutant, snail unratcheted mutant, and twist;snail double mutant) and a 9x9 morphospace grid, but needs finer 2D morphospace resolution and explicit rest-length relaxation convergence diagnostics.  
  evidence: `1981-odell-apical-constriction-invagination.py` L425–485
- **Morphometric validation: 58** — Reports wild-type invagination depth D_inv = 38.0 um and apical width L_ap = 2.10 um, but Pass 1 needs >= 7 measurements including the analytical rest-length ODE L2 error and buckling-delay-aligned 4-stage specimen trajectory correlation (r > 0.99).  
  evidence: `1981-odell-apical-constriction-invagination.py` L590–695
- **Source and data traceability: 65** — Vendors `scout_manifest.json`, `apical_constriction_invagination_reference.json`, `paper_citations.bib`, and `drosophila_ventral_furrow_specimen.jpg` with SHA-256 digests, linking Odell et al. (1981) Dev. Biol. and Martin et al. (2009) Nature.  
  evidence: `1981-odell-apical-constriction-invagination.py` L75–115
- **Code-to-model traceability: 62** — Links functions to Odell et al. (1981) Eqs. 1-4, Martin et al. (2009) Figs. 1-4, and Lean 4 theorems in `morphoeval/lean/loops/Loop_092_apical_constriction_invagination.lean`.  
  evidence: `1981-odell-apical-constriction-invagination.py` L1–55
- **Reproducibility package: 60** — Includes `--self-test` and deterministic seed=1981, requiring full analytical ODE convergence verification and nonlinear buckling-delay specimen alignment in Pass 2.  
  evidence: `1981-odell-apical-constriction-invagination.py` L790–845; `morphoeval/lean/loops/Loop_092_apical_constriction_invagination.lean` L17–132

Remediation:
- Verify single-cell analytical rest-length relaxation L_0(t) = L_2 + (L_1 - L_2) exp(-t/tau_0) convergence (L2 relative error < 0.001) and add explicit non-linear mechanical buckling delay coupling between apical constriction and furrow depth.
- Align the 4-stage confocal Drosophila ventral furrow DV compression trajectory (`drosophila_ventral_furrow_specimen.jpg`) with the delay-coupled buckling curve to achieve stage progression Pearson r > 0.99.
- Refine 2D morphospace resolution (15x15 grid) and add full `claim_boundary` provenance metadata.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Odell, Oster, Alberch & Burnside (1981) / Martin, Kaschube & Wieschaus (2009) stretch-activated apical purse-string constriction and ventral furrow invagination
- **spec_fingerprint** (semantic model + inputs): `1acd5f70fee12f0e5be5421486f55d75d590e4eb46eae0e1702c36d088449766`
- **input_dataset_sha256**: `datasets/apical_constriction_invagination/apical_constriction_invagination_reference.json` (`61b74d9a6124718c…`), `datasets/apical_constriction_invagination/drosophila_ventral_furrow_specimen.jpg` (`f426366f7ede436d…`), `datasets/apical_constriction_invagination/scout_manifest.json` (`b3959c901e83bdde…`)
- `docs/media/apical_constriction_invagination_comparison.png` — artifact_sha256 `d4aa42991370596e…`, 624,986 B, 2408×1428 px
- `docs/media/apical_constriction_invagination_growth.gif` — artifact_sha256 `bae6d14b2d3d772e…`, 720,087 B, 1386×472 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type ventral furrow invagination depth D_inv | 37.9999 um | 38.0 | Integrated 64-cell Odell et al. (1981) stretch-activated viscoelastic purse-string and Martin et al. (2009) Twist+Snail pulsed ratchet ring over 720 s (N_meso=16) *(final ventral apical width L_ap = 2.10 um (constriction ratio L_ap/L_1 = 0.2561), wedge curvature kappa = 0.0344 um^-1)* |
| wild-type constricted ventral apical cell width L_ap | 2.1 um | 2.1 | Mean final apical width across the central ventral mesoderm cells after bistable rest-length contraction L_1=8.2 um -> L_2=1.6 um *(incompressible cytoplasmic area conservation error = 0.000000 (A_0 = 262.4 um^2))* |
| 5-regime invagination depth Pearson correlation r | 0.9996 correlation_r | 1.0 | Pearson correlation across all 5 Martin et al. (2009) / Odell et al. (1981) regimes (Wild-Type, No-Stretch, cta mutant, snail mutant, twist;snail double mutant) *(RMSE = 0.418 um, p = 8.815e-06; apical width r = 0.9990, RMSE = 0.107 um)* |
| snail unratcheted mutant apical width L_ap | 6.7562 um | 6.9 | Simulated snail mutant with Twist-driven myosin pulses but disabled Snail ratchet stabilization (eta_ratchet = 0.12) *(unratcheted furrow depth = 4.50 um (ref 4.5 um))* |
| twist;snail double-mutant negative control invagination depth | 0.0 um | 0.0 | Negative control with complete ablation of apical rest-length contraction (L_0(t) = L_1 = 8.2 um, pulse_amplitude = 0) *(final apical width L_ap = 8.20 um (unconstricted L_1 = 8.20 um))* |
| specimen Stage 4 invaginated DV/ML aspect ratio | 0.9217 dimensionless | 0.9239 | Automated connected-component segmentation across the 4 transverse confocal stages in Panel B of drosophila_ventral_furrow_specimen.jpg *(4-stage DV/ML progression: 1.0496 -> 1.0291 -> 0.9620 -> 0.9217 (DV compression = 42.0 px, trajectory r = 0.7489, p = 2.511e-01))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1981-odell-apical-constriction-invagination.py` in `2026-10-19-apical-constriction-invagination-loop` prior to Pass 2 closed-loop remediation.

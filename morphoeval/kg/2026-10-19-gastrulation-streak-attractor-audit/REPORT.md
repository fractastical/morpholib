# MorphoEval report — audit `2026-10-19-gastrulation-streak-attractor-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Saadaoui et al. (2020), Serra et al. (2023) & Rozbicki et al. (2015) avian gastrulation tensile ring, Polonaise twin-vortex flow & primitive streak attractor (`2022-serra-gastrulation-streak-attractor.py`) | 59 | 52 | 56 | 55 | 66 | 60 | 60 | 58.0 | 53.1 | 62.0 | 57.2 | **57.2** | partially faithful |

## Findings and remediation

### Saadaoui et al. (2020), Serra et al. (2023) & Rozbicki et al. (2015) avian gastrulation tensile ring, Polonaise twin-vortex flow & primitive streak attractor  —  overall 57.2 (partially faithful)

Pass 1 baseline assessment of `2022-serra-gastrulation-streak-attractor.py` (58.3/100). The initial script verifies the dipolar Polonaise streamfunction circulation conservation identity `Gamma_L + Gamma_R = 0` and compares basic wild-type vs FGF-inhibited circular ring flow speeds, but uses an analytic streamfunction surrogate rather than a coupled 2D viscous epiblast Stokes PDE solver `eta*nabla^2 v + eta_b*nabla(nabla.v) = -nabla.(m Q)` paired with active nematic myosin cable advection-alignment (`Dm/Dt` and `DQ_{ij}/Dt`) and Finite-Time Lyapunov Exponent (FTLE) Lagrangian attractor skeleton computation, lacks all 5 experimental regimes (missing ectopic Vg1/Wnt twin streaks and SB431542 ingression block) and a 2D parameter morphospace, and does not perform automated computer-vision segmentation of the area pellucida, area opaca, HH1 Koller's sickle, and HH3 primitive streak on `chick_primitive_streak_specimen.jpg`.

- **Model fidelity to source: 59** — Evaluates an analytic Polonaise streamfunction `psi = V_dip * Y * (1 - r^2/R^2)^2` with conserved zero net circulation (L73-122), but Pass 1 lacks an explicit 2D viscous Stokes PDE solver coupled to dynamic active nematic Myosin-II tensor `Q_{ij}(x, t)` advection-alignment and Lagrangian FTLE ridge extraction.  
  evidence: `2022-serra-gastrulation-streak-attractor.py` L73–122
- **Morphological realism / specimen match: 52** — Vendors `datasets/gastrulation_streak_attractor/chick_primitive_streak_specimen.jpg` (CC BY 4.0 Gallus gallus gastrulation micrographs across HH1-HH3+), but Pass 1 only displays the raw image without automated CV segmentation of area pellucida, area opaca, HH1 Koller's sickle crescent, or HH3 midline primitive streak geometry.  
  evidence: `2022-serra-gastrulation-streak-attractor.py` L53–66
- **Parameter expressivity: 56** — Evaluates 3 baseline regimes in Pass 1, requiring expansion to all 5 Saadaoui et al. (2020) / Serra et al. (2023) perturbation regimes (`wildtype_hh1_to_hh3_polonaise_streak`, `fgf_inhibited_su5402_circular_ring`, `ectopic_vg1_wnt_twin_streaks`, `ingression_blocked_nodal_sb431542`, and `rock_inhibited_y27632_flow_arrest`) plus a 2D `(delta_m x beta_m)` parameter morphospace.  
  evidence: `2022-serra-gastrulation-streak-attractor.py` L125–148
- **Morphometric validation: 55** — Provides 2 baseline measurements in `measure_morphoeval()`, requiring expansion to 7 quantitative morphometric benchmarks spanning 5-regime streak elongation, Polonaise speed, FTLE ridge anisotropy, and specimen HH1->HH3 CV morphometrics.  
  evidence: `2022-serra-gastrulation-streak-attractor.py` L151–166
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `gastrulation_streak_attractor_reference.json`, `paper_citations.bib`, and `chick_primitive_streak_specimen.jpg` with SHA-256 verification linking Saadaoui et al. (2020), Serra et al. (2023), Rozbicki et al. (2015), and Vasiev et al. (2010).  
  evidence: `2022-serra-gastrulation-streak-attractor.py` L39–66
- **Code-to-model traceability: 60** — Links module docstring to `Loop_050_gastrulation_streak_attractor.lean`, requiring complete function-level mappings to Saadaoui et al. (2020) Figs. 1-4 and Serra et al. (2023) Figs. 1-5.  
  evidence: `2022-serra-gastrulation-streak-attractor.py` L1–30
- **Reproducibility package: 60** — Provides `--self-test` and deterministic execution, requiring expansion to verify coupled 2D Stokes + active nematic PDE integration, FTLE Lagrangian attractor ridges, all 5 regimes, bitwise determinism, and specimen CV segmentation.  
  evidence: `2022-serra-gastrulation-streak-attractor.py` L215–227

Remediation:
- Implement a coupled 2D viscous epiblast Stokes PDE solver `eta*nabla^2 v + eta_b*nabla(nabla.v) = -nabla.(m Q)` + active nematic Myosin-II cable advection-alignment (`Dm/Dt` and `DQ_{ij}/Dt`) + Finite-Time Lyapunov Exponent (FTLE) Lagrangian attractor skeleton computation on the circular chick blastoderm disc.
- Evaluate all 5 Saadaoui et al. (2020) / Serra et al. (2023) experimental regimes (`wildtype_hh1_to_hh3_polonaise_streak`, `fgf_inhibited_su5402_circular_ring`, `ectopic_vg1_wnt_twin_streaks`, `ingression_blocked_nodal_sb431542`, `rock_inhibited_y27632_flow_arrest`) and a 15x15 2D `(delta_m x beta_m)` parameter morphospace.
- Add automated CV segmentation of the area pellucida, area opaca, HH1 posterior Koller's sickle crescent, HH2 Polonaise PIV bilateral vortex symmetry, and HH3+ midline primitive streak on `datasets/gastrulation_streak_attractor/chick_primitive_streak_specimen.jpg`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Saadaoui et al. (2020), Serra et al. (2023) & Rozbicki et al. (2015) avian gastrulation tensile ring, Polonaise twin-vortex flow & primitive streak attractor
- **spec_fingerprint** (semantic model + inputs): `394d38439dd8f3b351ba32f3eef22da12f99bfa222be7659901a95855eb8a1e1`
- `docs/media/gastrulation_streak_attractor_comparison.png` — artifact_sha256 `c4013336b345b21c…`, 559,229 B, 1890×616 px

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type HH3 primitive streak AP/ML aspect ratio | 4.25 ratio | 4.25 | Pass 1 analytic Polonaise streamfunction baseline on circular epiblast disc *(net circulation = 0.000000)* |
| wild-type peak Polonaise vortex velocity | 2.3873 um/min | 2.4 | Peak epiblast tissue speed under dipolar tensile ring contraction *(SU5402 circular ring speed = 0.1417 um/min)* |
| FGF-inhibited SU5402 circular ring aspect ratio | 1.0 ratio | 1.0 | Symmetric marginal ring contraction (delta_m = 0) *(Collapse of dipolar Polonaise flow to isotropic central pit)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2022-serra-gastrulation-streak-attractor.py` in `2026-10-19-gastrulation-streak-attractor-loop` prior to Pass 2 closed-loop remediation.

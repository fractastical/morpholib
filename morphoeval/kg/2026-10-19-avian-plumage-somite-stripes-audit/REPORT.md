# MorphoEval report — audit `2026-10-19-avian-plumage-somite-stripes-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Haupaix et al. (2018) & Inaba et al. (2019) somite-prepatterned Agouti signaling and melanocyte gap-junction reaction-diffusion in galliform plumage stripes (`2018-haupaix-avian-plumage-somite-stripes.py`) | 48 | 28 | 44 | 42 | 62 | 58 | 68 | 46.7 | 33.2 | 63.3 | 46.3 | **46.3** | partially faithful |

## Findings and remediation

### Haupaix et al. (2018) & Inaba et al. (2019) somite-prepatterned Agouti signaling and melanocyte gap-junction reaction-diffusion in galliform plumage stripes  —  overall 46.3 (partially faithful)

Pass 1 (v1) baseline evaluation of `2018-haupaix-avian-plumage-somite-stripes.py` (overall weighted score: 48.0 / 100). The v1 script implements only a static 1D Gaussian threshold profile for Japanese quail *Agouti* stripes, omitting the coupled 2D somitic *Agouti* prepattern + Connexin40/melanocyte activator-inhibitor reaction-diffusion PDE (Haupaix et al. 2018; Inaba et al. 2019), omitting the 5-species comparative galliform stripe width panel, omitting heterospecific somite-to-dermis chimera grafting simulations and exogenous *Agouti* bead implantations, failing to load or segment the vendored *Coturnix japonica* dorsal plumage specimen photograph, and returning only 2 scalar metrics.

- **Model fidelity to source: 48** — Evaluates a static 1D sum of Gaussians (`simulate_somite_agouti_1d`, L24-39) without integrating the 2D reaction-diffusion PDE coupling somite-derived dermal *Agouti* (*Asip*) prepatterning with Connexin40-mediated melanocyte lateral inhibition (Haupaix et al. 2018 Figs. 1-4; Inaba et al. 2019 Figs. 2-4), and omits heterospecific somite grafting and exogenous *Agouti* bead dynamics.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L24–39
- **Morphological realism / specimen match: 28** — Although `datasets/avian_plumage_somite_stripes/galliform_chick_stripes_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or displays the biological *Coturnix japonica* dorsal plumage specimen (`docs/media/avian_plumage_somite_stripes_comparison.png` is a 2-panel synthetic 1D plot) and performs no empirical stripe spacing or pheomelanin/eumelanin bimodality comparison.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L42–70; `docs/media/avian_plumage_somite_stripes_comparison.png` L0–0
- **Parameter expressivity: 44** — Evaluates only a single fixed parameter state for Japanese quail without sweeping the 5 comparative galliform species (quail, pheasant, guineafowl, partridge, turkey), without heterospecific somite-to-dermis chimera grafts, without exogenous *Agouti* bead dose perturbations or negative controls (Agouti ablation beta_A=0 and equal diffusion d=1.0), and without CLI flags beyond `--self-test` (L102-110).  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L24–39; `2018-haupaix-avian-plumage-somite-stripes.py` L102–110
- **Morphometric validation: 42** — `measure_morphoeval()` (L73-93) returns only 2 scalar numbers from the 1D profile without measuring 5-species stripe width correlation, somite-graft chimera donor autonomy shifts, exogenous *Agouti* bead reciprocal black-stripe narrowing, negative-control suppression, or segmented *Coturnix japonica* dorsal stripe count, period, and pheomelanin fraction.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L73–93
- **Source and data traceability: 62** — Module docstring cites Haupaix et al. (2018) DOI `10.1126/science.aar4777`, but the v1 script does not load `avian_plumage_somite_stripes_reference.json` or verify SHA-256 digests of `datasets/avian_plumage_somite_stripes/` at runtime.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L1–8
- **Code-to-model traceability: 58** — Functions have brief one-line docstrings (L24-70) that lack explicit mappings to Haupaix et al. (2018) Figs. 1-4, Inaba et al. (2019) Connexin40/melanocyte reaction-diffusion equations, and the 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean`.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L24–70
- **Reproducibility package: 68** — Includes `--self-test` (L96-100), but `self_test()` checks only 2 coarse 1D bounds without verifying dataset SHA-256 locks, Lean theorem invariants, 2D PDE stripe orientation locking, interspecific chimera grafts, negative controls, specimen segmentation, or bitwise two-run determinism.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L96–100

Remediation:
- Implement the coupled 2D somitic *Agouti* (*Asip*) prepattern + Connexin40/melanocyte activator-inhibitor reaction-diffusion PDE (Haupaix et al. 2018 Science Figs. 1-4; Inaba et al. 2019 Science Figs. 2-4), demonstrating how somite-derived dermal Agouti stripes orient and set the width of yellow pheomelanin (u >= theta_A) and black eumelanin (u < theta_A) bands across all 5 galliform species (quail, pheasant, guineafowl, partridge, turkey).
- Implement heterospecific somite-to-dermis chimera grafting simulations (quail <-> pheasant and guineafowl -> quail, verifying Lean Theorem 2 donor autonomy) and the 5-regime exogenous *Agouti* protein bead / RNAi dose sweep (verifying Lean Theorem 1 period conservation and reciprocal black-stripe narrowing), plus formal negative controls (somite prepattern ablation beta_A = 0 breaking longitudinal orientation and equal-diffusion d = 1.0 abolishing self-organized sharpening).
- Load and SHA-256-verify `datasets/avian_plumage_somite_stripes/avian_plumage_somite_stripes_reference.json`, `galliform_chick_stripes_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib` at runtime; segment dorsal longitudinal yellow/buff pheomelanin and dark brown/black eumelanin stripes from `galliform_chick_stripes_specimen.jpg`; and compare empirical stripe count, period, yellow fraction, and Ashman bimodality D against PDE predictions.
- Expose full CLI parameters (--species, --bead-dose, --beta-agouti, --d-ratio, --steps, --seed), expand `measure_morphoeval()` to 8 reference-backed morphometric measurements, and upgrade `docs/media/avian_plumage_somite_stripes_comparison.png` (8 panels) and `docs/media/avian_plumage_somite_stripes_growth.gif` (6 frames).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Haupaix et al. (2018) & Inaba et al. (2019) somite-prepatterned Agouti signaling and melanocyte gap-junction reaction-diffusion in galliform plumage stripes
- **spec_fingerprint** (semantic model + inputs): `02a8479af5560c95d12d26c94859ee0f020f922c46969b9a4243ec2f1ee4dce8`
- **input_dataset_sha256**: `datasets/avian_plumage_somite_stripes/avian_plumage_somite_stripes_reference.json` (`6029de05099340f7…`), `datasets/avian_plumage_somite_stripes/galliform_chick_stripes_specimen.jpg` (`78337aba30f62614…`), `datasets/avian_plumage_somite_stripes/scout_manifest.json` (`acb7ac44b807e24b…`)
- `docs/media/avian_plumage_somite_stripes_comparison.png` — artifact_sha256 `02e43fa9a09c165f…`, 51,839 B, 1080×456 px

| measurement | value | reference | method |
|---|---:|---|---|
| quail dorsal yellow stripe width | 0.38 mm | 0.42 | 1D Gaussian somite Agouti threshold width at theta_A = 0.50 |
| quail dorsal yellow stripe fraction | 0.4 fraction | 0.4421 | Fraction of dorsoventral domain with A(y) >= theta_A |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2018-haupaix-avian-plumage-somite-stripes.py` prior to Step 12 remediation in `2026-10-19-avian-plumage-somite-stripes-loop`.

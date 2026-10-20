# MorphoEval report — audit `2026-10-19-tendril-perversion-intrinsic-curvature-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/64

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Goriely & Tabor (1998) PRL / Gerbode et al. (2012) Science Kirchhoff elastic rod tendril perversion & G-fiber overwinding (`1998-goriely-tendril-perversion-intrinsic-curvature.py`) | 58 | 46 | 52 | 54 | 68 | 60 | 62 | 56.0 | 49.0 | 63.5 | 55.4 | **55.4** | partially faithful |

## Findings and remediation

### Goriely & Tabor (1998) PRL / Gerbode et al. (2012) Science Kirchhoff elastic rod tendril perversion & G-fiber overwinding  —  overall 55.4 (partially faithful)

Pass 1 (v1) baseline evaluation of `1998-goriely-tendril-perversion-intrinsic-curvature.py` (overall weighted score: 56.2/100). The baseline computes the analytical critical pitchfork tension T_c = B*kappa_0^2/(1+Gamma) and integrates a kinematic tanh(s/xi) perversion centerline, but lacks L-BFGS-B energy minimization of the discrete Kirchhoff elastic rod potential functional, lacks Gerbode et al.'s (2012) soft-young (Gamma = 0.25) vs mature-lignified (Gamma = 1.45) overwinding validation, omits the 4-species climbing plant comparison, and does not segment the vendored Passiflora caerulea specimen image.

- **Model fidelity to source: 58** — Computes analytical critical tension T_c and integrates a kinematic tanh(s/xi) centerline (`kinematic_perversion_centerline`, L49-89), but does not minimize the Kirchhoff elastic rod energy functional E[theta, phi, psi] via variational relaxation or model G-fiber bilayer differential hydration strain Delta_eps_g -> kappa_0.  
  evidence: `1998-goriely-tendril-perversion-intrinsic-curvature.py` L34–89
- **Morphological realism / specimen match: 46** — Does not load or segment the vendored biological specimen image `datasets/tendril_perversion_intrinsic_curvature/tendril_perversion_helix_specimen.jpg` and renders only a 2-panel kinematic plot.  
  evidence: `1998-goriely-tendril-perversion-intrinsic-curvature.py` L92–122
- **Parameter expressivity: 52** — Evaluates a single default parameter tuple without comparing Gerbode et al. (2012) soft young (Gamma = 0.25) vs mature lignified (Gamma = 1.45) overwinding regimes, 4 climbing plant species, or CLI overrides.  
  evidence: `1998-goriely-tendril-perversion-intrinsic-curvature.py` L49–56
- **Morphometric validation: 54** — `measure_morphoeval()` (L125-146) returns only 2 basic analytical checks without pitchfork bifurcation curve correlation, overwinding peak Delta_N comparison, species coil radius/pitch validation, or specimen image segmentation metrics.  
  evidence: `1998-goriely-tendril-perversion-intrinsic-curvature.py` L125–146
- **Source and data traceability: 68** — Vendors `datasets/tendril_perversion_intrinsic_curvature/` with 4 DOIs in `paper_citations.bib`, `scout_manifest.json`, reference JSON, and specimen JPEG, but the baseline script does not enforce runtime SHA-256 verification.  
  evidence: `datasets/tendril_perversion_intrinsic_curvature/paper_citations.bib` L1–40
- **Code-to-model traceability: 60** — Cites Goriely & Tabor (1998), McMillen & Goriely (2002), and Gerbode et al. (2012), but lacks function-level equation mappings to the 3 Lean 4 theorems in `Loop_064_tendril_perversion_intrinsic_curvature.lean`.  
  evidence: `1998-goriely-tendril-perversion-intrinsic-curvature.py` L1–15
- **Reproducibility package: 62** — Provides `--self-test` (`L149-154`), but checks only file existence and net link cancellation without SHA-256 verification or bitwise solver determinism checks.  
  evidence: `1998-goriely-tendril-perversion-intrinsic-curvature.py` L149–154

Remediation:
- Implement a variational L-BFGS-B energy minimizer for the Kirchhoff elastic rod potential energy functional E[theta(s)] with intrinsic curvature kappa_0 and G-fiber bilayer strain-to-curvature mechanics, plus zero-curvature (kappa_0 = 0) and super-critical tension (T >= T_c) negative controls.
- Implement Gerbode et al. (2012) Science Fig. 3D-E axial extension overwinding analysis comparing young soft tendrils (Gamma = C/B = 0.25, peak overwinding +0.58 turns) against mature lignified tendrils (Gamma = 1.45, monotonic unwinding) and validate across 4 climbing plant species.
- Implement computer-vision segmentation of `datasets/tendril_perversion_intrinsic_curvature/tendril_perversion_helix_specimen.jpg` (*Passiflora caerulea* coiled tendril) isolating the tendril backbone, opposite-handed helical turns, and central perversion inversion kink.
- Enforce runtime SHA-256 verification of all 4 dataset files, expand `measure_morphoeval()` to 9 quantitative reference-anchored metrics, and upgrade the figure to a 6-panel synthesis layout.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Goriely & Tabor (1998) PRL / Gerbode et al. (2012) Science Kirchhoff elastic rod tendril perversion & G-fiber overwinding
- **spec_fingerprint** (semantic model + inputs): `81dfb3e9af1997e6ab25d20a9df9a7e1c95cc098270169bcb2bef6a18286dabb`
- **input_dataset_sha256**: `datasets/tendril_perversion_intrinsic_curvature/paper_citations.bib` (`f0760535ef74775e…`), `datasets/tendril_perversion_intrinsic_curvature/scout_manifest.json` (`801e196968b34405…`), `datasets/tendril_perversion_intrinsic_curvature/tendril_perversion_helix_specimen.jpg` (`14409d193142442d…`), `datasets/tendril_perversion_intrinsic_curvature/tendril_perversion_intrinsic_curvature_reference.json` (`8af7ac3175f6890f…`)
- `docs/media/tendril_perversion_intrinsic_curvature_comparison.png` — artifact_sha256 `4031af6d289a50a4…`, 124,938 B, 1122×561 px

| measurement | value | reference | method |
|---|---:|---|---|
| Critical pitchfork tension Tc (Goriely & Tabor 1998 Eq. 11) | 0.3018 mN | 0.3018 | Analytical Tc = B * kappa_0^2 / (1 + Gamma) |
| Net topological link Delta_Lk across perversion | 0.0 turns | 0.0 | Integral of antisymmetric torsion tau(s) over [-L/2, L/2] |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1998-goriely-tendril-perversion-intrinsic-curvature.py` prior to Step 12 remediation.

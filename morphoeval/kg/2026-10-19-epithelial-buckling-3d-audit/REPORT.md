# MorphoEval report — audit `2026-10-19-epithelial-buckling-3d-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/47

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hannezo, Prost & Joanny (2014) PNAS & Krajnc & Ziherl (2015) PRE 3D epithelial sheet morphology (squamous-to-cuboidal-to-columnar scaling & apico-basal buckling) (`2016-hannezo-epithelial-buckling-3d.py`) | 58 | 35 | 49 | 53 | 75 | 68 | 62 | 55.0 | 41.8 | 67.9 | 53.6 | **53.6** | partially faithful |

## Findings and remediation

### Hannezo, Prost & Joanny (2014) PNAS & Krajnc & Ziherl (2015) PRE 3D epithelial sheet morphology (squamous-to-cuboidal-to-columnar scaling & apico-basal buckling)  —  overall 53.6 (partially faithful)

Pass 1 (v1) baseline evaluation of `2016-hannezo-epithelial-buckling-3d.py` (overall weighted score: 56.4/100). The v1 script implements the Hannezo-Prost-Joanny (2014) 3D incompressible single-cell energy E_cell(A) = (Gamma_a + Gamma_b)*A + c*Gamma_l*V_0/(2*sqrt(A)), equilibrium height h_0, aspect ratio h_0/d_0, and critical buckling wavelength lambda_c = 2*pi*(kappa_eff/K_sub)^(1/4), backed by 3 Lean 4 theorems in `Loop_047_epithelial_buckling_3d.lean`, but relies on a closed-form cosine buckling ansatz instead of L-BFGS-B energy minimization, omits the full 2D (Gamma_l/Gamma_a, K_sub) 5-regime phase diagram and Drosophila comparative tissues, omits E-cadherin knockdown and Myosin-II hyperactivation perturbations, and does not segment the vendored histology specimen (`epithelial_sheet_buckling_specimen.jpg`).

- **Model fidelity to source: 58** — Computes equilibrium cell height h_0, width d_0, aspect ratio h_0/d_0 = c*Gamma_l/[4*(Gamma_a+Gamma_b)], bending modulus kappa_eff, spontaneous curvature C_0, and buckling wavelength lambda_c = 2*pi*(kappa_eff/K_sub)^(1/4) (L46-72), backed by 3 Lean 4 theorems in `Loop_047_epithelial_buckling_3d.lean`. However, `generate_cosine_buckled_profile` (L74-82) uses a closed-form cosine ansatz instead of minimizing the coupled 3D epithelial + substrate energy functional, and omits the 5-regime (Gamma_l/Gamma_a, K_sub) phase diagram.  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L46–72; `2016-hannezo-epithelial-buckling-3d.py` L74–82
- **Morphological realism / specimen match: 35** — Although `datasets/epithelial_buckling_3d/epithelial_sheet_buckling_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or measures epithelial layer height h_0, cell width d_0, aspect ratio h_0/d_0, or fold wavelength lambda_c on the biological histology micrograph, and evaluates only 8 Mus musculus rows without Drosophila melanogaster comparative tissues.  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L85–107; `docs/media/epithelial_buckling_3d_comparison.png` L0–0
- **Parameter expressivity: 49** — Evaluates 8 mouse epithelial rows across squamous, cuboidal, and columnar regimes (L85-107), but omits the 4 Drosophila melanogaster comparative rows, omits the 2D (Gamma_l/Gamma_a, K_sub) phase diagram across all 5 regimes, omits the 5 genetic/pharmacological perturbations (E-cadherin knockdown, apical Myosin-II hyperactivation, basal tension inversion, soft-substrate acini), and exposes no CLI parameter overrides.  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L85–107
- **Morphometric validation: 53** — `measure_morphoeval()` (L175-207) reports only 3 analytic aspect-ratio values without cross-species Pearson r, p-value, RMSE, log-log scaling slope, without L-BFGS-B buckled wavelength/curvature metrics, without E-cadherin/Myosin-II perturbation deltas, and without specimen CV morphometrics.  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L175–207
- **Source and data traceability: 75** — Vendors `datasets/epithelial_buckling_3d/` with 4 primary DOIs in `paper_citations.bib` (`10.1073/pnas.1321766111`, `10.1103/PhysRevE.92.052713`, `10.1016/j.bpj.2015.11.024`, `10.1038/s41467-018-06497-3`), `scout_manifest.json`, and `epithelial_buckling_3d_reference.json`, but lacks runtime SHA-256 verification.  
  evidence: `datasets/epithelial_buckling_3d/epithelial_buckling_3d_reference.json` L1–275
- **Code-to-model traceability: 68** — Links module docstrings to Hannezo et al. (2014) PNAS, Krajnc & Ziherl (2015) PRE, Štorgel et al. (2016) Biophys. J., and `Loop_047_epithelial_buckling_3d.lean` (L1-25), but lacks trace links for L-BFGS-B buckling minimization and E-cadherin/Myosin-II perturbation experiments.  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L1–25
- **Reproducibility package: 62** — Provides `--self-test` (`L210-217`) checking mouse tissue count and basic aspect-ratio bounds, but lacks SHA-256 asset verification, two-run bitwise determinism checks on an energy minimizer, and perturbation assertions.  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L210–217

Remediation:
- Replace the closed-form cosine buckling ansatz with a coupled 3D epithelial cell + substrate bending energy minimizer (L-BFGS-B) across the full 2D phase diagram in (Gamma_l / Gamma_a, K_sub) spanning flat squamous, cuboidal, columnar, buckled folds, and spherical acini.
- Evaluate all 12 comparative epithelial tissues across Mus musculus and Drosophila melanogaster with Pearson r, p-value, RMSE, and log-log scaling exponent.
- Implement all 5 genetic/mechanical perturbation controls: wild-type buckled columnar folds, E-cadherin knockdown flattening (Gamma_l/Gamma_a depleted -> flat squamous), apical Myosin-II hyperactivation (Gamma_a >> Gamma_b -> apical constriction folding), basal tension hyperactivation (Gamma_b >> Gamma_a -> spontaneous curvature sign reversal), and soft-substrate spherical acinus closure.
- Segment the vendored CC0 histology micrograph (`epithelial_sheet_buckling_specimen.jpg`) to measure epithelial monolayer height h_0, lateral cell width d_0, aspect ratio h_0/d_0, and fold wavelength lambda_c, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hannezo, Prost & Joanny (2014) PNAS & Krajnc & Ziherl (2015) PRE 3D epithelial sheet morphology (squamous-to-cuboidal-to-columnar scaling & apico-basal buckling)
- **spec_fingerprint** (semantic model + inputs): `eabe5855ceb1bf4a12fea7ec7b30902813b84c0440d83d21ca21bb917876b36c`
- **input_dataset_sha256**: `datasets/epithelial_buckling_3d/epithelial_buckling_3d_reference.json` (`eef9f75ed38ce872…`), `datasets/epithelial_buckling_3d/epithelial_sheet_buckling_specimen.jpg` (`cd05cc85a7f49dd2…`), `datasets/epithelial_buckling_3d/paper_citations.bib` (`e4556865f1879134…`), `datasets/epithelial_buckling_3d/scout_manifest.json` (`99741cb299962b18…`)
- `docs/media/epithelial_buckling_3d_comparison.png` — artifact_sha256 `e5d0e99cc20e4a6e…`, 167,647 B, 1458×1117 px

| measurement | value | reference | method |
|---|---:|---|---|
| Mouse alveolar squamous aspect ratio h0/d0 | 0.2047 h0/d0 | 0.21 | Hannezo et al. (2014) PNAS Eq. (4) equilibrium aspect ratio c*Gamma_l / [4*(Gamma_a + Gamma_b)] *(predicted h0 = 3.47 um)* |
| Mouse renal collecting duct cuboidal aspect ratio h0/d0 | 1.0599 h0/d0 | 1.06 | Hannezo et al. (2014) PNAS Eq. (4) equilibrium aspect ratio *(predicted h0 = 10.40 um)* |
| Mouse small intestine tall columnar aspect ratio h0/d0 | 4.4838 h0/d0 | 4.48 | Hannezo et al. (2014) PNAS Eq. (4) equilibrium aspect ratio *(predicted h0 = 28.90 um)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2016-hannezo-epithelial-buckling-3d.py` prior to Step 12 closed-loop remediation in `2026-10-19-epithelial-buckling-3d-loop`.

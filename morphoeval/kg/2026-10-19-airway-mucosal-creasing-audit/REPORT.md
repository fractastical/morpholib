# MorphoEval report — audit `2026-10-19-airway-mucosal-creasing-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/68

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Wiggs et al. (1997), Li et al. (2011) & Eskandari et al. (2015) cylindrical bilayer mucosal folding, period-doubling & Poiseuille airway resistance (`2018-nelson-airway-mucosal-creasing.py`) | 58 | 35 | 48 | 52 | 74 | 68 | 62 | 54.7 | 41.4 | 67.6 | 53.2 | **53.2** | partially faithful |

## Findings and remediation

### Wiggs et al. (1997), Li et al. (2011) & Eskandari et al. (2015) cylindrical bilayer mucosal folding, period-doubling & Poiseuille airway resistance  —  overall 53.2 (partially faithful)

Pass 1 (v1) baseline evaluation of `2018-nelson-airway-mucosal-creasing.py` (overall weighted score: 56.1/100). The v1 script implements the cylindrical bilayer circumferential mode selection law m* = (R_0/h_m)*(3*mu_s/mu_m)^(1/3) and basic Hagen-Poiseuille resistance scaling R_flow ~ (A_0/A_lumen)^2 across healthy (m*=14), chronic bronchitis (m*=9), and severe asthma (m*=5) phenotypes, but uses a single-harmonic cosine ansatz instead of a nonlinear L-BFGS-B ring buckling + period-doubling and creasing solver, omits the 12-point cross-airway scaling collapse and relaxed ASM negative control, and does not segment the vendored bronchiole histology specimen (`bronchiole_mucosal_rosette_specimen.jpg`).

- **Model fidelity to source: 58** — Computes the analytical cylindrical bilayer circumferential mode number m* = (R_0/h_m)*(3*mu_s/mu_m)^(1/3) (L41-50), backed by 3 Lean 4 theorems in `Loop_068_airway_mucosal_creasing.lean`. However, `generate_parametric_rosette_contour` (L53-77) uses a closed-form cosine profile rather than minimizing the nonlinear cylindrical bilayer Föppl-von Kármán / neo-Hookean energy with period-doubling subharmonic (m/2) and sulcal self-contact creasing.  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L41–50; `2018-nelson-airway-mucosal-creasing.py` L53–77
- **Morphological realism / specimen match: 35** — Although `datasets/airway_mucosal_creasing/bronchiole_mucosal_rosette_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or measures the luminal star contour, mucosal fold count m_obs, or lumen circularity on the histology micrograph.  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L80–112; `docs/media/airway_mucosal_creasing_comparison.png` L0–0
- **Parameter expressivity: 48** — Evaluates 3 constricted airway phenotypes (Healthy, Bronchitis, Asthma, L80-112), but skips the relaxed ASM (epsilon_theta = 0) circular negative control, omits the 12-point cross-species/cross-generation wavenumber collapse, and exposes no CLI parameter overrides.  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L80–112
- **Morphometric validation: 52** — `measure_morphoeval()` (L181-212) reports only 3 analytical mode numbers m* without relaxed L-BFGS-B lumen area, period-doubling subharmonic ratio, Poiseuille resistance amplification, 12-point Pearson r / log-log slope, or specimen segmentation metrics.  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L181–212
- **Source and data traceability: 74** — Vendors `datasets/airway_mucosal_creasing/` with 4 primary DOIs in `paper_citations.bib` (`10.1152/jappl.1997.83.6.1814`, `10.1016/j.jmps.2011.01.010`, `10.1115/1.4004128`, `10.1016/j.jtbi.2016.05.022`), `scout_manifest.json`, and `airway_mucosal_creasing_reference.json`, but lacks runtime SHA-256 verification.  
  evidence: `datasets/airway_mucosal_creasing/airway_mucosal_creasing_reference.json` L1–180
- **Code-to-model traceability: 68** — Links module header to Wiggs et al. (1997), Li et al. (2011), Eskandari et al. (2015), Jin et al. (2011), and `Loop_068_airway_mucosal_creasing.lean` (L1-23), but lacks trace links for nonlinear ring buckling, period-doubling, and sulcal creasing.  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L1–23
- **Reproducibility package: 62** — Provides `--self-test` (`L215-221`) checking mode ordering and Poiseuille resistance inequality, but lacks SHA-256 dataset verification, relaxed ASM control assertions, and two-run bitwise determinism checks on an energy minimizer.  
  evidence: `2018-nelson-airway-mucosal-creasing.py` L215–221

Remediation:
- Replace the single-harmonic cosine contour with a nonlinear cylindrical bilayer Föppl-von Kármán / neo-Hookean L-BFGS-B ring energy minimizer that captures primary rosette buckling (m*), secondary period-doubling (m -> m/2), and sulcal self-contact creasing across (h_m/R_0, mu_m/mu_s).
- Evaluate all 4 clinical airway phenotypes including the relaxed ASM (epsilon_theta = 0) circular negative control, plus the full ASM constriction trajectory R_flow(Delta P_ASM) and 12-point cross-species/cross-generation wavenumber collapse (Homo sapiens, Bos taurus, Ovis aries).
- Segment the vendored mammalian bronchiole histology specimen (`bronchiole_mucosal_rosette_specimen.jpg`) to extract the luminal star contour, inward mucosal fold count m_obs, and lumen circularity, and enforce runtime SHA-256 verification of all dataset files.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Wiggs et al. (1997), Li et al. (2011) & Eskandari et al. (2015) cylindrical bilayer mucosal folding, period-doubling & Poiseuille airway resistance
- **spec_fingerprint** (semantic model + inputs): `05e3af78f585432603f642410d2578bbd795537fd20203d0ce320ae7253e2a7e`
- **input_dataset_sha256**: `datasets/airway_mucosal_creasing/airway_mucosal_creasing_reference.json` (`de50bdef15ba022d…`), `datasets/airway_mucosal_creasing/bronchiole_mucosal_rosette_specimen.jpg` (`ecdf9c611a8c6f1c…`), `datasets/airway_mucosal_creasing/paper_citations.bib` (`dc5cda198aa48e19…`), `datasets/airway_mucosal_creasing/scout_manifest.json` (`c73a19386245eae3…`)
- `docs/media/airway_mucosal_creasing_comparison.png` — artifact_sha256 `285fff9751c8f0c9…`, 208,536 B, 1428×1177 px

| measurement | value | reference | method |
|---|---:|---|---|
| Healthy bronchiole circumferential fold mode number m* | 19.74 folds | 14.0 | Analytical bilayer buckling formula m* = (R_0/h_m)*(3*mu_s/mu_m)^(1/3) |
| Chronic bronchitis circumferential fold mode number m* | 12.74 folds | 9.0 | Analytical bilayer buckling formula m* = (R_0/h_m)*(3*mu_s/mu_m)^(1/3) |
| Severe asthma circumferential fold mode number m* | 6.6 folds | 5.0 | Analytical bilayer buckling formula m* = (R_0/h_m)*(3*mu_s/mu_m)^(1/3) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2018-nelson-airway-mucosal-creasing.py` prior to Step 12 closed-loop remediation in `2026-10-19-airway-mucosal-creasing-loop`.

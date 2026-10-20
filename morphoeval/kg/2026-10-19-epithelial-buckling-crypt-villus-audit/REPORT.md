# MorphoEval report — audit `2026-10-19-epithelial-buckling-crypt-villus-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/63

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hannezo et al. (2011) PRL & Shyer et al. (2013) Science sequential smooth-muscle mucosal buckling into ridges, zigzags, and hexagonal villi (`2013-shyer-intestinal-villi-buckling.py`) | 57 | 34 | 48 | 52 | 74 | 68 | 62 | 54.0 | 40.8 | 67.6 | 52.8 | **52.8** | partially faithful |

## Findings and remediation

### Hannezo et al. (2011) PRL & Shyer et al. (2013) Science sequential smooth-muscle mucosal buckling into ridges, zigzags, and hexagonal villi  —  overall 52.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `2013-shyer-intestinal-villi-buckling.py` (overall weighted score: 55.8/100). The v1 script implements the Hannezo-Shyer mucosal bending rigidity B_e = E_e*h_e^3/(12*(1-nu^2)), mesenchymal foundation stiffness K_s = E_s/H_s, and quartic buckling wavelength scaling lambda_* ~ (B_e/K_s)^(1/4) across Chick E6-E16, but relies on a closed-form 2D Fourier ansatz instead of minimizing the Föppl-von Kármán / neo-Hookean mucosal energy functional F[w], omits the 12-point 3-species comparative collapse (chick, quail, mouse) and stage-specific FK506 smooth-muscle ablation / surgical slit controls, and does not segment the vendored histology specimen (`intestinal_villi_crypts_specimen.jpg`).

- **Model fidelity to source: 57** — Computes mucosal bending rigidity B_e = E_e*h_e^3/(12*(1-nu^2)), stromal foundation stiffness K_s = E_s/H_s, and characteristic wavelength lambda_* ~ 2*pi*(B_e/K_s)^(1/4) (L39-53), backed by 3 Lean 4 theorems in `Loop_063_epithelial_buckling_crypt_villus.lean`. However, `generate_parametric_mucosal_surface` (L56-85) uses a hardcoded Fourier superposition rather than minimizing the coupled Föppl-von Kármán + neo-Hookean free-energy functional F[w], and lacks FK506/surgical smooth-muscle ablation controls.  
  evidence: `2013-shyer-intestinal-villi-buckling.py` L39–53; `2013-shyer-intestinal-villi-buckling.py` L56–85
- **Morphological realism / specimen match: 34** — Although `datasets/epithelial_buckling_crypt_villus/intestinal_villi_crypts_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or measures hexagonal villus coordination and bond-orientational order |Psi_6| on the biological histology specimen, and compares only against 6 Chick stages without Quail or Mouse comparative data.  
  evidence: `2013-shyer-intestinal-villi-buckling.py` L88–115; `docs/media/epithelial_buckling_crypt_villus_comparison.png` L0–0
- **Parameter expressivity: 48** — Evaluates 6 embryonic stages of Chick (E6-E16) across smooth, ridge, zigzag, and hexagonal villi regimes (L88-115), but omits the 12-point 3-species morphospace collapse (Chick, Quail, Mouse), omits the 5 FK506/surgical smooth-muscle ablation conditions (E6, E10 slit, E11, E14, WT E16), and exposes no CLI parameter overrides.  
  evidence: `2013-shyer-intestinal-villi-buckling.py` L88–115
- **Morphometric validation: 52** — `measure_morphoeval()` (L181-213) reports only 3 analytic wavelength numbers (Chick E10, E14, E16) without 2D spectral/FFT wavelength extraction from a relaxed energy minimizer, without zigzag chevron angle theta_zz or hexagonal order |Psi_6| measurements, without FK506 negative-control deltas, and without specimen image morphometrics.  
  evidence: `2013-shyer-intestinal-villi-buckling.py` L181–213
- **Source and data traceability: 74** — Vendors `datasets/epithelial_buckling_crypt_villus/` with 4 primary DOIs in `paper_citations.bib` (`10.1126/science.1238842`, `10.1103/PhysRevLett.107.078104`, `10.1103/PhysRevLett.110.024302`, `10.1073/pnas.1205669109`), `scout_manifest.json`, and `epithelial_buckling_crypt_villus_reference.json`, but lacks runtime SHA-256 verification in `2013-shyer-intestinal-villi-buckling.py`.  
  evidence: `datasets/epithelial_buckling_crypt_villus/epithelial_buckling_crypt_villus_reference.json` L1–240
- **Code-to-model traceability: 68** — Links module docstrings to Shyer et al. (2013) Science, Hannezo et al. (2011) PRL, and the 3 Lean 4 theorems in `Loop_063_epithelial_buckling_crypt_villus.lean` (L1-21), but lacks trace links for Föppl-von Kármán energy minimization and FK506 smooth-muscle ablation experiments.  
  evidence: `2013-shyer-intestinal-villi-buckling.py` L1–21
- **Reproducibility package: 62** — Provides `--self-test` (`L216-223`) checking stage count and basic amplitude bounds, but lacks SHA-256 asset verification, two-run bitwise determinism checks on an energy minimizer, and FK506 ablation assertions.  
  evidence: `2013-shyer-intestinal-villi-buckling.py` L216–223

Remediation:
- Replace the closed-form 2D Fourier ansatz with a 2D pseudo-spectral / L-BFGS-B Föppl-von Kármán + neo-Hookean stromal foundation free-energy minimizer F[w] that relaxes mucosal height w(x,y) across smooth, 1D ridge, 2D herringbone zigzag, and hexagonal villus-crypt regimes.
- Implement the 12-point comparative wavelength scaling collapse across Chick (Gallus gallus), Quail (Coturnix japonica), and Mouse (Mus musculus) with Pearson r, p-value, RMSE, and log-log scaling slope.
- Implement Shyer et al. (2013) Figs. 2-4 FK506 smooth-muscle inhibition and surgical muscle-slit rescue controls (FK506 at E6 -> smooth; surgical slit at E10 -> flat; FK506 at E11 -> arrested at ridges; FK506 at E14 -> arrested at zigzags; WT E16 -> hexagonal villi).
- Segment the vendored CC0 intestinal villi histology micrograph (`intestinal_villi_crypts_specimen.jpg`) to extract villus centroids, Delaunay triangulation coordination number Z, hexagonal bond-orientational order |Psi_6|, and inter-villus spacing CV, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hannezo et al. (2011) PRL & Shyer et al. (2013) Science sequential smooth-muscle mucosal buckling into ridges, zigzags, and hexagonal villi
- **spec_fingerprint** (semantic model + inputs): `f6a04e7da51440cd41541428ef80cd7b7e9f5332411d929e37970e56635d68c5`
- **input_dataset_sha256**: `datasets/epithelial_buckling_crypt_villus/epithelial_buckling_crypt_villus_reference.json` (`1887bf2160034e33…`), `datasets/epithelial_buckling_crypt_villus/intestinal_villi_crypts_specimen.jpg` (`30afcb789bbdc60c…`), `datasets/epithelial_buckling_crypt_villus/paper_citations.bib` (`a8bc0251af8be542…`), `datasets/epithelial_buckling_crypt_villus/scout_manifest.json` (`1e6e1bbc98fb4873…`)
- `docs/media/epithelial_buckling_crypt_villus_comparison.png` — artifact_sha256 `6e9b91bbf80b94a3…`, 164,272 B, 1495×1105 px

| measurement | value | reference | method |
|---|---:|---|---|
| Chick E10 longitudinal ridge wavelength lambda_sim | 348.42 um | 118.0 | Quartic plate-on-foundation wavelength 2*pi*(B_e/K_s)^(1/4) at Chick E10 |
| Chick E14 herringbone zigzag wavelength lambda_sim | 272.95 um | 94.0 | Quartic plate-on-foundation wavelength at Chick E14 |
| Chick E16 hexagonal villus spacing lambda_sim | 237.58 um | 82.0 | Quartic plate-on-foundation wavelength at Chick E16 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2013-shyer-intestinal-villi-buckling.py` prior to Step 12 closed-loop remediation in `2026-10-19-epithelial-buckling-crypt-villus-loop`.

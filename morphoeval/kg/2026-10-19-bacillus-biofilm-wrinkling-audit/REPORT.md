# MorphoEval report — audit `2026-10-19-bacillus-biofilm-wrinkling-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/94

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Asally et al. (2012) PNAS & Yan et al. (2019) Nature Physics osmotic matrix swelling, localized cell-death delamination, and Föppl–von Kármán radial wrinkle spokes in Bacillus subtilis biofilms (`2013-asally-bacillus-biofilm-wrinkling.py`) | 58 | 35 | 49 | 52 | 74 | 68 | 63 | 55.0 | 41.4 | 68.0 | 53.5 | **53.5** | partially faithful |

## Findings and remediation

### Asally et al. (2012) PNAS & Yan et al. (2019) Nature Physics osmotic matrix swelling, localized cell-death delamination, and Föppl–von Kármán radial wrinkle spokes in Bacillus subtilis biofilms  —  overall 53.5 (partially faithful)

Pass 1 (v1) baseline evaluation of `2013-asally-bacillus-biofilm-wrinkling.py` (overall weighted score: 56.2/100). The v1 script implements the Föppl–von Kármán bilayer characteristic wavelength lambda_c = 2*pi*h*(E_bar_f/(3*E_bar_s))^(1/3), critical compressive buckling strain epsilon_c = 0.25*(3*E_bar_s/E_bar_f)^(2/3), and radial spoke count N_spokes = 2*pi*R_ring/lambda_c across the 7-point agar stiffness sweep E_s in [0.5, 10.0] kPa, but uses a closed-form polar cosine ansatz instead of relaxing a 2D Föppl–von Kármán + localized cell-death delamination energy functional F[w], omits the biofilm thickness sweep h in [40, 140] um, Delta eps / Delta tasA matrix-deficient mutant and artificial score-guided spoke controls, and does not segment the vendored macrocolony biofilm specimen (`bacillus_biofilm_specimen.jpg`).

- **Model fidelity to source: 58** — Computes plane-strain moduli E_bar_f and E_bar_s, characteristic Föppl–von Kármán wavelength lambda_c = 2*pi*h*(E_bar_f/(3*E_bar_s))^(1/3), critical strain epsilon_c, and radial spoke count N_spokes (L34-53), backed by 3 Lean 4 theorems in `Loop_094_bacillus_biofilm_wrinkling.lean`. However, `generate_parametric_biofilm_surface` (L56-73) uses a hardcoded polar ansatz rather than minimizing a 2D circular Föppl–von Kármán + localized cell-death delamination free-energy functional, and lacks Delta eps/tasA mutant and artificial cell-death score controls.  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L34–53; `2013-asally-bacillus-biofilm-wrinkling.py` L56–73
- **Morphological realism / specimen match: 35** — Although `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or extracts macrocolony count and radial spoke ridge profiles from the biological specimen image.  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L76–94; `docs/media/bacillus_biofilm_wrinkling_comparison.png` L0–0
- **Parameter expressivity: 49** — Evaluates 7 agar stiffnesses E_s in [0.5, 10.0] kPa (L76-94), but omits the 6-point biofilm thickness scaling sweep h in [40, 140] um, omits the 5 genetic and mechanical-score controls (WT, Delta eps/tasA mutant, FeCl3 cell-death suppression, artificial m=6 and m=8 spoke templates), and exposes no CLI parameter overrides.  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L76–94
- **Morphometric validation: 52** — `measure_morphoeval()` (L152-181) reports only 4 analytic numbers (standard, stiff, and soft agar wavelengths plus spoke count) without Pearson r, p-value, RMSE, log-log cube-root scaling slope, thickness scaling correlation, mutant negative-control amplitude delta, or specimen image radial spoke morphometrics.  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L152–181
- **Source and data traceability: 74** — Vendors `datasets/bacillus_biofilm_wrinkling/` with 4 primary DOIs in `paper_citations.bib` (`10.1073/pnas.1212429109`, `10.1038/s41567-018-0392-9`, `10.1073/pnas.1217178110`, `10.1073/pnas.1102300108`), `scout_manifest.json`, and `bacillus_biofilm_wrinkling_reference.json`, but lacks runtime SHA-256 verification in `2013-asally-bacillus-biofilm-wrinkling.py`.  
  evidence: `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_wrinkling_reference.json` L1–210
- **Code-to-model traceability: 68** — Links module docstrings to Asally et al. (2012) PNAS, Yan et al. (2019) Nature Physics, and the 3 Lean 4 theorems in `Loop_094_bacillus_biofilm_wrinkling.lean` (L1-11), but lacks trace links for 2D Föppl–von Kármán energy relaxation, localized cell-death delamination, and artificial score-guided spoke selection.  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L1–11
- **Reproducibility package: 63** — Provides `--self-test` (`L184-190`) checking agar sweep length and monotonicity, but lacks SHA-256 asset verification, two-run bitwise determinism checks on an energy minimizer, and mutant/ablation assertions.  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L184–190

Remediation:
- Replace the closed-form polar ansatz with a 2D circular colony Föppl–von Kármán + localized cell-death delamination L-BFGS-B free-energy minimizer F[w] that relaxes biofilm height w(x,y) into a central concentric ring and outer radial spokes.
- Implement both the 7-point agar stiffness sweep E_s in [0.5, 10.0] kPa (verifying cube-root scaling exponent -1/3) and the 6-point biofilm thickness sweep h in [40, 140] um (verifying linear wavelength and inverse spoke scaling).
- Implement Asally et al. (2012) PNAS Figs. 1-4 genetic and mechanical controls: wild-type NCIB 3610, Delta epsA-O / Delta tasA matrix-deficient flat mutant, FeCl3/D-Leu cell-death-suppressed control, and artificial mechanical-score templates guiding custom spoke counts (m = 6, m = 8).
- Segment the vendored CC BY-SA 3.0 Bacillus subtilis macrocolony specimen (`bacillus_biofilm_specimen.jpg`) to extract isolated macrocolonies, focal colony radial spoke count N_spokes, and angular spoke spacing CV, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Asally et al. (2012) PNAS & Yan et al. (2019) Nature Physics osmotic matrix swelling, localized cell-death delamination, and Föppl–von Kármán radial wrinkle spokes in Bacillus subtilis biofilms
- **spec_fingerprint** (semantic model + inputs): `80d4742b3055639fafb02a0909f223313e723e53cf52d4df159742cd92171fba`
- **input_dataset_sha256**: `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_specimen.jpg` (`0ab531dfc2ded692…`), `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_wrinkling_reference.json` (`7d866d4d5136d397…`), `datasets/bacillus_biofilm_wrinkling/paper_citations.bib` (`ae278450d6851340…`), `datasets/bacillus_biofilm_wrinkling/scout_manifest.json` (`d546628168a39289…`)
- `docs/media/bacillus_biofilm_wrinkling_comparison.png` — artifact_sha256 `9799ba47f741b45e…`, 212,694 B, 1680×1400 px

| measurement | value | reference | method |
|---|---:|---|---|
| Standard 1.5% agar (E_s = 2.5 kPa) Föppl-von Kármán wrinkle wavelength | 750.87 um | 750.0 | Föppl-von Kármán bilayer formula lambda_c = 2*pi*h*(E_bar_f/(3*E_bar_s))^(1/3) *(Baseline analytic wavelength at E_s = 2.5 kPa, h = 80 um)* |
| Standard 1.5% agar (E_s = 2.5 kPa) radial wrinkle spoke count | 13 count | 13 | Circumferential mode selection N_spokes = round(2*pi*R_ring/lambda_c) *(Baseline spoke count around R_ring = 1600 um)* |
| Stiff agar (E_s = 10.0 kPa) wrinkle wavelength | 473.02 um | 472.0 | Föppl-von Kármán bilayer formula at E_s = 10.0 kPa *(Shortened wavelength on stiff agar)* |
| Soft agar (E_s = 0.5 kPa) wrinkle wavelength | 1283.96 um | 1280.0 | Föppl-von Kármán bilayer formula at E_s = 0.5 kPa *(Longer wavelength on compliant soft agar)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2013-asally-bacillus-biofilm-wrinkling.py` prior to Step 12 closed-loop remediation in `2026-10-19-bacillus-biofilm-wrinkling-loop`.

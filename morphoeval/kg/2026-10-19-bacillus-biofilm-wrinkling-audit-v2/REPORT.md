# MorphoEval report — audit `2026-10-19-bacillus-biofilm-wrinkling-audit-v2`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/94 · supersession chain `2026-10-19-bacillus-biofilm-wrinkling-audit -> 2026-10-19-bacillus-biofilm-wrinkling-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Asally et al. (2012) PNAS & Yan et al. (2019) Nature Physics osmotic matrix swelling, localized cell-death delamination, and Föppl–von Kármán radial wrinkle spokes in Bacillus subtilis biofilms (`2013-asally-bacillus-biofilm-wrinkling.py`) | 95 | 92 | 93 | 94 | 95 | 93 | 94 | 94.3 | 92.8 | 94.1 | 93.6 | **93.6** | faithful |

## Findings and remediation

### Asally et al. (2012) PNAS & Yan et al. (2019) Nature Physics osmotic matrix swelling, localized cell-death delamination, and Föppl–von Kármán radial wrinkle spokes in Bacillus subtilis biofilms  —  overall 93.6 (faithful)

Pass 2 (v2) evaluation of `2013-asally-bacillus-biofilm-wrinkling.py` (overall weighted score: 93.6/100, up from 56.2/100 in v1). The remediated script relaxes a 2D circular colony Föppl–von Kármán + localized cell-death delamination free-energy functional via pseudo-spectral L-BFGS-B minimization, reproducing the wild-type central concentric ring and N_spokes = 13 outer radial spokes; validates the 7-point agar stiffness sweep E_s in [0.5, 10.0] kPa (r = 1.0000, RMSE = 1.94 um, log-log exponent -0.3338 ± 0.0008 vs -1/3) and 6-point biofilm thickness sweep h in [40, 140] um (r = 1.0000, slope = 0.9999 ± 0.0017); verifies all 5 genetic and artificial mechanical-score controls (WT, Delta eps/tasA flat mutant, FeCl3 cell-death suppression, guided m=6 and m=8 spoke templates); and segments the SHA-256-locked CC BY-SA 3.0 specimen (`bacillus_biofilm_specimen.jpg`, n = 11 colonies, focal N_spokes = 12, mean spoke angle = 30.0 deg, spacing CV = 0.3156).

- **Model fidelity to source: 95** — Implements plane-strain Föppl–von Kármán bilayer buckling mechanics (`compute_fvk_wavelength_and_strain`, L91-147), spatiotemporal basal cell-death patterning (`build_cell_death_template`, L153-175), and a 2D pseudo-spectral + L-BFGS-B energy minimizer (`relax_fvk_biofilm_colony`, L178-297) with genetic and artificial score controls (`evaluate_genetic_and_mechanical_controls`, L393-425), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_094_bacillus_biofilm_wrinkling.lean`.  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L91–147; `2013-asally-bacillus-biofilm-wrinkling.py` L178–297; `morphoeval/lean/loops/Loop_094_bacillus_biofilm_wrinkling.lean` L14–108
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC BY-SA 3.0 Bacillus subtilis macrocolony biofilm photograph `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_specimen.jpg`, segments n = 11 isolated circular macrocolonies across the agar plate, and extracts N_spokes = 12 radial wrinkle ridges (mean angular spacing = 30.0 deg, CV = 0.3156) on the focal colony (`segment_bacillus_biofilm_specimen`, L431-505).  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L431–505; `docs/media/bacillus_biofilm_wrinkling_comparison.png` L0–0
- **Parameter expressivity: 93** — Spans the 7-point agar stiffness sweep E_s in [0.5, 10.0] kPa (L303-355), the 6-point biofilm thickness sweep h in [40, 140] um (L358-390), 5 genetic and mechanical-score controls (WT, Delta eps/tasA, FeCl3 cell-death suppression, guided m=6 and m=8 templates, L393-425), and exposes `--E-s-kPa`, `--h-um`, `--epsilon-g`, `--artificial-spokes`, and `--seed` CLI overrides (L652-676).  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L303–425; `2013-asally-bacillus-biofilm-wrinkling.py` L652–676
- **Morphometric validation: 94** — `measure_morphoeval()` (L511-606) computes 9 reference-anchored morphometric measurements with uncertainty and statistical tests: agar stiffness wavelength Pearson r = 1.0000 (p = 3.74e-12, RMSE = 1.94 um), log-log agar stiffness exponent |alpha_E| = 0.3338 ± 0.0008 (ref 0.3333), thickness wavelength Pearson r = 1.0000 (p = 4.57e-11, slope = 0.9999 ± 0.0017), relaxed FvK standard-agar wavelength 773.32 um (ref 750.0 um), WT spoke count 13 (ref 13), Delta eps/tasA mutant amplitude 0.0 um (ref 0.0 um), guided m=6 score spoke count 6 (ref 6), specimen focal spoke count 12 (ref 13), and specimen mean spoke angle 30.0 deg (ref 27.69 deg).  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L511–606
- **Source and data traceability: 95** — Vendors `datasets/bacillus_biofilm_wrinkling/` with 4 primary DOIs in `paper_citations.bib` (`10.1073/pnas.1212429109`, `10.1038/s41567-018-0392-9`, `10.1073/pnas.1217178110`, `10.1073/pnas.1102300108`), `scout_manifest.json`, `bacillus_biofilm_wrinkling_reference.json`, and `bacillus_biofilm_specimen.jpg`, all verified at runtime via SHA-256 checks (`verify_dataset_integrity`, L69-85).  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L69–85; `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_wrinkling_reference.json` L1–210
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation and control to Asally et al. (2012) PNAS Figs. 1-4, Yan et al. (2019) Nature Physics Eqs. (1)-(3), Trejo et al. (2013) PNAS, Seminara et al. (2012) PNAS, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_094_bacillus_biofilm_wrinkling.lean` (L1-35, L91-215).  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L1–35
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L612-649`) verifying SHA-256 dataset integrity, agar stiffness cube-root scaling slope (-1/3), biofilm thickness linear scaling, WT vs Delta eps/tasA and FeCl3 cell-death controls, artificial m=6 and m=8 spoke guidance, specimen colony & spoke segmentation, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2013-asally-bacillus-biofilm-wrinkling.py` L612–649

Remediation:
- Couple the 2D circular Föppl–von Kármán delamination solver to a 3D poroelastic osmotic water-transport PDE resolving radial fluid influx from the agar into the expanding biofilm channel network.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Asally et al. (2012) PNAS & Yan et al. (2019) Nature Physics osmotic matrix swelling, localized cell-death delamination, and Föppl–von Kármán radial wrinkle spokes in Bacillus subtilis biofilms
- **spec_fingerprint** (semantic model + inputs): `80d4742b3055639fafb02a0909f223313e723e53cf52d4df159742cd92171fba`
- **input_dataset_sha256**: `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_specimen.jpg` (`0ab531dfc2ded692…`), `datasets/bacillus_biofilm_wrinkling/bacillus_biofilm_wrinkling_reference.json` (`7d866d4d5136d397…`), `datasets/bacillus_biofilm_wrinkling/paper_citations.bib` (`ae278450d6851340…`), `datasets/bacillus_biofilm_wrinkling/scout_manifest.json` (`d546628168a39289…`)
- `docs/media/bacillus_biofilm_wrinkling_comparison.png` — artifact_sha256 `68b793a887248a10…`, 692,871 B, 2700×1650 px

| measurement | value | reference | method |
|---|---:|---|---|
| Agar stiffness sweep (0.5-10 kPa) wrinkle wavelength correlation | 1.0 Pearson r | 1.0 | Pearson correlation between Föppl-von Kármán lambda_c and empirical wavelength across 7 agar stiffnesses (p=3.74e-12, RMSE=1.94 um) *(RMSE = 1.94 um, p = 3.74e-12)* |
| Log-log agar stiffness wavelength scaling exponent alpha_E (lambda_c ~ E_s^alpha_E) | 0.3338 |exponent| | 0.3333 | Log-log OLS regression slope |d log(lambda)/d log(E_s)| across E_s in [0.5, 10.0] kPa (SE ± 0.0008) *(Signed slope = -0.3338 ± 0.0008)* |
| Biofilm thickness scaling (40-140 um) wavelength correlation | 1.0 Pearson r | 1.0 | Linear wavelength scaling lambda_c ~ h^1.0 across 6 biofilm thicknesses (slope=0.9999, p=4.57e-11) *(Log-log thickness slope = 0.9999 ± 0.0017)* |
| Standard 1.5% agar (E_s = 2.5 kPa) relaxed FvK wrinkle wavelength | 773.32 um | 750.0 | 2D L-BFGS-B Föppl-von Kármán + cell-death delamination relaxation circumferential wavelength at R_ring = 1600 um *(Analytic lambda_c = 750.87 um)* |
| Wild-type NCIB 3610 radial spoke count N_spokes on standard agar | 13 count | 13 | Circumferential peak detection on 2D FvK relaxed height field w(x,y) at r = 1.55 * R_ring *(Peak amplitude w_max = 162.0 um)* |
| Matrix-deficient mutant (Delta epsA-O / Delta tasA) wrinkle amplitude | 0.0 um | 0.0 | Negative control with osmotic swelling strain epsilon_g = 0.015 < epsilon_c_eff *(Confirms zero wrinkling when matrix swelling strain is below critical threshold)* |
| Artificial cell-death / mechanical score pattern (m = 6) guided spoke count | 6 count | 6 | 2D FvK relaxation on artificial 6-spoke basal cell-death delamination template (Asally et al. 2012 Fig. 4) *(Guided peak amplitude = 168.0 um)* |
| Specimen focal B. subtilis macrocolony radial wrinkle spoke count | 12 count | 13 | DoG ridge filtering (sigma=1.0, 4.5 px) and circumferential peak detection on bacillus_biofilm_specimen.jpg focal colony *(n = 11 isolated macrocolonies segmented on agar plate)* |
| Specimen focal macrocolony mean angular spoke spacing | 30.0 deg | 27.69 | Mean angular separation between adjacent radial wrinkle ridges in focal specimen macrocolony *(Angular spacing CV = 0.3156)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2013-asally-bacillus-biofilm-wrinkling.py` after completing Step 12 closed-loop remediation in `2026-10-19-bacillus-biofilm-wrinkling-loop`.

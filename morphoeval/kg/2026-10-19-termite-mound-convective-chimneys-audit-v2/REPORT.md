# MorphoEval report — audit `2026-10-19-termite-mound-convective-chimneys-audit-v2`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/97 · supersession chain `2026-10-19-termite-mound-convective-chimneys-audit -> 2026-10-19-termite-mound-convective-chimneys-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| King, Ocko & Mahadevan (2015) PNAS, Ocko et al. (2017) PNAS & Ocko, Heyde & Mahadevan (2019) PNAS solar-powered diurnal thermosiphon ventilation and convective pheromone morphogenesis of cathedral termite mounds (`2015-ocko-termite-mound-convective-chimneys.py`) | 95 | 94 | 94 | 94 | 95 | 94 | 94 | 94.7 | 94.0 | 94.3 | 94.3 | **94.3** | faithful |

## Findings and remediation

### King, Ocko & Mahadevan (2015) PNAS, Ocko et al. (2017) PNAS & Ocko, Heyde & Mahadevan (2019) PNAS solar-powered diurnal thermosiphon ventilation and convective pheromone morphogenesis of cathedral termite mounds  —  overall 94.3 (faithful)

Pass 2 (v2) evaluation of `2015-ocko-termite-mound-convective-chimneys.py` (overall weighted score: 94.3/100, up from 56.8/100 in v1). The remediated script solves the coupled 24-hour diurnal solar-thermal buoyancy thermosiphon & Taylor–Aris oscillatory CO₂ ventilation cycle (flute velocity Pearson r = 1.0000, RMSE = 0.0060 cm/s; chimney CO₂ Pearson r = 0.9921, RMSE = 0.0869% CO₂; exact mass continuity A_c u_c + A_f u_f = 0); integrates the Ocko, Heyde & Mahadevan (2019) PNAS moving-boundary convective pheromone mound morphogenesis solver across the 7-point Péclet sweep Pe in [0.0, 4.2] (H/W Pearson r = 1.0000, RMSE = 0.0046) and 6-point Southern Hemisphere latitude equatorward spire tilt sweep |phi_lat| in [0.0, 24.0] deg S (Pearson r = 1.0000, RMSE = 0.1038 deg, Namibia 19.2 deg S -> 18.82 deg N tilt); evaluates *Macrotermes michaelseni*, *Odontotermes obesus*, and shaded/isothermal dome controls (Pe = 0 -> H/W = 0.4989); and segments the SHA-256-locked CC BY-SA 4.0 cathedral termite mound specimen (`macrotermes_cathedral_mound_specimen.jpg`, H/W_base = 1.9494, taper half-angle 9.46 deg, 8 visible front-hemisphere flutes = 16 full-perimeter flutes).

- **Model fidelity to source: 95** — Implements the coupled 24-hour Boussinesq Darcy–Brinkman thermosiphon velocity & Taylor–Aris oscillatory CO₂ clearance solver (`simulate_diurnal_thermosiphon_cycle`, L91-209) and the Ocko, Heyde & Mahadevan (2019) PNAS moving-boundary convective pheromone mound morphogenesis & equatorward solar-tracking tilt solver (`simulate_convective_mound_morphogenesis`, L270-370), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_097_termite_mound_convective_chimneys.lean`.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L91–209; `2015-ocko-termite-mound-convective-chimneys.py` L270–370; `morphoeval/lean/loops/Loop_097_termite_mound_convective_chimneys.lean` L16–126
- **Morphological realism / specimen match: 94** — Vendors and SHA-256-verifies the CC BY-SA 4.0 cathedral termite mound photograph `datasets/termite_mound_convective_chimneys/macrotermes_cathedral_mound_specimen.jpg`, segments the earthen mound silhouette (`H = 347 px`, `W_base = 178.0 px`, `H/W_base = 1.9494` vs wild-type *Macrotermes* `H/W = 1.95`), extracts spire taper half-angle `9.46 deg`, and detects `8` prominent sunlit vertical flute ridges across the front hemisphere corresponding to `16` full-perimeter flutes (`segment_cathedral_mound_specimen`, L502-585).  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L502–585; `docs/media/termite_mound_convective_chimneys_comparison.png` L0–0
- **Parameter expressivity: 94** — Spans the 9-point 24-hour diurnal thermosiphon trajectory (L212-263), the 7-point convective Péclet morphogenesis sweep `Pe in [0.0, 4.2]` (L373-419), the 6-point geographic latitude equatorward spire tilt sweep `|phi_lat| in [0.0, 24.0] deg S` (L422-450), 3 species & ablation controls (*Macrotermes michaelseni*, *Odontotermes obesus*, and shaded isothermal dome `Pe = 0`, L453-497), and exposes `--peclet`, `--delta-t-day`, `--latitude-deg`, and `--seed` CLI overrides (L636-664).  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L212–497; `2015-ocko-termite-mound-convective-chimneys.py` L636–664
- **Morphometric validation: 94** — `measure_morphoeval()` (L566-630) computes 9 reference-anchored morphometric measurements with p-values and RMSE: 24h flute velocity Pearson r = 1.0000 (p = 1.02e-18, RMSE = 0.0060 cm/s), 24h CO₂ clearance Pearson r = 0.9921 (p = 1.44e-7, RMSE = 0.0869%), 7-point Péclet H/W Pearson r = 1.0000 (p = 1.45e-14, RMSE = 0.0046), 6-point latitude spire tilt Pearson r = 1.0000 (p = 1.00e-9, RMSE = 0.1038 deg), WT *M. michaelseni* H/W = 1.9427 (ref 1.95), isothermal dome negative control H/W = 0.4989 (ref 0.50), specimen H/W_base = 1.9494 (ref 1.95), specimen spire taper half-angle 9.46 deg (ref 11.8 deg), and specimen full-perimeter flute count 16 (ref 16).  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L566–630
- **Source and data traceability: 95** — Vendors `datasets/termite_mound_convective_chimneys/` with 5 primary DOIs in `paper_citations.bib` (`10.1073/pnas.1713534114`, `10.1073/pnas.1818759116`, `10.1073/pnas.1423242112`, `10.1007/s00114-002-0401-4`, `10.1016/j.csbj.2020.08.012`), `scout_manifest.json`, `termite_mound_convective_chimneys_reference.json`, and `macrotermes_cathedral_mound_specimen.jpg`, all verified at runtime via SHA-256 checks (`verify_dataset_integrity`, L66-85).  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L66–85; `datasets/termite_mound_convective_chimneys/termite_mound_convective_chimneys_reference.json` L1–145
- **Code-to-model traceability: 94** — Module and function docstrings map every governing equation and control to King, Ocko & Mahadevan (2015) PNAS, Ocko et al. (2017) PNAS, Ocko, Heyde & Mahadevan (2019) PNAS, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_097_termite_mound_convective_chimneys.lean` (L1-35, L91-370).  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L1–35
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L633-686`) verifying SHA-256 dataset integrity, thermosiphon flow reversal & mass continuity, Péclet spire elongation, Taylor–Aris dispersion enhancement, 24h diurnal velocity & CO₂ trajectories, Péclet & latitude tilt sweeps, cathedral mound CV segmentation, and two-run bitwise determinism.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L633–686

Remediation:
- Couple the 2D/3D convective pheromone mound morphogenesis solver to micro-CT porous wall drainage permeability measurements across seasonal rainfall regimes.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### King, Ocko & Mahadevan (2015) PNAS, Ocko et al. (2017) PNAS & Ocko, Heyde & Mahadevan (2019) PNAS solar-powered diurnal thermosiphon ventilation and convective pheromone morphogenesis of cathedral termite mounds
- **spec_fingerprint** (semantic model + inputs): `d09fb37c4086797405dbf184a42b7af11e3ae9291d9bd6946a94bdc2be0204df`
- **input_dataset_sha256**: `datasets/termite_mound_convective_chimneys/macrotermes_cathedral_mound_specimen.jpg` (`71ab909aae615390…`), `datasets/termite_mound_convective_chimneys/paper_citations.bib` (`21248770fa6f59e6…`), `datasets/termite_mound_convective_chimneys/scout_manifest.json` (`cdefb06d8e025efd…`), `datasets/termite_mound_convective_chimneys/termite_mound_convective_chimneys_reference.json` (`36184ab6989fedbf…`)
- `docs/media/termite_mound_convective_chimneys_comparison.png` — artifact_sha256 `d49033b52ede8d1f…`, 964,600 B, 2775×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| Diurnal 24-hour thermosiphon flute velocity u_flute(t) correlation | 1.0 Pearson r | 1.0 | Pearson correlation between simulated Boussinesq thermosiphon flute velocity and empirical 24h trajectory across 9 diurnal timepoints (p=1.02e-18, RMSE=0.0060 cm/s) *(RMSE = 0.0060 cm/s, mass continuity residual = 0.00e+00)* |
| Diurnal 24-hour oscillatory CO2 clearance trajectory correlation | 0.9921 Pearson r | 1.0 | Pearson correlation between Taylor-Aris oscillatory ventilation CO2 trajectory and empirical chimney CO2 across 24 hours (p=1.44e-07, RMSE=0.0869%) *(RMSE = 0.0869% CO2)* |
| Convective Peclet morphogenesis sweep (Pe = 0.0..4.2) aspect ratio H/W correlation | 1.0 Pearson r | 1.0 | Pearson correlation between moving-boundary pheromone morphogenesis H/W(Pe) and empirical reference across 7 Peclet levels (p=1.45e-14, RMSE=0.0046) *(RMSE = 0.0046)* |
| Equatorward solar-tracking spire tilt vs latitude (|phi| = 0..24 deg S) correlation | 1.0 Pearson r | 1.0 | Pearson correlation between simulated Northward spire tilt angle and field observations across 6 Southern Hemisphere latitudes (p=1.00e-09, RMSE=0.1038 deg) *(Namibia 19.2 deg S -> 18.82 deg N tilt)* |
| Wild-type Macrotermes michaelseni (Pe = 3.2) cathedral spire aspect ratio H/W | 1.9427 ratio | 1.95 | Moving-boundary convective pheromone morphogenesis solver at wild-type Pe = 3.2 *(H/R = 3.90 conical cathedral spire)* |
| Shaded / isothermal no-buoyancy negative control (Pe = 0.0, dT = 0 K) aspect ratio H/W | 0.4989 ratio | 0.5 | Isothermal pure-diffusion negative control (dT = 0 K, u_flute = 0 cm/s, Pe = 0.0) *(Stagnant nest CO2 = 4.82% vs 2.15% in WT)* |
| Specimen cathedral termite mound segmented aspect ratio H/W_base | 1.9494 ratio | 1.95 | Terracotta chromaticity silhouette segmentation on macrotermes_cathedral_mound_specimen.jpg (H=347 px, W_base=178.0 px) *(Mean-width aspect ratio H/W_mean = 2.384)* |
| Specimen cathedral termite mound spire taper half-angle | 9.46 degrees | 11.8 | Linear regression of upper-75% half-width vs height on segmented cathedral mound silhouette *(Spire centerline tilt = 4.31 deg)* |
| Specimen cathedral termite mound estimated full-perimeter surface flute count | 16 flutes | 16 | Gaussian-detrended horizontal intensity peak detection across mid-mound band (8 visible front-hemisphere ridges x 2) *(8 prominent sunlit flute buttresses segmented on visible face)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2015-ocko-termite-mound-convective-chimneys.py` after completing Step 12 closed-loop remediation in `2026-10-19-termite-mound-convective-chimneys-loop`.

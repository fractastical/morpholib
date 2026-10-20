# MorphoEval report — audit `2026-10-19-termite-mound-convective-chimneys-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/97

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| King, Ocko & Mahadevan (2015) PNAS, Ocko et al. (2017) PNAS & Ocko, Heyde & Mahadevan (2019) PNAS solar-powered diurnal thermosiphon ventilation and convective pheromone morphogenesis of cathedral termite mounds (`2015-ocko-termite-mound-convective-chimneys.py`) | 58 | 35 | 51 | 53 | 74 | 68 | 64 | 55.7 | 41.8 | 68.4 | 53.9 | **53.9** | partially faithful |

## Findings and remediation

### King, Ocko & Mahadevan (2015) PNAS, Ocko et al. (2017) PNAS & Ocko, Heyde & Mahadevan (2019) PNAS solar-powered diurnal thermosiphon ventilation and convective pheromone morphogenesis of cathedral termite mounds  —  overall 53.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2015-ocko-termite-mound-convective-chimneys.py` (overall weighted score: 56.8/100). The v1 script implements algebraic formulas for diurnal thermosiphon flow reversal (`u_flute = K_buoy * dT`, `u_chimney = -(A_f/A_c) * u_flute`), convective Péclet aspect ratio scaling (`H/R = 1 + eta_Pe * Pe`), and Taylor–Aris CO₂ dispersion (`D_eff = D_0 + alpha_TA * u_flute^2`), but lacks a coupled 24-hour diurnal thermal-buoyancy PDE solver with nest CO₂ mass-balance clearance, lacks the Ocko et al. (2019) PNAS moving-boundary pheromone advection-diffusion-deposition solver across `Pe` and solar latitude tilt `phi_lat`, omits the *Macrotermes michaelseni* vs *Odontotermes obesus* and shaded/isothermal dome controls, and never loads or segments the vendored cathedral termite mound specimen (`macrotermes_cathedral_mound_specimen.jpg`).

- **Model fidelity to source: 58** — Implements algebraic formulas for Boussinesq thermosiphon velocity, convective Péclet aspect ratio scaling, and Taylor–Aris CO₂ dispersion (L34-74), backed by 3 Lean 4 theorems in `Loop_097_termite_mound_convective_chimneys.lean`. However, it does not integrate the coupled 24-hour diurnal solar-thermal ODE/PDE for flute-chimney temperature and nest CO₂ clearance or the Ocko et al. (2019) moving-boundary pheromone advection-diffusion-deposition solver.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L34–74; `2015-ocko-termite-mound-convective-chimneys.py` L77–96
- **Morphological realism / specimen match: 35** — Although `datasets/termite_mound_convective_chimneys/macrotermes_cathedral_mound_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or extracts mound silhouette aspect ratio `H/W`, spire taper half-angle, or vertical surface flute ridge count from the biological specimen photograph.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L77–155; `docs/media/termite_mound_convective_chimneys_comparison.png` L0–0
- **Parameter expressivity: 51** — Evaluates the 9-point diurnal temperature table (L77-96), but omits the 7-point convective Péclet morphogenesis sweep `Pe in [0.0, 4.2]`, omits the 6-point geographic latitude equatorward spire tilt sweep `|phi_lat| in [0.0, 24.0] deg S`, omits the *Macrotermes* vs *Odontotermes* vs shaded isothermal dome controls, and exposes no CLI parameter overrides.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L77–96
- **Morphometric validation: 53** — `measure_morphoeval()` (L188-200) reports only 4 scalar values without Pearson r, p-value, RMSE, diurnal velocity and CO₂ correlations, convective Péclet `H/W` correlation, equatorward latitude tilt correlation, or empirical specimen CV silhouette metrics.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L188–200
- **Source and data traceability: 74** — Vendors `datasets/termite_mound_convective_chimneys/` with 5 primary DOIs in `paper_citations.bib` (`10.1073/pnas.1713534114`, `10.1073/pnas.1818759116`, `10.1073/pnas.1423242112`, `10.1007/s00114-002-0401-4`, `10.1016/j.csbj.2020.08.012`), `scout_manifest.json`, and `termite_mound_convective_chimneys_reference.json`, but lacks runtime SHA-256 verification in `2015-ocko-termite-mound-convective-chimneys.py`.  
  evidence: `datasets/termite_mound_convective_chimneys/termite_mound_convective_chimneys_reference.json` L1–145
- **Code-to-model traceability: 68** — Links module docstrings to King et al. (2015), Ocko et al. (2017, 2019), and the 3 Lean 4 theorems in `Loop_097_termite_mound_convective_chimneys.lean` (L1-14), but lacks trace links for moving-boundary pheromone morphogenesis, equatorward solar spire tilt, and CV flute segmentation.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L1–14
- **Reproducibility package: 64** — Provides `--self-test` (`L203-215`) checking basic velocity sign and Péclet ratio, but lacks SHA-256 asset verification, PDE mass-conservation assertions, latitude tilt checks, and two-run bitwise determinism verification.  
  evidence: `2015-ocko-termite-mound-convective-chimneys.py` L203–215

Remediation:
- Implement a coupled 24-hour diurnal solar-thermal buoyancy thermosiphon & nest CO2 clearance solver resolving daytime East/North/West flute updrafts, central chimney downdrafts, nighttime flow reversal, and Taylor-Aris oscillatory CO2 ventilation.
- Implement the Ocko, Heyde & Mahadevan (2019) PNAS moving-boundary mound morphogenesis solver coupling upward convective pheromone advection-diffusion (Pe = u_0 H_0 / D_c) to threshold soil deposition across the 7-point Peclet sweep Pe in [0.0, 4.2] and the 6-point geographic latitude equatorward spire tilt sweep |phi_lat| in [0.0, 24.0] deg S.
- Implement the species & ablation controls comparing Macrotermes michaelseni (Namibia cathedral spire, Pe = 3.2), Odontotermes obesus (India fluted mound, Pe = 2.3), and the shaded/isothermal no-buoyancy dome negative control (Delta T = 0 K, Pe = 0.0 -> H/W = 0.50).
- Segment the vendored CC BY-SA 4.0 cathedral termite mound photograph (`macrotermes_cathedral_mound_specimen.jpg`) to extract mound silhouette aspect ratio H/W, spire taper half-angle, equatorward tilt angle, and vertical surface flute ridge count, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### King, Ocko & Mahadevan (2015) PNAS, Ocko et al. (2017) PNAS & Ocko, Heyde & Mahadevan (2019) PNAS solar-powered diurnal thermosiphon ventilation and convective pheromone morphogenesis of cathedral termite mounds
- **spec_fingerprint** (semantic model + inputs): `d09fb37c4086797405dbf184a42b7af11e3ae9291d9bd6946a94bdc2be0204df`
- **input_dataset_sha256**: `datasets/termite_mound_convective_chimneys/macrotermes_cathedral_mound_specimen.jpg` (`71ab909aae615390…`), `datasets/termite_mound_convective_chimneys/paper_citations.bib` (`21248770fa6f59e6…`), `datasets/termite_mound_convective_chimneys/scout_manifest.json` (`cdefb06d8e025efd…`), `datasets/termite_mound_convective_chimneys/termite_mound_convective_chimneys_reference.json` (`36184ab6989fedbf…`)
- `docs/media/termite_mound_convective_chimneys_comparison.png` — artifact_sha256 `a6502606cb4e064b…`, 212,741 B, 1680×1190 px

| measurement | value | reference | method |
|---|---:|---|---|
| Midday peak exterior flute updraft velocity u_flute (dT = +6.8 K) | 2.856 cm/s | 2.86 | Analytic Boussinesq Darcy-Brinkman buoyancy velocity u_flute = K_buoy * dT at midday dT = 6.8 K *(Baseline algebraic scaling without thermal inertia lag)* |
| Midday peak central chimney downdraft velocity u_chimney (dT = +6.8 K) | -3.57 cm/s | -3.57 | Mass-conserving return velocity u_chimney = -(A_f/A_c) * u_flute at A_f/A_c = 1.25 *(Satisfies A_c u_c + A_f u_f = 0)* |
| Wild-type Macrotermes cathedral mound aspect ratio H/W (Pe = 3.2) | 1.9496 ratio | 1.95 | Analytic convective Peclet aspect ratio H/W = (1 + eta_Pe * Pe) / 2 at Pe = 3.2 *(H/R = 3.90 conical spire)* |
| Isothermal pure-diffusion dome aspect ratio H/W (Pe = 0.0) | 0.5 ratio | 0.5 | Isothermal baseline aspect ratio H/W at Pe = 0.0 (H/R = 1.0 hemispherical dome) *(Hemispherical dome H = R)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2015-ocko-termite-mound-convective-chimneys.py` prior to Step 12 closed-loop remediation in `2026-10-19-termite-mound-convective-chimneys-loop`.

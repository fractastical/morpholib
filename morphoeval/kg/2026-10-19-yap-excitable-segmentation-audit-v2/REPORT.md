# MorphoEval report — audit `2026-10-19-yap-excitable-segmentation-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-yap-excitable-segmentation-audit -> 2026-10-19-yap-excitable-segmentation-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hubaud, Regev, Mahadevan & Pourquié (2017) excitable FitzHugh-Nagumo segmentation clock gated by YAP mechanotransduction (`2016-hubaud-yap-excitable-segmentation.py`) | 94 | 89 | 92 | 92 | 93 | 91 | 91 | 93.3 | 90.1 | 91.7 | 91.5 | **91.5** | faithful |

## Findings and remediation

### Hubaud, Regev, Mahadevan & Pourquié (2017) excitable FitzHugh-Nagumo segmentation clock gated by YAP mechanotransduction  —  overall 91.5 (faithful)

Pass 2 post-remediation evaluation of `2016-hubaud-yap-excitable-segmentation.py` (91.7/100, up from 60.6/100 in Pass 1; +31.1 overall). The remediated script implements the full FitzHugh-Nagumo fast-slow excitable segmentation-clock model gated by nuclear YAP mechanotransduction (`I_eff = I_basal + J_Dll1 - gamma_YAP * Y_nuc`), solves exact cubic nullcline intersections and supercritical Hopf bifurcation thresholds (`|u_H| = 0.9675`, `Y_nuc_crit = 0.6109`, cubic knee barrier `4/3 = 1.3333`), reproduces all 6 experimental perturbation regimes from Hubaud et al. (2017) (isolated fibronectin quiescent, Dll1-coated, Latrunculin A F-actin inhibition rescue at `T_osc = 141.50 min`, Y-27632 Rockout rescue, intact high-density mPSM, and constitutively active `Yap5SA` arrest; oscillating fraction Pearson `r = 0.9976`, `RMSE = 0.0469`), demonstrates suprathreshold single-pulse excitability, solves the 2D reaction-diffusion mPSM target-wave PDE (`wave_contrast = 0.931` vs `0.0144` under `Yap5SA` block), and segments the vendored vertebrate somite/PSM SEM specimen (`mouse_psm_somites_specimen.jpg`, bilateral spacing `47.14 px`, `CV = 0.2010`), backed by 3 machine-checked Lean 4 theorems in `Loop_027_yap_excitable_segmentation.lean`.

- **Model fidelity to source: 94** — Faithfully implements Hubaud et al.'s (2017) FitzHugh-Nagumo cubic fast-slow ODE/PDE `tau_u du/dt = u - u^3/3 - v + I_eff(Y_nuc, J_Dll1) + D_u nabla^2 u`, `tau_v dv/dt = u + a - b v` with exact cubic root continuation, Jacobian trace/determinant Hopf stability analysis (`|u_H| = sqrt(1 - epsilon*b) = 0.9675`), suprathreshold single-spike excitability, and 2D diffusive target-wave propagation (L120-520), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_027_yap_excitable_segmentation.lean`.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L120–235; `2016-hubaud-yap-excitable-segmentation.py` L240–520; `morphoeval/lean/loops/Loop_027_yap_excitable_segmentation.lean` L22–98
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal vertebrate embryo somite/PSM SEM (`datasets/yap_excitable_segmentation/mouse_psm_somites_specimen.jpg`) and segments both bilateral paraxial somite rows (`segment_mouse_psm_somites_specimen`, L525-595), extracting 8 upper and 8 lower somites, bilateral mean spacing 47.14 px (CV = 0.2010), symmetry ratio 0.9969, and phase-aligned FHN wave correlation (Panel F).  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L525–595; `docs/media/yap_excitable_segmentation_comparison.png` L0–0
- **Parameter expressivity: 92** — Evaluates a 40-point nuclear YAP bifurcation continuation across `Y_nuc in [0, 1]`, all 6 experimental regimes from Hubaud et al. (2017) Figs. 1-6 (spread fibronectin, Dll1-coated, Latrunculin A, Y-27632 Rockout, intact high-density mPSM, and `Yap5SA` constitutive block), subthreshold vs suprathreshold single-pulse excitability on quiescent cells, and 2D intact vs `Yap5SA` PDE wave propagation, plus CLI flags (`--Y-nuc`, `--J-Dll1`, `--seed`).  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L330–430; `2016-hubaud-yap-excitable-segmentation.py` L435–520
- **Morphometric validation: 92** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L600-665): FHN supercritical Hopf threshold `|u_H| = 0.9675` (cubic knee barrier `4/3 = 1.3333`), critical nuclear YAP threshold `Y_nuc_crit = 0.6109`, Latrunculin A-rescued autonomous oscillation period `T_osc = 141.50 min` vs empirical `142.0 min`, 6-regime oscillating cell fraction Pearson `r = 0.9976` (`p = 8.43e-06`, `RMSE = 0.0469`), and bilateral SEM specimen somite spacing `47.14 +/- 9.48 px` (`CV = 0.2010`).  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L600–665
- **Source and data traceability: 93** — Vendors `datasets/yap_excitable_segmentation/` with `scout_manifest.json`, `yap_excitable_segmentation_reference.json`, `mouse_psm_somites_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1016/j.cell.2017.08.043`, `10.1016/S0006-3495(61)86902-6`, `10.1038/nature11804`, `10.1038/nature10137`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L85-115).  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L85–115; `datasets/yap_excitable_segmentation/yap_excitable_segmentation_reference.json` L1–145
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered figures in Hubaud et al. (2017) Cell Figs. 1-7 and the 3 Lean 4 theorems in `Loop_027_yap_excitable_segmentation.lean` and `morphoeval/math_kg_loops/hubaud-yap-excitable-segmentation.yaml`.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L1–60; `morphoeval/math_kg_loops/hubaud-yap-excitable-segmentation.yaml` L1–102
- **Reproducibility package: 91** — Deterministic RNG (`seed=2016`), standalone `--self-test` suite verifying SHA-256 dataset integrity, cubic nullcline knees (`Delta_f = 4/3`), Hopf threshold (`|u_H| = 0.9675`), Latrunculin A period (`141.50 min`), `Yap5SA` arrest, single-pulse excitability, bitwise 2D PDE determinism (`atol=0.0`), and specimen segmentation (`run_self_test`, L885-950), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2016-hubaud-yap-excitable-segmentation.py` L885–950; `docs/media/yap_excitable_segmentation_growth.gif` L0–0

Remediation:
- Couple the 2-variable FitzHugh-Nagumo excitable core to an explicit spatial FGF/Wnt gradient decay along the anteroposterior axis to model wavefront arrest of excitable waves into discrete somite clefts.
- Incorporate explicit substrate-stiffness traction mechanics E_sub -> F-actin -> LATS1/2 -> Y_nuc phosphorylation kinetics (Dupont et al. 2011) alongside intercellular Dll1-Notch phase coupling.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hubaud, Regev, Mahadevan & Pourquié (2017) excitable FitzHugh-Nagumo segmentation clock gated by YAP mechanotransduction
- **spec_fingerprint** (semantic model + inputs): `cd7e22124efc1260387a25c216bb931744f6e84e2199a778a5e3eb6d86c4980d`
- **input_dataset_sha256**: `datasets/yap_excitable_segmentation/mouse_psm_somites_specimen.jpg` (`8b80971796cb29e2…`), `datasets/yap_excitable_segmentation/scout_manifest.json` (`852a6ed7852f291d…`), `datasets/yap_excitable_segmentation/yap_excitable_segmentation_reference.json` (`629b1b5e9d80a634…`)
- `docs/media/yap_excitable_segmentation_comparison.png` — artifact_sha256 `a3efccf96655e778…`, 1,162,113 B, 2775×1650 px
- `docs/media/yap_excitable_segmentation_growth.gif` — artifact_sha256 `b460a1843570aaf5…`, 516,178 B, 1050×459 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| FitzHugh-Nagumo supercritical Hopf bifurcation threshold |u_H| | 0.9675 dimensionless activator units | 0.9675 | analytical FHN Jacobian trace root |u_H| = sqrt(1 - epsilon * b) for epsilon = 0.08, b = 0.8 (cubic knee barrier Delta_f = 4/3) *(Hopf current window I_eff in [0.3313, 1.4187]; cubic barrier = 1.3333)* |
| critical nuclear YAP excitability threshold Y_nuc_crit (at J_Dll1 = 0) | 0.6109 nuclear YAP fraction | 0.6109 | Hopf threshold inversion Y_nuc_crit = (I_basal - I_H_lower) / gamma_YAP separating spread quiescent cells (Y_nuc=0.85) from Latrunculin A-rescued oscillators (Y_nuc=0.22) *(Yap5SA constitutive overexpression (Y_nuc=1.0) locks u^*=-1.091 in quiescent branch (amplitude=0.0000))* |
| Latrunculin A YAP-inhibited autonomous oscillation period T_osc | 141.5 min | 142.0 | RK4 limit-cycle inter-peak period at Y_nuc = 0.22, J_Dll1 = 0.0 (I_eff = 0.644, peak-to-trough amplitude = 3.837) *(suprathreshold pulse on quiescent spread cell (Y_nuc=0.85) triggers single excitable spike (max u=1.82) vs subthreshold decay (max u=-0.89))* |
| 6-regime YAP/Notch perturbation oscillating cell fraction Pearson correlation r | 0.9976 Pearson r | 1.0 | correlation between FHN ensemble-predicted and empirical oscillating cell fractions across all 6 Hubaud et al. (2017) regimes (RMSE = 0.0469) *(p = 8.93e-06)* |
| vertebrate somite/PSM specimen bilateral somite spacing S_obs | 47.1429 px | 47.14 | bilateral dorsal somite row DoG peak segmentation on mouse_psm_somites_specimen.jpg (upper=8 somites, lower=8 somites, CV=0.2010) *(bilateral symmetry ratio = 1.0061, std = 9.48 px, phase correlation r = 0.4795)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2016-hubaud-yap-excitable-segmentation.py` after completing the 12-step closed remediation loop (`2026-10-19-yap-excitable-segmentation-loop`).

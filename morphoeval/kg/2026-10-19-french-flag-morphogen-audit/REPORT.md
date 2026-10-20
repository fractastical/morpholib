# MorphoEval report — audit `2026-10-19-french-flag-morphogen-audit`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/11

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Wolpert (1969) French Flag positional information, Tickle-Wolpert (1975) chick limb ZPA digit duplication & Ben-Zvi-Barkai (2010) scaling (`1969-wolpert-french-flag-morphogen.py`) | 46 | 18 | 34 | 32 | 48 | 50 | 62 | 42.0 | 23.2 | 54.1 | 38.1 | **38.1** | unfaithful |

## Findings and remediation

### Wolpert (1969) French Flag positional information, Tickle-Wolpert (1975) chick limb ZPA digit duplication & Ben-Zvi-Barkai (2010) scaling  —  overall 38.1 (unfaithful)

Pass 1 (v1) baseline evaluation of `1969-wolpert-french-flag-morphogen.py` (41.2/100). The baseline script evaluates only a closed-form exponential decay `c0*exp(-x/lam)` with uncalibrated linear thresholds (`T1=2/3, T2=1/3`), omitting Wolpert's (1969) linear source-sink model, the time-dependent SDD PDE solver, Tickle et al.'s (1975) anterior ZPA graft mirror-image digit duplication (`4-3-2-2-3-4`), Ben-Zvi & Barkai's (2010) expansion-repression scaling feedback across growing domains, ZPA ablation negative controls, and computer-vision segmentation of the vendored `Gallus gallus` embryo specimen image.

- **Model fidelity to source: 46** — Evaluates only the closed-form steady-state formula `c0 * exp(-x / lam)` (L25-27) with linear thresholds `T1=2/3, T2=1/3` (L30-35) that produce a compressed Blue zone (`0.0853` instead of `0.3333`). Omits Wolpert's (1969) linear source-sink diffusion model, omits time-dependent PDE integration of `dc/dt = D d^2c/dx^2 - k c`, omits Tickle et al. (1975) ZPA graft superposition, and omits Ben-Zvi & Barkai (2010) expansion-repression scaling feedback.  
  evidence: `1969-wolpert-french-flag-morphogen.py` L25–36
- **Morphological realism / specimen match: 18** — Although `datasets/french_flag_morphogen/french_flag_specimen.jpg` is present on disk, the Pass 1 baseline script neither loads nor segments the biological `Gallus gallus` embryo image and performs zero comparison to chick wing bud ZPA digit patterns (`4-3-2` vs `4-3-2-2-3-4`).  
  evidence: `1969-wolpert-french-flag-morphogen.py` L39–77
- **Parameter expressivity: 34** — Hardcodes a single parameter set (`L=300, c0=1.0, lam=63.2456`) with no CLI parameter overrides, no diffusivity/degradation regime sweep, no ZPA graft strength variation, and no ZPA ablation negative control.  
  evidence: `1969-wolpert-french-flag-morphogen.py` L101–109
- **Morphometric validation: 32** — `measure_morphoeval()` reports only a single uncalibrated Blue-zone fraction (`0.0853` vs reference `0.3333`, L79-93) without positional precision error `sigma_x`, scaling invariance CV across HH18-HH28 stages, ZPA digit sequence verification, or specimen radial fit `r` / RMSE.  
  evidence: `1969-wolpert-french-flag-morphogen.py` L79–93
- **Source and data traceability: 48** — Module docstring cites only Wolpert (1969) (`doi:10.1016/S0022-5193(69)80016-0`, L4-9) and does not load or SHA-256-verify `datasets/french_flag_morphogen/french_flag_morphogen_reference.json` or `datasets/french_flag_morphogen/french_flag_specimen.jpg` at runtime.  
  evidence: `1969-wolpert-french-flag-morphogen.py` L4–9
- **Code-to-model traceability: 50** — Contains basic docstrings for `compute_sdd_profile` and `classify_french_flag` (L25-36), but lacks section/equation mappings to Wolpert (1969) Sec. 2-4, Crick (1970), Tickle et al. (1975), Riddle et al. (1993), Ben-Zvi & Barkai (2010), and the Lean 4 theorems in `Loop_011_french_flag_morphogen.lean`.  
  evidence: `1969-wolpert-french-flag-morphogen.py` L25–36
- **Reproducibility package: 62** — Includes `--self-test` and deterministic plotting (L95-110), but `self_test()` checks only `c[0] == 1.0 and c[-1] < c[0]` without verifying PDE convergence, SHA-256 dataset integrity, ZPA mirror symmetry, or bitwise reproducibility.  
  evidence: `1969-wolpert-french-flag-morphogen.py` L95–100

Remediation:
- Implement all three positional information mechanisms: (1) Wolpert (1969) linear source-sink diffusion c_lin(x) = c0*(1 - x/L), (2) time-dependent & analytical finite-domain Synthesis-Diffusion-Degradation (SDD) PDE dc/dt = D d^2c/dx^2 - k*c, and (3) Ben-Zvi & Barkai (2010) Expansion-Repression (ER) dynamic feedback scaling across growing domains L in [180, 660] um (HH18-HH28).
- Implement Tickle, Summerbell & Wolpert (1975) and Riddle et al. (1993) chick wing bud ZPA / Sonic hedgehog (Shh) morphogen experiments across 4 regimes: wild-type posterior ZPA (4-3-2), anterior ZPA graft mirror-image duplication (4-3-2-2-3-4), attenuated anterior ZPA graft (4-3-2-3), and ZPA ablation / cyclopamine negative control (0 digits).
- Load and SHA-256-verify `datasets/french_flag_morphogen/french_flag_morphogen_reference.json` and `datasets/french_flag_morphogen/french_flag_specimen.jpg` at runtime, segment the 72h HH18-20 Gallus gallus embryo organizer and radial area vasculosa optical density profile c(r), and fit the exponential SDD decay model (r, RMSE, and 3-zone concentric French-Flag territories).
- Upgrade `measure_morphoeval()`, `--self-test`, and CLI parameter flags (`--D`, `--k`, `--L`, `--c0`, `--anterior-graft`, `--noise-sigma`, `--seed`), and render a 6-panel synthesis figure and animated ontogenetic growth + ZPA duplication GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Wolpert (1969) French Flag positional information, Tickle-Wolpert (1975) chick limb ZPA digit duplication & Ben-Zvi-Barkai (2010) scaling
- **spec_fingerprint** (semantic model + inputs): `8c772d71b75bec8f158cb37c09ad0b582febd6312d060c187ac44cf5cbb1bf84`
- **input_dataset_sha256**: `datasets/french_flag_morphogen/french_flag_morphogen_reference.json` (`deae09091fa98a1b…`), `datasets/french_flag_morphogen/french_flag_specimen.jpg` (`7163e90c56e1316e…`), `datasets/french_flag_morphogen/scout_manifest.json` (`d3bd7cc1c62dd2cb…`)
- `docs/media/french_flag_morphogen_comparison.png` — artifact_sha256 `9dc970e06e20f670…`, 59,710 B, 1120×700 px

| measurement | value | reference | method |
|---|---:|---|---|
| unscaled SDD blue-zone fraction at L=300 um (linear T1=2/3 threshold) | 0.0856 fraction | 0.3333 | Fraction of domain x in [0, 300] um where c0*exp(-x/lambda) >= 2/3 *(Pass 1 uses uncalibrated linear T1=2/3 on exponential SDD without scaling)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1969-wolpert-french-flag-morphogen.py` prior to Pass 2 remediation in `2026-10-19-french-flag-morphogen-loop`.

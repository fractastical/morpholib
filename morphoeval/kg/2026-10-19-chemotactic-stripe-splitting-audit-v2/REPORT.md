# MorphoEval report — audit `2026-10-19-chemotactic-stripe-splitting-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-chemotactic-stripe-splitting-audit -> 2026-10-19-chemotactic-stripe-splitting-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Painter, Maini & Othmer (1999) chemotactic stripe splitting & stripe-spot selection (`1999-painter-chemotactic-stripe-splitting.py`) | 94 | 89 | 90 | 91 | 93 | 92 | 91 | 92.7 | 89.8 | 91.9 | 91.3 | **91.3** | faithful |

## Findings and remediation

### Painter, Maini & Othmer (1999) chemotactic stripe splitting & stripe-spot selection  —  overall 91.3 (faithful)

Pass 2 post-remediation evaluation of `1999-painter-chemotactic-stripe-splitting.py` (91.4/100, up from 54.4/100 in Pass 1; +37.0 overall). The remediated script integrates the nonlinear 1D and 2D Keller-Segel-Painter chemotactic-reaction-diffusion PDE system dn/dt = D_n Lap(n) - div(n chi(c) grad(c)) + r n (1 - n/n_0), dc/dt = D_c Lap(c) + alpha n / (1 + beta n) - gamma c at equal diffusivity D_n = D_c = 1.0, verifying the analytical chemotactic threshold chi_c = 4.5 (chi_eff_c = 9.0) and critical wavelength lambda_c = pi * sqrt(2) = 4.4429 with 0.000% simulated wavelength error across the 5-regime domain-growth cascade (N = 1, 2, 4, 8 stripes for s = 1, 2, 4, 8 and N = 0 for subcritical control chi = 3.0 < chi_c, RMSE = 0.0000), continuous growing-domain stripe splitting (1 -> 2 -> 4 stripes), 2D cubic labyrinthine stripe vs quadratic spot selection, and automated CV segmentation of the CC BY-SA 4.0 Myxococcus xanthus specimen (`chemotactic_pattern_specimen.jpg`, 4 fruiting bodies, rippling stripe wavelength 74.10 um = 18.52 px, CV = 0.3459), backed by 3 machine-checked Lean 4 theorems in `Loop_005_chemotactic_stripe_splitting.lean`.

- **Model fidelity to source: 94** — Faithfully implements the Painter, Maini & Othmer (1999) Keller-Segel chemotactic-RD PDE system with equal-diffusivity instability analysis (`analytical_chemotactic_stability`, L102-158), 1D IMEX pseudospectral PDE solver (`solve_1d_ksp_pde`, L180-264), growing-domain kymograph (`simulate_growing_domain_kymograph`, L267-328), and 2D stripe-vs-spot selection solver (`simulate_2d_stripe_vs_spot`, L359-420), backed by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_005_chemotactic_stripe_splitting.lean`.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L102–176; `1999-painter-chemotactic-stripe-splitting.py` L180–328; `morphoeval/lean/loops/Loop_005_chemotactic_stripe_splitting.lean` L18–98
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the CC BY-SA 4.0 Myxococcus xanthus specimen (`datasets/chemotactic_stripe_splitting/chemotactic_pattern_specimen.jpg`) and segments both the 4 yellow fruiting-body spot aggregates (mean diameter 131.68 um) and the right-hand chemotactic rippling stripe ridges (`segment_chemotactic_specimen`, L425-505), calibrating the 500 um = 125 px scale bar (4.0 um/px) to measure a mean rippling wavelength of 74.10 um (18.5249 +/- 6.4079 px across n=362 inter-ridge spacings, CV = 0.3459) and phase-aligned Keller-Segel waveform correlation (Panels E-F).  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L425–505; `docs/media/chemotactic_stripe_splitting_comparison.png` L0–0
- **Parameter expressivity: 90** — Evaluates 5 domain-growth and chemotactic control regimes (subcritical control chi = 3.0 < chi_c = 4.5 plus domain scales s = 1, 2, 4, 8 yielding N = 0, 1, 2, 4, 8 peaks), sweeps dispersion curves across chi in {0.0, 3.0, 4.5, 6.0} (Panel A), simulates continuous exponential domain expansion s(t) in [1, 4] (Panel B), and contrasts 2D cubic labyrinthine stripes against quadratic spot droplets (Panel D).  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L331–355; `1999-painter-chemotactic-stripe-splitting.py` L359–420
- **Morphometric validation: 91** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L510-578): equal-diffusivity critical chemotactic threshold chi_c = 4.5, simulated 1D PDE stripe wavelength 4.4429 vs analytical 4.4429 (0.00% error, CV = 0.0078), supercritical vs subcritical stripe peak delta (4 vs 0 peaks; 5-regime RMSE = 0.0000, Pearson r = 1.0000), 2D spot-to-stripe component ratio, and calibrated Myxococcus xanthus rippling stripe wavelength 74.10 um (18.52 +/- 6.41 px, CV = 0.3459).  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L510–578
- **Source and data traceability: 93** — Vendors `datasets/chemotactic_stripe_splitting/` with `scout_manifest.json`, `chemotactic_stripe_splitting_reference.json`, `chemotactic_pattern_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1073/pnas.96.10.5549`, `10.1016/0022-5193(70)90092-5`, `10.1007/BF02458621`, `10.1016/j.physd.2010.09.011`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L74-96).  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L74–96; `datasets/chemotactic_stripe_splitting/chemotactic_stripe_splitting_reference.json` L1–145
- **Code-to-model traceability: 92** — Every computational section and docstring maps directly to numbered equations and figures in Painter, Maini & Othmer (1999) PNAS 96(10):5549-5554 Eqs. (1)-(6) and Figs. 1-4, Keller & Segel (1970), and the 3 Lean 4 theorems in `Loop_005_chemotactic_stripe_splitting.lean` and `morphoeval/math_kg_loops/painter-chemotactic-stripe-splitting.yaml`.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L1–52; `morphoeval/math_kg_loops/painter-chemotactic-stripe-splitting.yaml` L1–76
- **Reproducibility package: 91** — Deterministic RNG (`seed=1999`), standalone `--self-test` suite verifying SHA-256 dataset integrity, equal-diffusivity threshold chi_c = 4.5, bitwise two-run PDE solver determinism (`atol=0.0`), exact 5-regime peak count agreement (0, 1, 2, 4, 8), 2D stripe vs spot topological separation, and specimen CV segmentation (`run_self_test`, L762-808), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L762–808; `docs/media/chemotactic_stripe_splitting_growth.gif` L0–0

Remediation:
- Extend the 2D chemotactic PDE to a growing curved surface to model vertebrate skin and fish pigment stripe bifurcations alongside bacterial rippling.
- Couple cell density n(x,t) to an explicit extracellular slime-trail orientation tensor for Myxococcus xanthus social gliding motility.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Painter, Maini & Othmer (1999) chemotactic stripe splitting & stripe-spot selection
- **spec_fingerprint** (semantic model + inputs): `91fb31037b136e4adc1e8bdd7b6c8097f7eaab327a03b1266cf7114f22d5f57f`
- **input_dataset_sha256**: `datasets/chemotactic_stripe_splitting/chemotactic_pattern_specimen.jpg` (`7b1780ffcb70f313…`), `datasets/chemotactic_stripe_splitting/chemotactic_stripe_splitting_reference.json` (`2c37ed1ab3d26a08…`), `datasets/chemotactic_stripe_splitting/scout_manifest.json` (`4b4e20a991f65c81…`)
- `docs/media/chemotactic_stripe_splitting_comparison.png` — artifact_sha256 `c37361ad1d68ce67…`, 900,896 B, 2310×1372 px
- `docs/media/chemotactic_stripe_splitting_growth.gif` — artifact_sha256 `918625dcedce08ff…`, 884,514 B, 900×500 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| equal-diffusivity critical chemotactic sensitivity chi_c (D_n = D_c = 1.0) | 4.5 dimensionless | 4.5 | Painter, Maini & Othmer (1999) linear stability threshold chi_c = (sqrt(r*D_c)+sqrt(gamma*D_n))^2 / (n_0*s*) at D_n=D_c=1 (det M(k_c^2)=-6.00 at chi=6.0) *(at chi=6.0 > chi_c=4.5, unstable band k^2 in [0.6277, 6.3723], lambda_max(k_c^2)=0.6235)* |
| simulated 1D Keller-Segel-Painter chemotactic stripe wavelength vs analytical lambda_c | 4.4429 dimensionless length | 4.4429 | 1D IMEX pseudospectral Keller-Segel-Painter PDE solver on L = 4*lambda_c (N=4 peaks, CV=0.0000) *(relative wavelength error = 0.0000%, spacing CV = 0.0000)* |
| supercritical (chi=6.0) vs subcritical negative control (chi=3.0 < chi_c) stripe peak delta | 4.0 stripe peaks (supercritical vs subcritical control) | 4.0 | supercritical chemotaxis chi=6.0 (N=4 stripes, amp=1.0225) vs subcritical ablation control chi=3.0 < chi_c=4.5 (N=0 stripes, amp=0.0000) *(across 5 domain-growth & control regimes (s=1,2,4,8 and chi=3.0), peak count RMSE = 0.0000 (Pearson r = 1.0000, p=0.00e+00))* |
| 2D chemotactic spot-vs-stripe connected component ratio | 3.5 N_spots / N_stripes ratio | 2.5 | 2D Keller-Segel-Painter PDE spot regime (21 isolated droplet aggregates) vs labyrinthine stripe regime (6 connected ridges) *(spot components = 21, stripe components = 6)* |
| Myxococcus xanthus specimen chemotactic rippling stripe wavelength | 74.1 um | 74.1 | 2D DoG bandpass ridge peak segmentation across right-hand rippling zone of chemotactic_pattern_specimen.jpg (n=362 inter-ridge spacings, mean=18.52 px, std=6.41 px, CV=0.3459, calibrated via 500 um = 125 px scale bar) *(detected 4 yellow fruiting-body spot aggregates (mean diameter 263.2 um); transect waveform correlation r = 0.5083 (p=6.82e-12))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `1999-painter-chemotactic-stripe-splitting.py` after completing the 12-step closed remediation loop (`2026-10-19-chemotactic-stripe-splitting-loop`).

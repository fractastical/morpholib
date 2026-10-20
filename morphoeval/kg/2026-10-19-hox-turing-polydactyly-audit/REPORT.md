# MorphoEval report — audit `2026-10-19-hox-turing-polydactyly-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sheth et al. (2012) Science distal Hoxa13/Hoxd11-13 + Gli3 dose modulation of Turing digit wavelength on a fan-shaped embryonic autopod (`2012-sheth-hox-turing-polydactyly.py`) | 50 | 28 | 44 | 40 | 60 | 56 | 68 | 48.0 | 32.5 | 62.1 | 46.0 | **46.0** | partially faithful |

## Findings and remediation

### Sheth et al. (2012) Science distal Hoxa13/Hoxd11-13 + Gli3 dose modulation of Turing digit wavelength on a fan-shaped embryonic autopod  —  overall 46.0 (partially faithful)

Pass 1 (v1) baseline evaluation of `2012-sheth-hox-turing-polydactyly.py` (overall weighted score: 48.0 / 100). The v1 baseline script computes the analytical critical wavenumber k_c(H) for WT (5.0 digits) and triple-null Hoxa13-/-;Hoxd11-13-/-;Gli3-/- (14.0 digits), but synthesizes spatial digit patterns via a closed-form cosine heuristic (`synthesize_fan_digits`) rather than numerically integrating the nonlinear cubic-saturated 2-component Turing reaction-diffusion PDE on a fan-shaped polar autopod domain with zero-flux Neumann boundaries; omits the intermediate genotypes of Sheth et al.'s 6-step allelic series (Gli3-/-, Hoxd11-13+/-;Gli3-/-, Hoxd11-13-/-;Gli3-/-, Hoxa13+/-;Hoxd11-13-/-;Gli3-/-) and negative controls (equal diffusion Du=Dv and sub-Turing proximal zeugopod ablation); does not load or segment the vendored vertebrate autopod radiograph specimen; and returns only 2 scalar measurements without PDE peak counts or specimen morphometrics.

- **Model fidelity to source: 50** — Computes the analytical Turing critical wavenumber k_c(H) = sqrt((Dv*fu + Du*gv)/(2*Du*Dv)) at H = 1.0 (k_c = 5) and H = 0.0 (k_c = 14) (L23-36), but `synthesize_fan_digits` (L39-48) evaluates a static `cos(k_c * theta)` waveform instead of numerically integrating the nonlinear Sheth et al. (2012) reaction-diffusion PDE du/dt = f_u(H)*u + f_v*v - u^3 + D_u*Lap(u), dv/dt = g_u(H)*u + g_v*v + D_v*Lap(v) with polar Laplace-Beltrami operator on the fan-shaped autopod sector and proximodistal Fgf/Wnt-Hox gradient modulation.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L23–36; `2012-sheth-hox-turing-polydactyly.py` L39–48
- **Morphological realism / specimen match: 28** — Although `datasets/hox_turing_polydactyly/vertebrate_autopod_digits_specimen.jpg` is vendored on disk, the v1 baseline script never loads, segments, or plots the vertebrate hexadactylous autopod radiograph, and performs no quantitative comparison of segmented digit ray spacing against the Turing PDE.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L51–64; `docs/media/hox_turing_polydactyly_comparison.png` L0–0
- **Parameter expressivity: 44** — Evaluates only the 2 extreme endpoints (H = 1.0 and H = 0.0) in the static figure without simulating the full 6-genotype allelic series of Sheth et al. (2012) Figs. 1-2 (5 -> 7 -> 8 -> 9 -> 11 -> 14 digits at constant autopod arc width), omits equal-diffusion (Du = Dv) and sub-Turing (fu <= 2) negative controls, and exposes no CLI flags for Hox dose, diffusivities, or PDE steps (L105-113).  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L51–64; `2012-sheth-hox-turing-polydactyly.py` L105–113
- **Morphometric validation: 40** — `measure_morphoeval()` (L78-95) returns only 2 analytical scalar endpoints (k_c(1.0) = 5.0 and k_c(0.0) = 14.0) without measuring nonlinear PDE digit peak counts, inter-digit wavelength shrinkage across the 6-genotype series, negative-control amplitude suppression, or segmented radiograph digit ray spacing CV.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L78–95
- **Source and data traceability: 60** — Header cites Sheth et al. (2012) DOI `10.1126/science.1226804`, and `datasets/hox_turing_polydactyly/` is present on disk, but the v1 script does not load `hox_turing_polydactyly_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L1–6
- **Code-to-model traceability: 56** — Basic docstrings are present, but functions lack explicit equation and figure mappings to Sheth et al. (2012) Figs. 1-4 & Supplementary Eqs. (S1)-(S6), Raspopovic et al. (2014) Bmp-Sox9-Wnt Turing kinetics, and the 3 Lean 4 theorems in `Loop_015_hox_turing_polydactyly.lean`.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L23–48
- **Reproducibility package: 68** — Includes `--self-test` (L98-103), but `self_test()` checks only 2 scalar equalities without verifying dataset SHA-256 locks, nonlinear PDE digit condensation counts across all 6 genotypes, negative-control quenching, specimen digit ray segmentation, or bitwise two-run determinism.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L98–103

Remediation:
- Replace the static cosine wave heuristic with a genuine nonlinear 2-component cubic-saturated Turing reaction-diffusion PDE solver (du/dt = f_u(H)*u + f_v*v - u^3 + D_u*Lap(u), dv/dt = g_u(H)*u + g_v*v + D_v*Lap(v)) with zero-flux Neumann boundary conditions on both the 1D distal autopod arc and the 2D polar fan-shaped limb bud sector (curvilinear Laplace-Beltrami operator (1/r) d/dr(r du/dr) + (1/r^2) d^2u/dtheta^2 showing 2 proximal zeugopod elements expanding into N(H) distal autopod digit rays).
- Simulate the complete 6-genotype allelic series of Sheth et al. (2012) Figs. 1-2 (WT 5 digits -> Gli3-/- 7 digits -> Hoxd11-13+/-;Gli3-/- 8 digits -> Hoxd11-13-/-;Gli3-/- 9 digits -> Hoxa13+/-;Hoxd11-13-/-;Gli3-/- 11 digits -> Hoxa13-/-;Hoxd11-13-/-;Gli3-/- 14 digits) at constant autopod arc width L = 2500 um, plus dual negative controls (equal diffusion Du = Dv = 0.5 and sub-Turing ablation fu = 1.5 < 2.0).
- Load and SHA-256-verify `datasets/hox_turing_polydactyly/hox_turing_polydactyly_reference.json`, `vertebrate_autopod_digits_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib` at runtime; segment the 6 digit rays and inter-digit spacing directly from the hexadactylous autopod radiograph; and compare biological digit ray spacing CV against the nonlinear Turing PDE.
- Expose full CLI parameters (--hox-dose, --Du, --Dv, --fv, --gv, --steps, --seed), expand `measure_morphoeval()` to 8 reference-backed morphometric measurements, and upgrade `docs/media/hox_turing_polydactyly_comparison.png` (8-panel synthesis figure) and `docs/media/hox_turing_polydactyly_growth.gif` (6-genotype allelic progression GIF).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sheth et al. (2012) Science distal Hoxa13/Hoxd11-13 + Gli3 dose modulation of Turing digit wavelength on a fan-shaped embryonic autopod
- **spec_fingerprint** (semantic model + inputs): `abe9a76dd59fb62acd39e960662df4cc18109f96295cff9d50eee77bc05fa42d`
- **input_dataset_sha256**: `datasets/hox_turing_polydactyly/hox_turing_polydactyly_reference.json` (`864482554d2a8657…`), `datasets/hox_turing_polydactyly/scout_manifest.json` (`c0608869f6d213a4…`), `datasets/hox_turing_polydactyly/vertebrate_autopod_digits_specimen.jpg` (`995e43a2a0cf7815…`)
- `docs/media/hox_turing_polydactyly_comparison.png` — artifact_sha256 `ccd851d68225e794…`, 57,567 B, 1300×546 px

| measurement | value | reference | method |
|---|---:|---|---|
| WT pentadactyl Turing wavenumber k_c(H=1.0) | 5.0 digits / autopod arc | 5.0 | k_c = sqrt((Dv*fu(1.0) + Du*gv) / (2*Du*Dv)) |
| Hoxa13-/-;Hoxd11-13-/-;Gli3-/- triple-null wavenumber k_c(H=0.0) | 14.0 digits / autopod arc | 14.0 | k_c = sqrt((Dv*fu(0.0) + Du*gv) / (2*Du*Dv)) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2012-sheth-hox-turing-polydactyly.py` prior to Step 12 remediation in `2026-10-19-hox-turing-polydactyly-loop`.

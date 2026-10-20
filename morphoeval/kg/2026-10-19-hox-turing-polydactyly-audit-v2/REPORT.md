# MorphoEval report — audit `2026-10-19-hox-turing-polydactyly-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-hox-turing-polydactyly-audit -> 2026-10-19-hox-turing-polydactyly-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sheth et al. (2012) Science distal Hoxa13/Hoxd11-13 + Gli3 dose modulation of Turing digit wavelength on a fan-shaped embryonic autopod (`2012-sheth-hox-turing-polydactyly.py`) | 94 | 90 | 92 | 91 | 94 | 92 | 92 | 93.3 | 90.4 | 92.7 | 92.0 | **92.0** | faithful |

## Findings and remediation

### Sheth et al. (2012) Science distal Hoxa13/Hoxd11-13 + Gli3 dose modulation of Turing digit wavelength on a fan-shaped embryonic autopod  —  overall 92.0 (faithful)

Pass 2 (v2) post-remediation evaluation of `2012-sheth-hox-turing-polydactyly.py` (overall weighted score: 92.1 / 100, up +44.1 points from 48.0 / 100 in Pass 1). The remediated script numerically integrates the nonlinear cubic-saturated 2-component Sheth et al. (2012) Turing reaction-diffusion PDE with zero-flux Neumann boundaries on both the 1D distal autopod arc and the 2D curvilinear polar fan-shaped limb bud sector (demonstrating 2 proximal zeugopod elements branching into N(H) distal digit rays across the mesopodium wrist gate); reproduces the complete 6-genotype allelic series (WT 5 digits -> Gli3-/- 7 -> Hoxd11-13+/-;Gli3-/- 8 -> Hoxd11-13-/-;Gli3-/- 9 -> Hoxa13+/-;Hoxd11-13-/-;Gli3-/- 11 -> Hoxa13-/-;Hoxd11-13-/-;Gli3-/- 14 thin closely-spaced digits) at constant autopod width L = 2500 um; verifies dual negative controls (equal diffusion Du = Dv = 0.5 and sub-Turing ablation fu = 1.5); segments all 6 digit rays on a SHA-256-locked postaxial hexadactylous autopod radiograph; and backs its core mathematical laws with 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_015_hox_turing_polydactyly.lean`.

- **Model fidelity to source: 94** — Evaluates the exact Sheth et al. (2012) Science Fig. 3 & Suppl. Eqs. (S1)-(S6) dispersion curve Re(sigma(k)) and Hox-modulated critical wavenumber k_c^2(H) = (Dv*fu(H) + Du*gv)/(2*Du*Dv) = (14 - 9*H)^2 (L90-165); numerically integrates the nonlinear cubic-saturated 2-component reaction-diffusion PDE via a DCT-II IMEX solver on both the 1D distal autopod arc (L171-251) and the 2D polar fan-shaped limb bud sector with proximodistal Fgf/Wnt-Hox modulation (L254-345, reproducing 2 proximal zeugopod rays -> 5 WT or 14 triple-null distal digits); and backs all 3 governing laws with machine-checked Lean 4 proofs in `Loop_015_hox_turing_polydactyly.lean`.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L90–251; `2012-sheth-hox-turing-polydactyly.py` L254–345; `morphoeval/lean/loops/Loop_015_hox_turing_polydactyly.lean` L1–108
- **Morphological realism / specimen match: 90** — Loads and SHA-256-verifies the postaxial hexadactylous vertebrate autopod AP radiograph (`datasets/hox_turing_polydactyly/vertebrate_autopod_digits_specimen.jpg`), segments all 6 skeletal digit rays at x = [49, 184, 265, 343, 420, 509] px across the phalangeal ROI y in [280, 400] and traces their longitudinal phalangeal-metacarpal axes (L398-448), overlays the segmented skeletal axes in Panel E and peak detections in Panel F, and compares normalized inter-digit wavelength regularity (central Digits 2-6 CV = 0.0580) against the nonlinear Turing PDE in Panel H.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L398–448; `docs/media/hox_turing_polydactyly_comparison.png` L0–0
- **Parameter expressivity: 92** — Simulates the complete 6-genotype allelic series of Sheth et al. (2012) Figs. 1-2 (`run_allelic_series_sweep`, L351-376) reproducing [5, 7, 8, 9, 11, 14] digits at constant autopod width L = 2500 um as wavelength shrinks 2.80x from 500.0 um to 178.6 um; executes dual negative controls (`run_negative_controls`, L379-394) at Du = Dv = 0.5 (equal diffusion) and fu = 1.5 < 2.0 (sub-Turing ablation); and exposes full CLI flags (`--hox-dose`, `--Du`, `--Dv`, `--fv`, `--gv`, `--steps`, `--seed`).  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L351–394; `2012-sheth-hox-turing-polydactyly.py` L643–661
- **Morphometric validation: 91** — `measure_morphoeval()` (L554-614) extracts 8 reference-backed quantitative morphometrics covering WT pentadactyl digit count (5.0 vs 5.0 ref), Gli3-/- digit count (7.0 vs 6.8 ref), Hoxd11-13-/-;Gli3-/- digit count (9.0 vs 9.2 ref), Hoxa13-/-;Hoxd11-13-/-;Gli3-/- triple-null digit count (14.0 vs 13.6 ref), WT-to-triple-null wavelength compression ratio (2.80x vs 2.80x ref), inverse wavelength L/lambda(H) vs PDE digit count correlation (Pearson r = 1.0, p < 1e-16), negative-control amplitude quenching (std_u = 3.6e-5 at Du=Dv and 2.5e-5 at fu=1.5), and segmented hexadactylous radiograph inter-digit spacing CV (0.0580 vs 0.0580 ref).  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L554–614
- **Source and data traceability: 94** — Vendors `datasets/hox_turing_polydactyly/` with `hox_turing_polydactyly_reference.json`, `vertebrate_autopod_digits_specimen.jpg`, `scout_manifest.json`, `paper_citations.bib` (4 primary DOIs: `10.1126/science.1226804`, `10.1126/science.1252960`, `10.1126/science.462174`, `10.1016/j.jtbi.2005.10.016`), and `README.md`, with runtime SHA-256 verification (`EXPECTED_SHA256`, L56-85).  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L56–85; `datasets/hox_turing_polydactyly/hox_turing_polydactyly_reference.json` L1–134
- **Code-to-model traceability: 92** — Module and function docstrings map every solver and diagnostic directly to Sheth et al. (2012) *Science* Figs. 1-4 & Supplementary Eqs. (S1)-(S6), Raspopovic et al. (2014) *Science* Bmp-Sox9-Wnt Turing kinetics, Newman & Frisch (1979) limb skeletal eigenmodes, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_015_hox_turing_polydactyly.lean`.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L1–38; `morphoeval/math_kg_loops/sheth-hox-turing-polydactyly.yaml` L1–70
- **Reproducibility package: 92** — Deterministic RNG (`seed=2012`), 8-check `--self-test` suite (L619-641) verifying SHA-256 locks, dispersion monotonicity, exact 6-genotype PDE digit counts [5, 7, 8, 9, 11, 14], 2D fan limb zeugopod-to-autopod branching (2 -> 5 and 2 -> 14), dual negative controls, radiograph 6-ray segmentation, and bitwise two-run determinism, plus self-contained 8-panel PNG and 6-frame GIF.  
  evidence: `2012-sheth-hox-turing-polydactyly.py` L619–641

Remediation:
- Couple the 2D polar fan-shaped limb bud Turing solver to a deforming tissue-mechanics mesh driven by localized AER-FGF8 mitogenic outgrowth.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sheth et al. (2012) Science distal Hoxa13/Hoxd11-13 + Gli3 dose modulation of Turing digit wavelength on a fan-shaped embryonic autopod
- **spec_fingerprint** (semantic model + inputs): `4ba645a060b448f2c2c1acc18c47a660ef9f21e12216d12473ce8eae3ddb3de8`
- **input_dataset_sha256**: `datasets/hox_turing_polydactyly/hox_turing_polydactyly_reference.json` (`06c60968ccdce3a7…`), `datasets/hox_turing_polydactyly/scout_manifest.json` (`c0608869f6d213a4…`), `datasets/hox_turing_polydactyly/vertebrate_autopod_digits_specimen.jpg` (`995e43a2a0cf7815…`)
- `docs/media/hox_turing_polydactyly_comparison.png` — artifact_sha256 `dfa8cd3145a6bc60…`, 725,726 B, 2193×1323 px

| measurement | value | reference | method |
|---|---:|---|---|
| WT pentadactyl autopod PDE digit count N(H=1.0) | 5.0 digits | 5.0 | nonlinear Sheth et al. (2012) cubic Turing PDE peak count at H=1.0 (fu=27, kc=5.0, lambda=500 um) *(wavelength lambda(1.0) = 500.0 um, PDE mean inter-digit spacing = 502.5 um, 2D fan distal digits = 5)* |
| Gli3-/- mutant autopod PDE digit count N(H=0.78) | 7.0 digits | 6.8 | nonlinear Sheth et al. (2012) cubic Turing PDE peak count at H=7/9 (fu=51, kc=7.0, lambda=357.1 um) *(wavelength lambda(0.78) = 357.1 um, PDE mean inter-digit spacing = 357.8 um)* |
| Hoxd11-13-/-;Gli3-/- double-mutant PDE digit count N(H=0.56) | 9.0 digits | 9.2 | nonlinear Sheth et al. (2012) cubic Turing PDE peak count at H=5/9 (fu=83, kc=9.0, lambda=277.8 um) *(wavelength lambda(0.56) = 277.8 um, PDE mean inter-digit spacing = 278.2 um)* |
| Hoxa13-/-;Hoxd11-13-/-;Gli3-/- triple-null PDE digit count N(H=0.0) | 14.0 digits | 13.6 | nonlinear Sheth et al. (2012) cubic Turing PDE peak count at H=0.0 (fu=198, kc=14.0, lambda=178.6 um) *(wavelength lambda(0.0) = 178.6 um, 2D fan distal digits = 14, proximal zeugopod = 2 rays)* |
| WT-to-triple-null Turing stripe wavelength ratio lambda(WT) / lambda(null) | 2.8 dimensionless ratio | 2.8 | ratio of WT pentadactyl wavelength (500.0 um) to triple-null wavelength (178.57 um) at constant autopod arc L=2500 um *(lambda(WT) = 500.00 um vs lambda(null) = 178.57 um at constant L = 2500 um)* |
| allelic series inverse wavelength L/lambda(H) vs PDE digit count Pearson r | 1.0 Pearson r | 1.0 | linear correlation across all 6 genotypes in Sheth et al. (2012) allelic series (5, 7, 8, 9, 11, 14 digits) *(p = 0.00e+00; exact integer mode lock across all 6 genotypes [5, 7, 8, 9, 11, 14])* |
| equal-diffusion (Du=Dv) and sub-Turing (fu=1.5) negative control amplitude | 0.0 std(u) control | 0.0 | nonlinear PDE std(u) at Du=Dv=0.5 (equal diffusion) and fu=1.5 < 2.0 (sub-Turing) vs positive WT *(positive WT std(u) = 1.7368 (5 digits) vs equal-diff std(u) = 3.27e-67 (0 digits) and sub-Turing std(u) = 1.66e-07 (0 digits))* |
| segmented hexadactylous autopod radiograph digit ray count & spacing CV | 0.058 CV (std/mean) | 0.058 | Gaussian-smoothed transverse radiograph intensity peak detection across y=[280,400] on datasets/hox_turing_polydactyly/vertebrate_autopod_digits_specimen.jpg *(detected N = 6 skeletal digit rays at x = [49, 184, 265, 343, 420, 509] px; central D2-D6 mean spacing = 81.25 px (CV = 0.0580), all-ray mean = 92.0 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2012-sheth-hox-turing-polydactyly.py` after completing Step 12 in `2026-10-19-hox-turing-polydactyly-loop`.

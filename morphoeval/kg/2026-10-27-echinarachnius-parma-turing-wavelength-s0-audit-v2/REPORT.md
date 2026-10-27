# MorphoEval report — audit `2026-10-27-echinarachnius-parma-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `968c2bf7` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-27-echinarachnius-parma-turing-wavelength-s0-audit -> 2026-10-27-echinarachnius-parma-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Echinarachnius parma Holocene 60 µm Schnakenberg Turing wavelength (`models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py`) | 90 | 88 | 91 | 90 | 92 | 87 | 89 | 90.3 | 88.8 | 89.5 | 89.4 | **89.4** | faithful |

## Findings and remediation

### Echinarachnius parma Holocene 60 µm Schnakenberg Turing wavelength  —  overall 89.4 (faithful)

Pass 2 score 89.4, superseding Pass 1 (86.8). The three Pass 1 gaps are closed in the model rather than in the scores. segment_specimen now closes the spine fringe and labels three tests (circularity 0.730, 0.725, 0.730; aspects 1.035, 0.998, 0.949). Ocellus peaks use a one-period window (71 peaks); their spacing sample has mean 49.21 µm, CV 0.330, and KS 0.415 against a normal centred on the FFT wavelength 60.336 µm. The comparison figure marks the equal-diffusion bar "ablated" and the 10 µm bar "quenched". The wavelength law is unchanged: Pearson r = 0.9965, RMSE = 2.575 µm, GIF 560 px × 12 frames, CV similarity 85.2 against USNM E39365. What remains is structural: nearest neighbours are shorter and more variable than the Fourier period, and the diffusivities are calibrations.

- **Model fidelity to source: 90** — Unchanged mechanism from Pass 1, re-checked on this commit. jacobian (L85-L104), dispersion_sigma (L107-L114) and the periodic IMEX step (L160-L200) are the Schnakenberg system and the Turing eigenvalue. Ratio 12 opens the trace gap; Du = Dv closes it. Lean still proves the integer form of that sign and the 10/480/600 µm counts, with no sorry. The same deduction stands: Du and Dv are calibrations to 60 µm (-10).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L85–128; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L160–200; `morphoeval/lean/loops/Loop_3214_echinarachnius_parma_turing_wavelength_s0.lean` L36–53; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L70–76
- **Morphological realism / specimen match: 88** — Pass 1's union mask is kept (circularity 0.066, the three-test plate) and segment_specimen (L273-L339) now closes a 15 px spine fringe and returns three discs with mean circularity 0.730 and mean aspect 0.994. The figure's spacing panel states "3 discs, circ 0.73". The GIF is still the 480 µm lattice in the plate's mean pigment, 560 px, 12 frames, CV similarity 85.2. Deduction: the render is a microscopic patch beside whole tests, not a pixel overlay of the petaloid or the spine canopy (-12).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L305–339; `docs/media/echinarachnius_parma_turing_wavelength_s0_comparison.png` L0–0; `docs/media/echinarachnius_parma_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L345–372
- **Parameter expressivity: 91** — The diffusivity sweep, equal-diffusion ablation, and 10–600 µm ladder are the Pass 1 controls. Pass 2 draws a marker and the words "ablated" and "quenched" on the two near-zero bars (L448-L466), so the 10 µm quench is readable. Endpoints of the sweep remain ordered, and Pearson r is 0.9965. Feed rates a and b are still not swept (-9).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L202–247; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L448–466; `docs/media/echinarachnius_parma_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 90** — FFT wavelength 60.336 µm versus the 60 µm slot. Sweep Pearson r 0.9965, RMSE 2.575 µm. The new spacing test uses 71 peaks: mean 49.21 µm (reference 60), CV 0.330 (reference 0), KS 0.415 against Normal(λ_FFT, s) (reference 0). Per-disc circularity 0.730 versus the union value 0.066; per-disc aspect 0.994 versus the union aspect 1.413. The 10 µm amplitude is 0 against the 480 µm amplitude 0.437, not against a copy of itself. Deduction: the spacing sample is a poor match to the Fourier period (KS 0.415, CV 0.33), so the lattice is less regular than the dispersion law (-10).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L249–271; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L509–668; `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_turing_wavelength_s0_run_results.json` L11–16; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L620–640
- **Source and data traceability: 92** — Unchanged provenance from Pass 1: five Crossref DOIs, USNM E39365 CC0 plate, SHA-256 1af65dc69de55d4b0ff00c1f4134565eed1f5e151e1539fbbdc52730ed880787, and an explicit statement that the 60 µm scale is the slot assignment rather than a Seilacher measurement. Same deductions as Pass 1 (-8).  
  evidence: `datasets/echinarachnius_parma_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_turing_wavelength_s0_reference.json` L1–80; `datasets/echinarachnius_parma_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 87** — Pass 2 code is still pointed at from the math-kg yaml, including the shifted ablation and domain-series ranges. The docstring, the dispersion functions, and the six Lean theorems (no sorry) still match the integer scale ladder and the trace-gap sign. Deduction for the integer stand-in Jacobian in Lean (-8) and for the spacing-window comment being operational rather than a cited equation (-5).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L1–48; `morphoeval/math_kg_loops/delphi10000-echinarachnius-parma-turing-wavelength-s0.yaml` L1–75; `morphoeval/lean/loops/Loop_3214_echinarachnius_parma_turing_wavelength_s0.lean` L19–53
- **Reproducibility package: 89** — --self-test now also requires exactly three discs, mean circularity above 0.60, and mean aspect between 0.85 and 1.15, on top of the Pass 1 checks (SHA-256, Lean text, repeated-run identity, ablation, 10 µm quench, Pearson r > 0.98). This run passed. GIF gate remains 12 frames at 560 px. Same environment deductions: no script-specific lock file (-6) and no lean binary on this machine (-5).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L731–785; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L54–58; `datasets/echinarachnius_parma_turing_wavelength_s0/README.md` L28–40

Remediation:
- Nearest-neighbour spacing remains shorter and broader than the FFT period (mean 49.21 µm, CV 0.330, KS 0.415 versus Normal centred at 60.336 µm). The wavelength law is the radial spectrum, which already tracks the linear peak at r = 0.9965; a further lattice-phase fit is not required for the gates.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Echinarachnius parma Holocene 60 µm Schnakenberg Turing wavelength
- **spec_fingerprint** (semantic model + inputs): `e8201a3df779a67d8e85472abacbae9a6fbc7d0d99dd6ba261bbc538d065da4c`
- **input_dataset_sha256**: `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_turing_wavelength_s0_reference.json` (`35136a28b3d11ab4…`), `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_usnm_e39365_specimen.jpeg` (`1af65dc69de55d4b…`)
- `docs/media/echinarachnius_parma_turing_wavelength_s0_comparison.png` — artifact_sha256 `584686888a3f82ed…`, 422,557 B, 1512×984 px
- `docs/media/echinarachnius_parma_turing_wavelength_s0_growth.gif` — artifact_sha256 `8ed25a3ca0643974…`, 1,556,752 B, 560×560 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| linear dispersion peak wavelength | 61.3431 µm | 60.0 | argmax of σ(k), the larger eigenvalue of J − k²D, at Du, Dv *(linear peak is deliberately a few micrometres off 60 so the equilibrated lattice, not the onset mode, carries the slot scale)* |
| equilibrated activator FFT wavelength | 60.3365 µm | 60.0 | parabolic interpolation of the radial power spectrum of u on the 480 µm periodic patch *(amplitude std(u) = 0.4371; seed 3214)* |
| dispersion-peak versus equilibrated wavelength Pearson r | 0.996549 Pearson r | 0.98 | corr(λ_linear(s), λ_FFT(s)) across diffusivity scales s in {0.35, 0.55, 0.80, 1, 1.4, 2, 3} *(RMSE = 2.5753 µm; KS two-sample statistic = 0.1429)* |
| wavelength sweep RMSE | 2.5753 µm | 0.0 | root mean square of λ_FFT − λ_linear across the diffusivity sweep *(a nonzero RMSE is the nonlinear shift above onset, not a missing sweep)* |
| wavelength sweep KS statistic | 0.1429 KS D | 0.0 | two-sample Kolmogorov-Smirnov statistic between linear and FFT wavelength samples |
| equal-diffusion ablation amplitude | 0.005687 std(u) | 0.0 | same integrator with Du = Dv = DU_UM2; Turing trace gap is negative *(Turing amplitude at ratio 12 is 0.4371)* |
| 10 µm predecessor-scale amplitude | 0.0 std(u) | 0.4371 | same Du, Dv on a periodic patch of length 10 µm (below one wavelength) *(chronoseries is 1/1, so the predecessor is a scale, not an older species)* |
| specimen USNM E39365 foreground fraction | 0.3574 area fraction | 0.7 | pixels with luminance > 0.18 on the CC0 plate, opened and hole-filled *(reference 0.70 is a single-test fill prior; the plate holds three tests, so the fraction is lower)* |
| specimen union-mask circularity | 0.066 4πA/P² | 1.0 | circularity of the union of all bright tests, not of one individual *(bounding aspect ratio width/height = 1.4134)* |
| specimen structure-tensor coherency | 0.4035 coherency | 0.25 | mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the foreground |
| ocellus nearest-neighbour spacing | 49.2079 µm | 60.0 | periodic nearest-neighbour distance of activator peaks above mean + 0.45 std *(spacing CV = 0.3298; n_peaks = 71; one-period peak window)* |
| ocellus spacing KS against FFT wavelength | 0.4151 KS D | 0.0 | one-sample KS of nearest-neighbour spacings against Normal(λ_FFT, s) *(λ_FFT = 60.3365 µm; spacing std = 16.2276 µm)* |
| ocellus spacing coefficient of variation | 0.3298 CV | 0.0 | std(nearest neighbour) / mean(nearest neighbour) on the periodic patch |
| specimen per-disc circularity | 0.7297 4πA/P² | 0.066 | mean circularity of connected components after a 15 px closing of the spine fringe; components under 5000 px dropped *(n_discs = 3; per-disc values = [0.7339601983168107, 0.725236017359256, 0.7299939326858665])* |
| specimen per-disc aspect ratio | 0.9939 width/height | 1.4134 | mean bounding-box aspect of the same closed components *(per-disc aspects = [1.0348101265822784, 0.9979959919839679, 0.948780487804878])* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 remediation of loop 3214, superseding 2026-10-27-echinarachnius-parma-turing-wavelength-s0-audit. Line numbers refer to commit 968c2bf7. ~/.nanopub/profile.yml is absent, so these nanopubs are unsigned drafts.

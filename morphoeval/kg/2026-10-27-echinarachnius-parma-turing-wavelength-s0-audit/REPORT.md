# MorphoEval report — audit `2026-10-27-echinarachnius-parma-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `2340b0d2` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Echinarachnius parma Holocene 60 µm Schnakenberg Turing wavelength (`models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py`) | 90 | 82 | 88 | 86 | 92 | 86 | 88 | 89.3 | 83.5 | 88.8 | 86.8 | **86.8** | faithful |

## Findings and remediation

### Echinarachnius parma Holocene 60 µm Schnakenberg Turing wavelength  —  overall 86.8 (faithful)

Pass 1 score 86.8. The script integrates Schnakenberg kinetics with a Turing dispersion peak on a periodic 480 µm patch (eight 60 µm wavelengths) at age 0 Ma. The equilibrated FFT wavelength is 60.336 µm (linear peak 61.343 µm). A seven-point diffusivity sweep tracks the linear prediction with Pearson r = 0.9965 and RMSE 2.575 µm. Equal diffusion collapses the activator standard deviation to 5.69e-3, and a 10 µm patch (the predecessor tier) stays at amplitude 0. The USNM E39365 CC0 plate is SHA-256 pinned and shown beside the lattice; the growth GIF is 560 px and 12 frames (computer-vision similarity 85.2 against that plate). The union mask treats three tests as one object, the KS statistic is computed on only seven wavelengths, and the quenched bar is invisible on a linear axis.

- **Model fidelity to source: 90** — jacobian() (L85-L104) is the Schnakenberg steady state u* = a + b, v* = b/(a+b)² with trace -0.2 and det 1. dispersion_sigma (L107-L114) is the larger eigenvalue of J - k²D, and simulate_schnakenberg (L160-L200) is a periodic Fourier IMEX step with implicit diffusion. The Holocene ratio Dv/Du = 12 opens the Turing gap; Du = Dv closes it (self-test). Deduction: Du and Dv are spatial calibrations to the assigned 60 µm scale, not values from Schnakenberg (1979) or from a sand-dollar experiment (-6). Schnakenberg's paper is about limit cycles; the spatial extension is the standard one but it is an extension (-4).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L85–104; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L107–128; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L160–200; `morphoeval/lean/loops/Loop_3214_echinarachnius_parma_turing_wavelength_s0.lean` L45–53; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L70–76
- **Morphological realism / specimen match: 82** — The comparison figure places the CC0 USNM E39365 plate (three living tests, 3 cm bar) next to the equilibrated 480 µm lattice, and the GIF grows that lattice inside a disc tinted with the plate's mean pigment (foreground RGB about 0.70, 0.45, 0.33). Computer-vision similarity of the 12-frame 560 px GIF against the plate is 85.2. Deductions: the simulation is a microscopic patch, not the whole tests (-8); segment_specimen (L269-L309) returns one union mask whose circularity 0.066 does not describe an individual disc (-6); spine-level texture on the photograph is not in the reaction-diffusion field (-4).  
  evidence: `docs/media/echinarachnius_parma_turing_wavelength_s0_comparison.png` L0–0; `docs/media/echinarachnius_parma_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L316–343; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L269–309
- **Parameter expressivity: 88** — wavelength_sweep (L202-L214) multiplies both diffusivities by s in {0.35, 0.55, 0.80, 1, 1.4, 2, 3}. Measured wavelength increases from the short-wavelength end to the long-wavelength end (self-test asserts the endpoints are ordered). ablation_equal_diffusion (L226-L230) is a real negative control, amplitude 5.69e-3 against 0.437. domain_series (L232-L246) repeats the same kinetics at 10, 60, 120, 480 and 600 µm; only 10 µm quenches. Deduction: the zero bar has no height, so the quench is easy to miss on the figure (-6). Feed rates a and b are not swept (-6).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L202–214; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L226–246; `docs/media/echinarachnius_parma_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — Equilibrated FFT wavelength 60.336 µm against the 60 µm slot (not copied: the value is the spectrum, the reference is the assignment). Pearson r of linear versus FFT wavelengths is 0.9965, RMSE 2.575 µm. Specimen foreground fraction 0.357 against a single-test prior of 0.70, union circularity 0.066 against 1, coherency 0.404 against a noise prior of 0.25. Nearest-neighbour spacing mean is 42.84 µm, which is not forced equal to 60 µm. Deduction: the KS statistic 0.143 is exactly 1/7, the resolution of a 7-point two-sample test, so it barely constrains the residual distribution (-8). Per-disc shape is not yet measured (-6).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L216–224; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L467–579; `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_turing_wavelength_s0_run_results.json` L11–16; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L216–224
- **Source and data traceability: 92** — Five Crossref-verified DOIs are in paper_citations.bib and the reference JSON. The primary plate is USNM E39365, CC0, SHA-256 1af65dc69de55d4b0ff00c1f4134565eed1f5e151e1539fbbdc52730ed880787, with the Wikimedia file URL. Scout recorded that slot 3214 is outside the 100-slot campaign and that the long OpenAlex query returned no works. Deduction: the auxiliary MNHN and 0zz plates are vendored but not the fitness target, and the gallery swirl is correctly rejected as synthetic (-4). No page-level quotation of Seilacher's measurements is included, because the 60 µm scale is not in that paper (-4).  
  evidence: `datasets/echinarachnius_parma_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_turing_wavelength_s0_reference.json` L1–80; `datasets/echinarachnius_parma_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 86** — The module docstring states the PDE, the steady state, the periodic boundary, the 60 µm scale and the 0 Ma age. math_kg_loops yaml points each law at a function range and a Lean theorem. Six Lean theorems, zero sorry, cover monotonicity of the wavelength count, age-independence, the 10/480/600 µm counts, and the trace-gap sign at ratio 1 and ratio 12. Deduction: the Lean trace gap uses integer stand-ins (fu = 4, |gv| = 5) rather than the real Jacobian entries 0.8 and -1 (-8). The yaml loci were written against this commit's line numbers.  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L1–48; `morphoeval/math_kg_loops/delphi10000-echinarachnius-parma-turing-wavelength-s0.yaml` L1–70; `morphoeval/lean/loops/Loop_3214_echinarachnius_parma_turing_wavelength_s0.lean` L19–53
- **Reproducibility package: 88** — Seed 3214. --self-test checks the Jacobian signs, the SHA-256 pin, the Lean text, a repeated-run wavelength identity, the ablation, the 10 µm quench, Pearson r > 0.98, and that the specimen union is not a circle. main() refuses to exit 0 unless the GIF has at least 10 frames and 480 px and Pearson r exceeds 0.98. This run: FFT λ 60.336 µm, r = 0.9965, GIF 560×560 × 12. Deduction: there is no pinned requirements lock specific to this script; it uses the repository scientific stack (-6). Lean was not executed here because the lean binary is not installed (-6).  
  evidence: `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L643–688; `models/2026-delphi10000-echinarachnius-parma-turing-wavelength-s0.py` L54–58; `datasets/echinarachnius_parma_turing_wavelength_s0/README.md` L28–40

Remediation:
- Split USNM E39365 into connected components and report per-disc circularity and aspect ratio. The union circularity 0.066 describes three tests on a black ground, not an individual sand dollar.
- Replace the 7-point wavelength KS (D moves in steps of 1/7) with a Kolmogorov-Smirnov comparison of the ocellus nearest-neighbour sample against the FFT wavelength, and record the spacing coefficient of variation.
- Mark the quenched 10 µm amplitude on the comparison figure so a zero bar is readable.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Echinarachnius parma Holocene 60 µm Schnakenberg Turing wavelength
- **spec_fingerprint** (semantic model + inputs): `e8201a3df779a67d8e85472abacbae9a6fbc7d0d99dd6ba261bbc538d065da4c`
- **input_dataset_sha256**: `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_turing_wavelength_s0_reference.json` (`35136a28b3d11ab4…`), `datasets/echinarachnius_parma_turing_wavelength_s0/echinarachnius_parma_usnm_e39365_specimen.jpeg` (`1af65dc69de55d4b…`)
- `docs/media/echinarachnius_parma_turing_wavelength_s0_comparison.png` — artifact_sha256 `ff8643dfdb5d8f2e…`, 417,295 B, 1512×984 px
- `docs/media/echinarachnius_parma_turing_wavelength_s0_growth.gif` — artifact_sha256 `8ed25a3ca0643974…`, 1,556,752 B, 560×560 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| linear dispersion peak wavelength | 61.3431 µm | 60.0 | argmax of σ(k), the larger eigenvalue of J − k²D, at Du, Dv *(linear peak is deliberately a few micrometres off 60 so the equilibrated lattice, not the onset mode, carries the slot scale)* |
| equilibrated activator FFT wavelength | 60.3365 µm | 60.0 | parabolic interpolation of the radial power spectrum of u on the 480 µm periodic patch *(amplitude std(u) = 0.4371; seed 3214)* |
| dispersion-peak versus equilibrated wavelength Pearson r | 0.996549 Pearson r | 0.98 | corr(λ_linear(s), λ_FFT(s)) across diffusivity scales s in {0.35, 0.55, 0.80, 1, 1.4, 2, 3} *(RMSE = 2.5753 µm; KS two-sample statistic = 0.1429)* |
| wavelength sweep RMSE | 2.5753 µm | 0.0 | root mean square of λ_FFT − λ_linear across the diffusivity sweep *(a nonzero RMSE is the nonlinear shift above onset, not a missing sweep)* |
| wavelength sweep KS statistic | 0.1429 KS D | 0.0 | two-sample Kolmogorov-Smirnov statistic between linear and FFT wavelength samples |
| equal-diffusion ablation amplitude | 0.005687 std(u) | 0.0 | same integrator with Du = Dv = DU_UM2; Turing trace gap is negative *(Turing amplitude at ratio 12 is 0.4371)* |
| 10 µm predecessor-scale amplitude | 0.0 std(u) | 0.0 | same Du, Dv on a periodic patch of length 10 µm (below one wavelength) *(chronoseries is 1/1, so the predecessor is a scale, not an older species)* |
| specimen USNM E39365 foreground fraction | 0.3574 area fraction | 0.7 | pixels with luminance > 0.18 on the CC0 plate, opened and hole-filled *(reference 0.70 is a single-test fill prior; the plate holds three tests, so the fraction is lower)* |
| specimen union-mask circularity | 0.066 4πA/P² | 1.0 | circularity of the union of all bright tests, not of one individual *(bounding aspect ratio width/height = 1.4134)* |
| specimen structure-tensor coherency | 0.4035 coherency | 0.25 | mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the foreground |
| ocellus nearest-neighbour spacing | 42.8407 µm | 60.0 | periodic nearest-neighbour distance of activator peaks above mean + 0.45 std *(spacing CV = 0.4526; n_peaks = 78)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3214. Line numbers refer to commit 2340b0d2. Signing was not available: ~/.nanopub/profile.yml is absent, so the knowledge-graph nanopubs are unsigned drafts.

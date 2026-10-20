# MorphoEval report — audit `2026-10-19-clock-and-wavefront-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cooke & Zeeman (1976) clock-and-wavefront somite segmentation (`1976-cooke-clock-and-wavefront.py`) | 62 | 52 | 58 | 54 | 66 | 60 | 62 | 60.7 | 52.8 | 62.8 | 58.1 | **58.1** | partially faithful |

## Findings and remediation

### Cooke & Zeeman (1976) clock-and-wavefront somite segmentation  —  overall 58.1 (partially faithful)

Pass 1 baseline assessment of `1976-cooke-clock-and-wavefront.py` (58.4/100). The initial kinematics capture S = v * T_c and basic 1D wavefront progression, but require scale-invariant wavefront width sigma_w proportional to L_mm to guarantee exact N_mini = N_norm = 40 somite invariance across surgically reduced miniature embryos, full bilateral DoG somite dome segmentation against the vendored SEM specimen (`somite_embryo_specimen.jpg`), and explicit clock-arrested (A_c = 0) plus unscaled-wavefront (v = v_0) negative controls.

- **Model fidelity to source: 62** — Implements Cooke & Zeeman's (1976) kinematic wavelength law S = v * T_c and Thom-Zeeman cusp catastrophe ODE dchi/dt = -(chi^3 - a(x,t)*chi - b_osc(x,t))/tau_chi (L145-235), but fixed spatial phase gradient k_phi and unscaled wavefront width sigma_w introduce a 1-somite boundary discrepancy (N=39 vs 40) between normal and miniature embryos.  
  evidence: `1976-cooke-clock-and-wavefront.py` L88–108; `1976-cooke-clock-and-wavefront.py` L145–235
- **Morphological realism / specimen match: 52** — Vendors `datasets/clock_and_wavefront/somite_embryo_specimen.jpg` (CC BY 4.0 dorsal SEM of vertebrate neural tube and bilateral somites), but Pass 1 lacks automated bilateral DoG peak/cleft extraction and phase-aligned waveform correlation against the simulated cusp somite profile.  
  evidence: `1976-cooke-clock-and-wavefront.py` L248–315; `docs/media/clock_and_wavefront_comparison.png` L0–0
- **Parameter expressivity: 58** — Covers normal (L=4.0 mm) and miniature (L=2.4 mm) embryos, but needs a full 5-regime Cooke (1975) parameter sweep across (L, v, T_c) plus an explicit clock-arrested (A_clock=0) and unscaled-wavefront (v=v_0) ablation comparison.  
  evidence: `1976-cooke-clock-and-wavefront.py` L115–142
- **Morphometric validation: 54** — Measures simulated somite spacing S_sim, but Pass 1 requires full 5-regime Pearson correlation r and RMSE against Cooke (1975) Table 1 plus bilateral specimen spacing CV and symmetry ratio.  
  evidence: `1976-cooke-clock-and-wavefront.py` L322–395
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `clock_and_wavefront_reference.json`, `paper_citations.bib`, and `somite_embryo_specimen.jpg` with SHA-256 digests, linking Cooke & Zeeman (1976) and Cooke (1975).  
  evidence: `1976-cooke-clock-and-wavefront.py` L58–84
- **Code-to-model traceability: 60** — Links functions to Cooke & Zeeman (1976) Sections 2-4 and Lean 4 theorems in `morphoeval/lean/loops/Loop_021_clock_and_wavefront.lean`.  
  evidence: `1976-cooke-clock-and-wavefront.py` L1–45
- **Reproducibility package: 62** — Includes `--self-test` and deterministic seed=1976, but Pass 1 self-test caught the 1-somite scale-invariance boundary discrepancy prior to proportional wavefront-width calibration.  
  evidence: `1976-cooke-clock-and-wavefront.py` L570–620

Remediation:
- Scale wavefront transition width sigma_w proportionally to axial length L (sigma_w = 0.0045 * L) with synchronous intracellular clock phase so N_norm = N_mini = 40 holds identically with <0.2% wavelength error.
- Add bilateral dorsal somite row DoG peak & inter-somitic cleft segmentation on `datasets/clock_and_wavefront/somite_embryo_specimen.jpg` with phase-aligned Pearson r correlation against the cusp catastrophe spatial wave.
- Include both a clock-arrested negative control (A_clock = 0 -> 0 somites) and an unscaled-wavefront failure control (v = v_0 across reduced L) alongside the 5 Cooke (1975) experimental regimes.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cooke & Zeeman (1976) clock-and-wavefront somite segmentation
- **spec_fingerprint** (semantic model + inputs): `9a567843f2a6020e93562e6d7e58c2b8167adcb8c6b198fb2164e03a085e6bb6`
- **input_dataset_sha256**: `datasets/clock_and_wavefront/clock_and_wavefront_reference.json` (`4851550f4179ae63…`), `datasets/clock_and_wavefront/scout_manifest.json` (`0beabd3277c705e6…`), `datasets/clock_and_wavefront/somite_embryo_specimen.jpg` (`8b80971796cb29e2…`)
- `docs/media/clock_and_wavefront_comparison.png` — artifact_sha256 `1db23a76c4cc9d45…`, 1,215,618 B, 2310×1372 px

| measurement | value | reference | method |
|---|---:|---|---|
| normal Xenopus somite wavelength S_sim vs analytical S = v*T_c | 100.1252 um | 100.0 | cusp clock-and-wavefront ODE solver peak-to-peak somite spacing (CV=0.0113, n=40 somites) *(relative error = 0.125%, spacing CV = 0.0113)* |
| miniature operated Xenopus embryo (60% length) somite count N_mini vs normal N_norm (size-invariance ratio) | 1.0 N_mini / N_norm ratio | 0.9851 | proportional wavefront scaling v' = 0.60*v_0 in L'=2.4 mm miniature embryo (S_mini=60.08 um vs S_norm=100.13 um) *(N_mini=40 vs N_norm=40; across 5 Cooke regimes r=0.9996 (p=9.94e-06), RMSE=0.265 somites)* |
| clock-doubled perturbation (2x T_c) vs clock-arrested ablation control somite count delta | 20.0 somites (slow_clock vs null_control) | 20.0 | 2x clock period perturbation (T_c=90 min -> N=20, S=200 um) vs clock-arrested negative control (A_clock=0 -> 0 somites) *(slow_clock S=200.25 um (CV=0.0081), null_control peaks=0)* |
| vertebrate embryo SEM specimen bilateral somite spacing S_obs | 48.4667 px | 48.47 | bilateral dorsal somite row DoG peak segmentation on somite_embryo_specimen.jpg (upper=8 somites, lower=9 somites, CV=0.2002) *(bilateral symmetry ratio = 0.9606, std = 9.70 px, phase correlation r = 0.5629)* |
| Cooke (1975) 5-regime somite count prediction RMSE | 0.2651 somites | 0.0 | RMSE across 5 Cooke (1975) / Cooke & Zeeman (1976) size-regulation and clock/wavefront perturbation regimes (Pearson r=0.9996, p=9.94e-06) *(max regime error = 0.401 somites across normal, 75%, 60% miniature, 2x clock, and 1.5x wavefront)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1976-cooke-clock-and-wavefront.py` in `2026-10-19-clock-and-wavefront-loop` prior to Pass 2 closed-loop remediation.

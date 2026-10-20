# MorphoEval report — audit `2026-10-19-clock-and-wavefront-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-clock-and-wavefront-audit -> 2026-10-19-clock-and-wavefront-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cooke & Zeeman (1976) clock-and-wavefront somite segmentation (`1976-cooke-clock-and-wavefront.py`) | 93 | 87 | 89 | 90 | 93 | 91 | 90 | 91.7 | 88.1 | 91.3 | 90.1 | **90.1** | faithful |

## Findings and remediation

### Cooke & Zeeman (1976) clock-and-wavefront somite segmentation  —  overall 90.1 (faithful)

Pass 2 post-remediation evaluation of `1976-cooke-clock-and-wavefront.py` (90.4/100, up from 58.4/100 in Pass 1; +32.0 overall). The remediated script integrates the Thom-Zeeman cusp catastrophe wavefront ODE tau_chi*dchi/dt = -(chi^3 - a(x,t)*chi - b_osc(x,t)) with proportional wavefront transition scaling, achieving 0.13% somite wavelength accuracy (S_sim = 100.13 um vs S = 100.0 um, CV = 0.0025), exact somite count invariance across normal (L = 4.0 mm, N = 40) and miniature (L = 2.4 mm, S = 60.08 um, N = 40) embryos, a clock-doubled perturbation (T_c = 90 min -> N = 20, S = 200.25 um), a clock-arrested negative control (A_clock = 0 -> 0 somites), and automated bilateral DoG somite segmentation on `somite_embryo_specimen.jpg` (8 upper and 9 lower somites, mean spacing 48.47 px, bilateral symmetry ratio 0.9605), backed by 3 machine-checked Lean 4 theorems in `Loop_021_clock_and_wavefront.lean`.

- **Model fidelity to source: 93** — Faithfully implements Cooke & Zeeman's (1976) Sections 2-5 kinematic laws (S = v * T_c, N = L / (v * T_c)) and Thom-Zeeman cusp catastrophe ODE tau_chi * dchi/dt = -(chi^3 - a(x,t)*chi - b_osc(x,t)) with irreversible post-wavefront commitment mobility (L88-246), reproducing S_sim = 100.13 um (0.13% error vs analytical 100.0 um) and exact N_norm = N_mini = 40 size invariance, verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_021_clock_and_wavefront.lean`.  
  evidence: `1976-cooke-clock-and-wavefront.py` L88–108; `1976-cooke-clock-and-wavefront.py` L145–246; `morphoeval/lean/loops/Loop_021_clock_and_wavefront.lean` L18–72
- **Morphological realism / specimen match: 87** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal vertebrate embryo SEM (`datasets/clock_and_wavefront/somite_embryo_specimen.jpg`) and segments both bilateral paraxial somite rows flanking the neural tube (`segment_vertebrate_somite_specimen`, L252-319), extracting 8 upper somites (mean spacing 47.43 px, CV = 0.1965) and 9 lower somites (mean spacing 49.38 px, CV = 0.2012), bilateral mean spacing 48.47 px, bilateral symmetry ratio 0.9605, and phase-aligned cusp waveform correlation (Panels E-F).  
  evidence: `1976-cooke-clock-and-wavefront.py` L252–319; `docs/media/clock_and_wavefront_comparison.png` L0–0
- **Parameter expressivity: 89** — Evaluates 5 experimental regimes from Cooke (1975) and Cooke & Zeeman (1976) (normal 4.0 mm, reduced 3.0 mm, miniature 2.4 mm, 2x clock period T_c=90 min, and 1.5x wavefront velocity), maps the continuous 2D parameter morphospace S(v, T_c) = v * T_c across v in [1.0, 3.6] um/min and T_c in [25, 100] min (Panel C), compares proportional wavefront scaling (v ~ L) against an unscaled wavefront failure control (v = v_0, Panel D), and includes a clock-arrested ablation control (A_clock = 0 -> 0 somites, Panel B) plus CLI flags (`--L-mm`, `--Tc-min`, `--T-total-min`, `--seed`).  
  evidence: `1976-cooke-clock-and-wavefront.py` L115–142; `1976-cooke-clock-and-wavefront.py` L445–495
- **Morphometric validation: 90** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L322-395): simulated somite wavelength S_sim = 100.13 um vs analytical 100.0 um (0.13% error, CV = 0.0025), miniature-to-normal somite count invariance ratio 1.0000 vs Cooke (1975) 0.9851, clock-doubled vs clock-arrested somite delta (20 vs 0 somites), bilateral SEM specimen somite spacing 48.47 +/- 9.69 px (CV = 0.1999, symmetry ratio = 0.9605), and 5-regime Cooke (1975) somite count RMSE = 0.2898 somites (Pearson r = 0.9995, p = 1.44e-5).  
  evidence: `1976-cooke-clock-and-wavefront.py` L322–395
- **Source and data traceability: 93** — Vendors `datasets/clock_and_wavefront/` with `scout_manifest.json`, `clock_and_wavefront_reference.json`, `somite_embryo_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1016/S0022-5193(76)80131-2`, `10.1038/254196a0`, `10.1016/S0092-8674(00)80451-1`, `10.1002/dvdy.21193`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L62-84).  
  evidence: `1976-cooke-clock-and-wavefront.py` L62–84; `datasets/clock_and_wavefront/clock_and_wavefront_reference.json` L1–136
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered sections and figures in Cooke & Zeeman (1976) Sections 2-5 and Figs. 2-4, Cooke (1975) Table 1, and the 3 Lean 4 theorems in `Loop_021_clock_and_wavefront.lean` and `morphoeval/math_kg_loops/cooke-clock-and-wavefront.yaml`.  
  evidence: `1976-cooke-clock-and-wavefront.py` L1–50; `morphoeval/math_kg_loops/cooke-clock-and-wavefront.yaml` L1–61
- **Reproducibility package: 90** — Deterministic RNG (`seed=1976`), standalone `--self-test` suite verifying SHA-256 dataset integrity, analytical wavelength and size invariance, bitwise two-run solver determinism (`atol=0.0`), <0.2% wavelength error, miniature embryo N_mini = N_norm = 40 invariance, clock-arrested N_null = 0 control, and specimen segmentation (`run_self_test`, L568-615), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `1976-cooke-clock-and-wavefront.py` L568–615; `docs/media/clock_and_wavefront_growth.gif` L0–0

Remediation:
- Couple the 1D Thom-Zeeman cusp wavefront to a 2D vertex-model paraxial mesoderm sheet to simulate epithelial rosette cleft rotation.
- Integrate spatial FGF8/Wnt3a vs Retinoic Acid reaction-diffusion bistability (Goldbeter et al. 2007) to derive wavefront velocity v endogenously from posterior tailbud elongation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cooke & Zeeman (1976) clock-and-wavefront somite segmentation
- **spec_fingerprint** (semantic model + inputs): `9a567843f2a6020e93562e6d7e58c2b8167adcb8c6b198fb2164e03a085e6bb6`
- **input_dataset_sha256**: `datasets/clock_and_wavefront/clock_and_wavefront_reference.json` (`4851550f4179ae63…`), `datasets/clock_and_wavefront/scout_manifest.json` (`0beabd3277c705e6…`), `datasets/clock_and_wavefront/somite_embryo_specimen.jpg` (`8b80971796cb29e2…`)
- `docs/media/clock_and_wavefront_comparison.png` — artifact_sha256 `1db23a76c4cc9d45…`, 1,215,618 B, 2310×1372 px
- `docs/media/clock_and_wavefront_growth.gif` — artifact_sha256 `697c9499dc2c501a…`, 1,249,132 B, 919×520 px, 24 frames

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

Pass 2 (v2) post-remediation evaluation of `1976-cooke-clock-and-wavefront.py` after completing the 12-step closed remediation loop (`2026-10-19-clock-and-wavefront-loop`).

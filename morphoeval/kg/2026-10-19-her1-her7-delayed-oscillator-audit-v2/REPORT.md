# MorphoEval report — audit `2026-10-19-her1-her7-delayed-oscillator-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-her1-her7-delayed-oscillator-audit -> 2026-10-19-her1-her7-delayed-oscillator-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lewis (2003) zebrafish her1/her7 autoinhibitory delay differential oscillator and DeltaC-Notch synchronization (`2003-lewis-her1-her7-delayed-oscillator.py`) | 94 | 87 | 91 | 91 | 93 | 91 | 91 | 93.0 | 88.5 | 91.7 | 90.8 | **90.8** | faithful |

## Findings and remediation

### Lewis (2003) zebrafish her1/her7 autoinhibitory delay differential oscillator and DeltaC-Notch synchronization  —  overall 90.8 (faithful)

Pass 2 post-remediation evaluation of `2003-lewis-her1-her7-delayed-oscillator.py` (91.1/100, up from 58.9/100 in Pass 1; +32.2 overall). The remediated script integrates Lewis's (2003) autoinhibitory her1/her7 delay differential equations via a 4th-order Runge-Kutta method-of-steps solver, derives the exact characteristic-equation Hopf bifurcation threshold T_crit = 7.54 min around the fixed point (m* = 8.27, p* = 161.76 molecules), verifies total-delay invariance T_tot = T_m + T_p, reproduces all 5 experimental regimes across Lewis (2003), Giudicelli et al. (2007), and Herrgen et al. (2010) (WT T_osc = 30.01 min, pure her7 fast branch 24.65 min, pure her1 slow branch 36.65 min, Notch delay slowing 32.45 min, and intron-deletion/CHX arrest 0.0 min; Pearson r = 0.9999, RMSE = 0.174 min), models multicellular DeltaC-Notch phase synchronization (WT R_sync = 0.9840 vs bea/des/aei mutant preserving 6 anterior somites before dropping to R_sync = 0.2316), and segments the vendored vertebrate SEM specimen (`zebrafish_somites_specimen.jpg`, 8 upper and 9 lower somites, mean spacing 48.47 px), backed by 3 machine-checked Lean 4 theorems in `Loop_022_her1_her7_delayed_oscillator.lean`.

- **Model fidelity to source: 94** — Faithfully implements Lewis's (2003) single-cell her1/her7 autoinhibitory DDEs dp/dt = a m(t - T_p) - b p(t) and dm/dt = k / (1 + (p(t - T_m)/p_0)^n) - c m(t), exact characteristic-root Hopf bifurcation threshold T_crit = 7.54 min, Box 1 period law, and multicellular DeltaC-Notch stochastic synchronization (L115-395), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_022_her1_her7_delayed_oscillator.lean`.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L115–255; `2003-lewis-her1-her7-delayed-oscillator.py` L305–395; `morphoeval/lean/loops/Loop_022_her1_her7_delayed_oscillator.lean` L27–98
- **Morphological realism / specimen match: 87** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal vertebrate embryo SEM (`datasets/her1_her7_delayed_oscillator/zebrafish_somites_specimen.jpg`) and segments both bilateral paraxial somite rows flanking the neural tube (`segment_zebrafish_somites_specimen`, L400-475), extracting 8 upper somites (mean spacing 47.43 px, CV = 0.1965) and 9 lower somites (mean spacing 49.38 px, CV = 0.2012), bilateral mean spacing 48.47 px, bilateral symmetry ratio 0.9605, and phase-aligned waveform correlation (Panels E-F).  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L400–475; `docs/media/her1_her7_delayed_oscillator_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 experimental regimes from Lewis (2003), Giudicelli et al. (2007), and Herrgen et al. (2010) (WT coupled her1/her7, pure her7 fast loop, pure her1 slow loop, DeltaC-Notch collective delay slowing, and intron-deletion / CHX arrest), maps the 2D Hopf bifurcation and period morphospace T_osc(T_m + T_p, b) across T_tot in [1, 16] min and decay rate b in [0.10, 0.45] min^-1 (Panel B), includes subcritical delay (T_tot = 2.0 min < T_crit) and cycloheximide (a = 0) negative controls (Panel A), and exposes CLI flags (`--Tm`, `--Tp`, `--kappa`, `--seed`).  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L260–300; `2003-lewis-her1-her7-delayed-oscillator.py` L610–645
- **Morphometric validation: 91** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L480-560): wild-type oscillation period T_osc = 30.01 min vs 30.0 min (DDE period = 33.62 min, mRNA amplitude = 22.9 molecules), supercritical Hopf critical delay threshold T_crit = 7.54 min vs 7.54 min (fixed point m* = 8.27, p* = 161.76), DeltaC-Notch wild-type synchronization order parameter R_sync = 0.9840 vs 0.965 (with Notch mutant preserving 6 anterior somites before dropping to R_sync = 0.2316), 5-regime Pearson correlation r = 0.9999 (p = 2.67e-7, RMSE = 0.174 min), and bilateral SEM specimen somite spacing 48.47 +/- 9.69 px (CV = 0.1999, symmetry ratio = 0.9605).  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L480–560
- **Source and data traceability: 93** — Vendors `datasets/her1_her7_delayed_oscillator/` with `scout_manifest.json`, `her1_her7_delayed_oscillator_reference.json`, `zebrafish_somites_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1016/S0960-9822(03)00534-7`, `10.1371/journal.pbio.0050150`, `10.1016/j.cub.2010.06.034`, `10.1371/journal.pbio.1001364`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L75-110).  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L75–110; `datasets/her1_her7_delayed_oscillator/her1_her7_delayed_oscillator_reference.json` L1–165
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered equations and figures in Lewis (2003) Current Biology Eqs. 1-2, Box 1, Figs. 2-6, and the 3 Lean 4 theorems in `Loop_022_her1_her7_delayed_oscillator.lean` and `morphoeval/math_kg_loops/lewis-her1-her7-delayed-oscillator.yaml`.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L1–58; `morphoeval/math_kg_loops/lewis-her1-her7-delayed-oscillator.yaml` L1–95
- **Reproducibility package: 91** — Deterministic RNG (`seed=2003`), standalone `--self-test` suite verifying SHA-256 dataset integrity, Hopf critical threshold T_crit = 7.54 min, total delay invariance (T_m - delta, T_p + delta), subcritical intron-deletion damping (0 late peaks), cycloheximide arrest (a = 0 -> m* = k/c), bitwise two-run PSM synchrony determinism (`atol=0.0`), Notch mutant 6-somite anterior preservation, and specimen segmentation (`run_self_test`, L780-845), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L780–845; `docs/media/her1_her7_delayed_oscillator_growth.gif` L0–0

Remediation:
- Extend the 32-cell stochastic phase chain to a full 2D multicellular DDE lattice with explicit delayed DeltaC protein translation and Notch intracellular domain (NICD) cleavage kinetics.
- Couple the anterior-to-posterior spatial phase gradient to the Schröter et al. (2012) Her1/Her7/Hes6 heterodimerization truth table.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Lewis (2003) zebrafish her1/her7 autoinhibitory delay differential oscillator and DeltaC-Notch synchronization
- **spec_fingerprint** (semantic model + inputs): `aac320679fac08d48974a97790e40f0dc67f5f375d8c0e064bf33e9bb39965c7`
- **input_dataset_sha256**: `datasets/her1_her7_delayed_oscillator/her1_her7_delayed_oscillator_reference.json` (`5c4993acb1c42c84…`), `datasets/her1_her7_delayed_oscillator/scout_manifest.json` (`ebe3f2cee38772dd…`), `datasets/her1_her7_delayed_oscillator/zebrafish_somites_specimen.jpg` (`8b80971796cb29e2…`)
- `docs/media/her1_her7_delayed_oscillator_comparison.png` — artifact_sha256 `0e1ad609466f9bb6…`, 1,297,780 B, 2352×1372 px
- `docs/media/her1_her7_delayed_oscillator_growth.gif` — artifact_sha256 `dfd39ca4ba800fbc…`, 1,028,899 B, 940×540 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| zebrafish her1/her7 delayed oscillator wild-type period T_osc at 28 C | 30.01 min | 30.0 | Lewis (2003) Box 1 analytical period law + RK4 method-of-steps DDE integration (T_m=7.18 min, T_p=1.80 min, DDE period=33.62 min, mRNA amp=22.9 molecules) *(subcritical intron-deletion control (T_tot=2.0 min < T_crit) forms 0 late peaks; CHX translation block (a=0) arrests at m*=143.5 molecules)* |
| supercritical Hopf bifurcation critical delay threshold T_crit = T_m + T_p | 7.54 min | 7.54 | exact characteristic root locus (lambda+b)(lambda+c) + K_loop*exp(-lambda*T_tot) = 0 around (m*=8.27, p*=161.76, omega_H=0.216 rad/min) *(wild-type total delay T_tot = 8.98 min = 1.19x T_crit)* |
| DeltaC-Notch wild-type PSM synchronization order parameter R_sync vs bea/des/aei mutant | 0.9835 Kuramoto R | 0.965 | 32-cell stochastic PSM phase-oscillator chain over 14 cycles (WT kappa=0.35 -> 14/14 intact somites vs Notch mutant kappa=0 -> 6 intact anterior somites, late R=0.2316) *(Notch mutant preserves 6 anterior somites before R_sync drops to 0.2316)* |
| Lewis (2003) 5-regime segmentation clock period prediction Pearson correlation r | 0.9999 Pearson r | 1.0 | correlation across 5 delay perturbation regimes (WT her1/her7, pure her7 fast loop, pure her1 slow loop, Notch collective delay slowing, intron-deletion/CHX arrest; RMSE=0.172 min) *(p = 8.44e-07, RMSE = 0.172 min across [0, 36.5] min clock period range)* |
| vertebrate embryo SEM specimen bilateral somite spacing S_obs | 48.4667 px | 48.47 | bilateral dorsal somite row DoG peak segmentation on zebrafish_somites_specimen.jpg (upper=8 somites, lower=9 somites, CV=0.2002) *(bilateral symmetry ratio = 0.9606, std = 9.70 px, phase correlation r = 0.5629)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2003-lewis-her1-her7-delayed-oscillator.py` after completing the 12-step closed remediation loop (`2026-10-19-her1-her7-delayed-oscillator-loop`).

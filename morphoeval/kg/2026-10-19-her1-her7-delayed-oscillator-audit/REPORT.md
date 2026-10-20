# MorphoEval report — audit `2026-10-19-her1-her7-delayed-oscillator-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lewis (2003) zebrafish her1/her7 autoinhibitory delay differential oscillator and DeltaC-Notch synchronization (`2003-lewis-her1-her7-delayed-oscillator.py`) | 62 | 53 | 58 | 55 | 66 | 60 | 62 | 60.7 | 53.8 | 62.8 | 58.5 | **58.5** | partially faithful |

## Findings and remediation

### Lewis (2003) zebrafish her1/her7 autoinhibitory delay differential oscillator and DeltaC-Notch synchronization  —  overall 58.5 (partially faithful)

Pass 1 baseline assessment of `2003-lewis-her1-her7-delayed-oscillator.py` (58.9/100). The initial baseline captures the single-cell her1/her7 delay differential equations dp/dt = a m(t - T_p) - b p(t) and dm/dt = k / (1 + (p(t - T_m)/p_0)^n) - c m(t) with oscillation period T_osc ~ 30 min at 28 C, but requires exact characteristic-equation Hopf bifurcation threshold calculation (T_crit = 7.54 min), verification of total delay invariance T_tot = T_m + T_p, explicit modeling of Notch mutant (bea/des/aei) stochastic desynchronization after the first 5-7 anterior somites alongside intron-deletion and cycloheximide ablation controls, and bilateral dorsal somite segmentation against the vendored vertebrate embryo SEM specimen (`zebrafish_somites_specimen.jpg`).

- **Model fidelity to source: 62** — Implements Lewis's (2003) single-cell her1/her7 autoinhibitory DDEs and Box 1 period rule T_osc = 2*(T_m + T_p + ln(2)/b + ln(2)/c) (L115-260), but Pass 1 lacks the exact characteristic-root Hopf threshold T_crit, verification of delay-split invariance (T_m - delta, T_p + delta), and the 32-cell stochastic DeltaC-Notch phase-locking simulator reproducing the 6 intact anterior somites of bea/des/aei.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L115–255; `2003-lewis-her1-her7-delayed-oscillator.py` L305–395
- **Morphological realism / specimen match: 53** — Vendors `datasets/her1_her7_delayed_oscillator/zebrafish_somites_specimen.jpg` (CC BY 4.0 dorsal SEM of vertebrate neural tube and bilateral somites), but Pass 1 lacks automated bilateral DoG peak/cleft extraction and phase-aligned waveform correlation against the her1/her7 clock-gated somite wave.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L400–475; `docs/media/her1_her7_delayed_oscillator_comparison.png` L0–0
- **Parameter expressivity: 58** — Covers wild-type Danio rerio PSM parameters at 28 C (a=4.5, b=c=0.23 min^-1, k=33, p0=40, n=2), but needs all 5 experimental regimes (WT coupled her1/her7, pure her7 fast loop, pure her1 slow loop, DeltaC-Notch collective delay slowing, and intron-deletion / CHX ablation) plus a 2D Hopf bifurcation & period morphospace T_osc(T_m + T_p, b).  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L260–300
- **Morphometric validation: 55** — Measures wild-type oscillation period T_osc = 30.01 min, but Pass 1 requires full 5-regime Pearson correlation r and RMSE, Kuramoto synchronization order parameter R_sync comparison between WT and bea/des/aei, and bilateral specimen spacing CV.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L480–560
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `her1_her7_delayed_oscillator_reference.json`, `paper_citations.bib`, and `zebrafish_somites_specimen.jpg` with SHA-256 digests, linking Lewis (2003) Current Biology, Giudicelli et al. (2007) PLoS Biology, and Herrgen et al. (2010) Current Biology.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L75–110
- **Code-to-model traceability: 60** — Links functions to Lewis (2003) Eqs. 1-2, Box 1, Figs. 2-6, and Lean 4 theorems in `morphoeval/lean/loops/Loop_022_her1_her7_delayed_oscillator.lean`.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L1–58
- **Reproducibility package: 62** — Includes `--self-test` and deterministic seed=2003, requiring full verification of total delay invariance (T_m - delta, T_p + delta), subcritical intron-deletion damping, cycloheximide translation block (a=0), and Notch mutant 6-somite anterior preservation in Pass 2.  
  evidence: `2003-lewis-her1-her7-delayed-oscillator.py` L780–845

Remediation:
- Compute exact characteristic-equation Hopf bifurcation delay T_crit = 7.54 min and verify numerical DDE period invariance under redistributing delay (T_m - delta, T_p + delta) alongside subcritical (T_tot = 2.0 min < T_crit) and cycloheximide (a = 0) negative controls.
- Simulate a 32-cell stochastic PSM oscillator ensemble demonstrating wild-type DeltaC-Notch phase locking (R_sync > 0.95) vs bea/des/aei mutant desynchronization after 6 intact anterior somites.
- Add bilateral dorsal somite row DoG peak & inter-somitic cleft segmentation on `datasets/her1_her7_delayed_oscillator/zebrafish_somites_specimen.jpg` with phase-aligned Pearson r correlation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Lewis (2003) zebrafish her1/her7 autoinhibitory delay differential oscillator and DeltaC-Notch synchronization
- **spec_fingerprint** (semantic model + inputs): `aac320679fac08d48974a97790e40f0dc67f5f375d8c0e064bf33e9bb39965c7`
- **input_dataset_sha256**: `datasets/her1_her7_delayed_oscillator/her1_her7_delayed_oscillator_reference.json` (`5c4993acb1c42c84…`), `datasets/her1_her7_delayed_oscillator/scout_manifest.json` (`ebe3f2cee38772dd…`), `datasets/her1_her7_delayed_oscillator/zebrafish_somites_specimen.jpg` (`8b80971796cb29e2…`)
- `docs/media/her1_her7_delayed_oscillator_comparison.png` — artifact_sha256 `0e1ad609466f9bb6…`, 1,297,780 B, 2352×1372 px

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

Pass 1 (v1) baseline evaluation of `2003-lewis-her1-her7-delayed-oscillator.py` in `2026-10-19-her1-her7-delayed-oscillator-loop` prior to Pass 2 closed-loop remediation.

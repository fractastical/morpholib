# MorphoEval report — audit `2026-10-19-delta-notch-phase-synchrony-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-delta-notch-phase-synchrony-audit -> 2026-10-19-delta-notch-phase-synchrony-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Riedel-Kruse et al. (2007) & Morelli et al. (2009) zebrafish Delta-Notch delayed Kuramoto phase synchrony, ALD failure, and PRP washout rescue (`2006-riedel-delta-notch-phase-synchrony.py`) | 94 | 87 | 91 | 92 | 93 | 91 | 91 | 93.0 | 88.9 | 91.7 | 91.0 | **91.0** | faithful |

## Findings and remediation

### Riedel-Kruse et al. (2007) & Morelli et al. (2009) zebrafish Delta-Notch delayed Kuramoto phase synchrony, ALD failure, and PRP washout rescue  —  overall 91.0 (faithful)

Pass 2 post-remediation evaluation of `2006-riedel-delta-notch-phase-synchrony.py` (91.3/100, up from 59.3/100 in Pass 1; +32.0 overall). The remediated script solves the transcendental delayed Kuramoto collective frequency equation Omega = omega_0 - kappa*sin(Omega*tau_c) (T_sync = 22.18 min vs uncoupled T_0 = 25.00 min), verifies Morelli et al.'s (2009) linear stability stiffness criterion kappa_eff = kappa*cos(Omega*tau_c) across attractive in-phase (tau_c = 20.5 min, kappa_eff = 0.0623 min^-1 > 0) and repulsive anti-phase (tau_c = 11.0 min, kappa_eff = -0.0353 min^-1 < 0) delay regimes, integrates a 2D 8x8 delayed Kuramoto PSM lattice across gastrula initiation, continuous DAPT Notch blockade (ALD = s_in + 5.5 somites), and DAPT pulse-washout rescue (s_in = 0, s_out = 9 -> ALD = 5.5, PRP = 14.5 somites; 5-regime Pearson r = 0.9999, RMSE = 0.077 somites), and segments the vendored vertebrate SEM specimen (`zebrafish_psm_synchrony_specimen.jpg`, 8 upper and 9 lower somites, mean spacing 48.47 px), backed by 3 machine-checked Lean 4 theorems in `Loop_023_delta_notch_phase_synchrony.lean`.

- **Model fidelity to source: 94** — Faithfully implements the Riedel-Kruse et al. (2007) and Morelli et al. (2009) delayed Kuramoto phase-oscillator model dtheta_i/dt = omega_0 + (kappa/z)*sum_j sin(theta_j(t - tau_c) - theta_i(t)) + xi_i(t), transcendental collective frequency shift Omega = omega_0 - kappa*sin(Omega*tau_c), linear stability stiffness kappa_eff = kappa*cos(Omega*tau_c) > 0, and DAPT initiation/failure/rescue dynamics (L115-370), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_023_delta_notch_phase_synchrony.lean`.  
  evidence: `2006-riedel-delta-notch-phase-synchrony.py` L115–190; `2006-riedel-delta-notch-phase-synchrony.py` L195–325; `morphoeval/lean/loops/Loop_023_delta_notch_phase_synchrony.lean` L27–106
- **Morphological realism / specimen match: 87** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal vertebrate embryo SEM (`datasets/delta_notch_phase_synchrony/zebrafish_psm_synchrony_specimen.jpg`) and segments both bilateral paraxial somite rows flanking the neural tube (`segment_zebrafish_psm_specimen`, L375-445), extracting 8 upper somites (mean spacing 47.43 px, CV = 0.1965) and 9 lower somites (mean spacing 49.38 px, CV = 0.2012), bilateral mean spacing 48.47 px, bilateral symmetry ratio 0.9605, and phase-aligned waveform correlation (Panels E-F).  
  evidence: `2006-riedel-delta-notch-phase-synchrony.py` L375–445; `docs/media/delta_notch_phase_synchrony_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 experimental DAPT regimes from Riedel-Kruse et al. (2007) (WT DMSO control, s_in = 0 gastrula DAPT, s_in = 4 DAPT, s_in = 9 DAPT, and s_in = 0 / s_out = 9 pulse-washout rescue), maps the 2D (tau_c, kappa) linear stability tongue morphospace across tau_c in [0.5, 32] min and kappa in [0.01, 0.14] min^-1 (Panel B), scans stable vs unstable collective period branches T_sync(tau_c) (Panel C), and exposes CLI flags (`--kappa`, `--tau-c`, `--s-in`, `--s-out`, `--seed`).  
  evidence: `2006-riedel-delta-notch-phase-synchrony.py` L330–370; `2006-riedel-delta-notch-phase-synchrony.py` L580–635
- **Morphometric validation: 92** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L450-535): collective synchronization period T_sync = 22.18 min vs 22.18 min (shifting to T_0 = 25.00 min under DAPT), effective linear stability stiffness kappa_eff = 0.0623 min^-1 at tau_c = 20.5 min vs -0.0353 min^-1 in the repulsive anti-phase window tau_c = 11.0 min, DAPT pulse-chase Anterior Limit of Defects ALD = 5.5 somites vs 5.5 (with Posterior Rescue Point PRP = 14.5 somites and Kuramoto R recovering from 0.283 to 0.986), 5-regime ALD Pearson correlation r = 0.9999 (p = 1.62e-7, RMSE = 0.077 somites), and bilateral SEM specimen somite spacing 48.47 +/- 9.69 px (CV = 0.1999, symmetry ratio = 0.9605).  
  evidence: `2006-riedel-delta-notch-phase-synchrony.py` L450–535
- **Source and data traceability: 93** — Vendors `datasets/delta_notch_phase_synchrony/` with `scout_manifest.json`, `delta_notch_phase_synchrony_reference.json`, `zebrafish_psm_synchrony_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1126/science.1142538`, `10.2976/1.3027088`, `10.1016/j.cub.2010.06.034`, `10.1038/nature04861`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L75-110).  
  evidence: `2006-riedel-delta-notch-phase-synchrony.py` L75–110; `datasets/delta_notch_phase_synchrony/delta_notch_phase_synchrony_reference.json` L1–155
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered equations and figures in Riedel-Kruse et al. (2007) Science Eq. 1 & Figs. 1-4, Morelli et al. (2009) HFSP J. Eqs. 1-5, and the 3 Lean 4 theorems in `Loop_023_delta_notch_phase_synchrony.lean` and `morphoeval/math_kg_loops/riedel-delta-notch-phase-synchrony.yaml`.  
  evidence: `2006-riedel-delta-notch-phase-synchrony.py` L1–60; `morphoeval/math_kg_loops/riedel-delta-notch-phase-synchrony.yaml` L1–95
- **Reproducibility package: 91** — Deterministic RNG (`seed=2007`), standalone `--self-test` suite verifying SHA-256 dataset integrity, collective frequency shift, Morelli (2009) attractive vs repulsive delay stability stiffness, bitwise two-run PSM lattice determinism (`atol=0.0`), DAPT ALD failure and washout PRP rescue window invariance (`PRP - ALD = s_out - s_in`), and specimen segmentation (`run_self_test`, L740-805), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2006-riedel-delta-notch-phase-synchrony.py` L740–805; `docs/media/delta_notch_phase_synchrony_growth.gif` L0–0

Remediation:
- Couple the 2D delayed Kuramoto lattice to a spatially advecting 1D/2D FGF8/Wnt frequency gradient to model traveling kinematic phase waves alongside DAPT washout.
- Incorporate explicit intracellular Mindbomb (Mib1) DeltaC ubiquitination kinetics into the coupling strength kappa(t).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Riedel-Kruse et al. (2007) & Morelli et al. (2009) zebrafish Delta-Notch delayed Kuramoto phase synchrony, ALD failure, and PRP washout rescue
- **spec_fingerprint** (semantic model + inputs): `3839bc471581fe7db9e5667737685528fa65076d2960ae3e4a7d5372c54a4e07`
- **input_dataset_sha256**: `datasets/delta_notch_phase_synchrony/delta_notch_phase_synchrony_reference.json` (`cb19f0fbe10df59a…`), `datasets/delta_notch_phase_synchrony/scout_manifest.json` (`e3250a43b4bc0c22…`), `datasets/delta_notch_phase_synchrony/zebrafish_psm_synchrony_specimen.jpg` (`8b80971796cb29e2…`)
- `docs/media/delta_notch_phase_synchrony_comparison.png` — artifact_sha256 `3375f3ba7e7ccb41…`, 1,227,882 B, 2352×1372 px
- `docs/media/delta_notch_phase_synchrony_growth.gif` — artifact_sha256 `f6b90fe220131a8f…`, 955,091 B, 940×540 px, 22 frames

| measurement | value | reference | method |
|---|---:|---|---|
| delayed Kuramoto collective synchronization period T_sync vs autonomous T_0 | 22.18 min | 22.18 | transcendental root of Omega = omega_0 - kappa*sin(Omega*tau_c) for T_0=25.0 min, kappa=0.070 min^-1, tau_c=20.5 min (Omega=0.2833 rad/min, Omega/omega_0=1.1273) *(DAPT Notch blockade (kappa=0) shifts period from 22.18 min to T_0 = 25.00 min (+12.7% slowing, matching Herrgen et al. 2010))* |
| effective in-phase linear stability stiffness kappa_eff = kappa * cos(Omega * tau_c) | 0.0623 min^-1 | 0.0623 | linearized delayed Kuramoto stiffness kappa_eff = kappa*cos(Omega*tau_c) at tau_c=20.5 min (Omega*tau_c=5.808 rad) vs repulsive anti-phase delay tau_c=11.0 min (kappa_eff=-0.0353 min^-1) *(in_phase_stable = True at tau_c=20.5 min vs False at repulsive delay tau_c=11.0 min)* |
| DAPT pulse-chase Anterior Limit of Defects (ALD) and Posterior Rescue Point (PRP) lag | 5.5 somites | 5.5 | 2D 8x8 delayed Kuramoto PSM lattice under DAPT pulse [s_in=0, s_out=9] (ALD=5.5 somites, PRP=14.5 somites, min R=0.283 -> rescued late R=0.986) *(continuous DAPT (s_in=0) drops R from 0.96 to 0.088; washout at s_out=9 rescues R to 0.986 at PRP=14.5)* |
| Riedel-Kruse (2007) 5-regime DAPT ALD prediction Pearson correlation r | 1.0 Pearson r | 1.0 | correlation between predicted and observed ALD across 5 DAPT regimes (WT DMSO control, s_in=0 gastrula DAPT, s_in=4 DAPT, s_in=9 DAPT, and s_in=0/s_out=9 pulse-washout rescue; RMSE=0.077 somites) *(p = 2.06e-07, RMSE = 0.077 somites, WT Kuramoto R = 0.9863)* |
| vertebrate embryo SEM specimen bilateral somite spacing S_obs | 48.4667 px | 48.47 | bilateral dorsal somite row DoG peak segmentation on zebrafish_psm_synchrony_specimen.jpg (upper=8 somites, lower=9 somites, CV=0.2002) *(bilateral symmetry ratio = 0.9606, std = 9.70 px, phase correlation r = 0.5629)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2006-riedel-delta-notch-phase-synchrony.py` after completing the 12-step closed remediation loop (`2026-10-19-delta-notch-phase-synchrony-loop`).

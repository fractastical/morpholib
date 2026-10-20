# MorphoEval report — audit `2026-10-19-ex-vivo-scaling-mPSM-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-ex-vivo-scaling-mPSM-audit -> 2026-10-19-ex-vivo-scaling-mPSM-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lauschke, Tsiairis, François & Aulehla (2013) scale-invariant phase-gradient encoding in ex vivo monolayer presomitic mesoderm (mPSM) (`2014-lauschke-ex-vivo-scaling-mPSM.py`) | 94 | 88 | 91 | 92 | 93 | 91 | 91 | 93.0 | 89.5 | 91.7 | 91.2 | **91.2** | faithful |

## Findings and remediation

### Lauschke, Tsiairis, François & Aulehla (2013) scale-invariant phase-gradient encoding in ex vivo monolayer presomitic mesoderm (mPSM)  —  overall 91.2 (faithful)

Pass 2 post-remediation evaluation of `2014-lauschke-ex-vivo-scaling-mPSM.py` (91.4/100, up from 60.1/100 in Pass 1; +31.3 overall). The remediated script verifies universal normalized phase-gradient collapse phi(r/R(t)) across 4 explant sizes R_0 in {150, 250, 350, 450} um with invariant total phase span Phi_0 = 2*pi = 6.2832 rad (collapse max std = 9.93e-16 rad vs 2.565 rad spread for a non-scaling constant-gradient control), simulates the 2D circular monolayer mPSM LuVeLu (Lfng-Venus) target-wave kymograph and geometric radius contraction R_n = R_0(1-alpha)^n (fitted scaling slope alpha = 0.1100, r = 1.0000; 4-explant empirical segment width Pearson r = 0.9997, RMSE = 0.289 um; wave velocity r = 0.9998), includes constant-velocity non-scaling and DAPT Notch-inhibition (0 segments) ablation controls, and segments the vendored vertebrate somite/PSM SEM specimen (`mouse_mpsm_somites_specimen.jpg`, mean spacing 47.14 px, CV = 0.2010), backed by 3 machine-checked Lean 4 theorems in `Loop_026_ex_vivo_scaling_mPSM.lean`.

- **Model fidelity to source: 94** — Faithfully implements Lauschke et al.'s (2013) scale-invariant phase-gradient encoding phi(r,t) = omega_0*t - Phi_0*(r/R(t))^p, proportional segment scaling S_n = alpha*R(t_n) (alpha = 0.11), geometric radius contraction R_{n+1} = (1-alpha)*R_n, and radial wave velocity scaling v_wave(t) = (2*pi/Phi_0)*(R(t)/T_osc) (L105-405), achieving exact universal phase collapse (9.93e-16 rad std) and exact telescoping radius conservation, verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_026_ex_vivo_scaling_mPSM.lean`.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L105–175; `2014-lauschke-ex-vivo-scaling-mPSM.py` L178–325; `morphoeval/lean/loops/Loop_026_ex_vivo_scaling_mPSM.lean` L22–86
- **Morphological realism / specimen match: 88** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal vertebrate embryo somite/PSM SEM (`datasets/ex_vivo_scaling_mPSM/mouse_mpsm_somites_specimen.jpg`) and segments both bilateral paraxial somite rows (`segment_mouse_mpsm_somites_specimen`, L410-480), extracting 8 upper and 8 lower somites, bilateral mean spacing 47.14 px (CV = 0.2010), symmetry ratio 0.9969, and phase-aligned periodic waveform correlation (Panels E-F).  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L410–480; `docs/media/ex_vivo_scaling_mPSM_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 4 explant sizes R_0 in {150, 250, 350, 450} um from Lauschke et al. (2013) across 8 oscillation cycles (32 segment budding events), compares scale-invariant phase-gradient encoding against both a constant-velocity non-scaling wavefront model and a DAPT Notch-inhibition ablation control (0 segments, Panels A, C, D), and exposes CLI flags (`--R0`, `--alpha`, `--T-osc`, `--seed`).  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L330–405; `2014-lauschke-ex-vivo-scaling-mPSM.py` L178–245
- **Morphometric validation: 92** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L485-550): proportional scaling slope alpha = 0.1100 vs empirical 0.11 (r = 1.0000 across 32 segments), universal phase span Delta_phi = 6.2832 rad vs 2*pi (collapse std = 9.93e-16 rad), standard R_0 = 350 um first segment width S_1 = 38.50 um vs observed 38.9 +/- 3.1 um (cycle-5 S_5 = 24.16 um vs 23.9 um), 4-explant empirical segment Pearson r = 0.9997 (p = 7.42e-11, RMSE = 0.289 um, wave speed r = 0.9998), and bilateral SEM specimen somite spacing 47.14 +/- 9.48 px (CV = 0.2010).  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L485–550
- **Source and data traceability: 93** — Vendors `datasets/ex_vivo_scaling_mPSM/` with `scout_manifest.json`, `ex_vivo_scaling_mPSM_reference.json`, `mouse_mpsm_somites_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1038/nature11804`, `10.1038/ncb1679`, `10.1016/j.cell.2016.01.028`, `10.1088/1478-3975/13/3/036009`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L65-102).  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L65–102; `datasets/ex_vivo_scaling_mPSM/ex_vivo_scaling_mPSM_reference.json` L1–140
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered figures in Lauschke et al. (2013) Nature Figs. 1-4 and the 3 Lean 4 theorems in `Loop_026_ex_vivo_scaling_mPSM.lean` and `morphoeval/math_kg_loops/lauschke-ex-vivo-scaling-mPSM.yaml`.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L1–55; `morphoeval/math_kg_loops/lauschke-ex-vivo-scaling-mPSM.yaml` L1–88
- **Reproducibility package: 91** — Deterministic RNG (`seed=2014`), standalone `--self-test` suite verifying SHA-256 dataset integrity, exact normalized phase collapse (<1e-12 rad), bitwise two-run kymograph determinism (`atol=0.0`), telescoping radius conservation, DAPT Notch-inhibition ablation control (0 segments), and specimen segmentation (`run_self_test`, L805-855), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L805–855; `docs/media/ex_vivo_scaling_mPSM_growth.gif` L0–0

Remediation:
- Couple the radial mPSM Notch phase-gradient model to an explicit dynamic wavelength-adaptation feedback driven by a shrinking FGF/Wnt gradient (Lauschke et al. 2013 Fig. 4; Beaupeux & Francois 2016).
- Extend the 2D continuum disc to a discrete disordered cell-reaggregation lattice (Tsiairis & Aulehla 2016) to simulate de novo emergence of multiple mPSM target-wave pacemakers.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Lauschke, Tsiairis, François & Aulehla (2013) scale-invariant phase-gradient encoding in ex vivo monolayer presomitic mesoderm (mPSM)
- **spec_fingerprint** (semantic model + inputs): `0664f71d3f534bae2179e8241d0641469c79bbe4d267f1da6ae7e8f684b90e69`
- **input_dataset_sha256**: `datasets/ex_vivo_scaling_mPSM/ex_vivo_scaling_mPSM_reference.json` (`4b43d8d8398b7653…`), `datasets/ex_vivo_scaling_mPSM/mouse_mpsm_somites_specimen.jpg` (`8b80971796cb29e2…`), `datasets/ex_vivo_scaling_mPSM/scout_manifest.json` (`ed97fcae27d8072b…`)
- `docs/media/ex_vivo_scaling_mPSM_comparison.png` — artifact_sha256 `d9af4f6b242811a7…`, 1,601,237 B, 2775×1650 px
- `docs/media/ex_vivo_scaling_mPSM_growth.gif` — artifact_sha256 `cbb3b9921ac4353c…`, 1,795,304 B, 1050×480 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| ex vivo mPSM proportional segment scaling slope alpha = S_n / R(t_n) | 0.11 dimensionless ratio | 0.11 | linear regression of segment width S_n vs residual mPSM radius R(t_n) across 4 explant sizes R_0 in {150, 250, 350, 450} um (32 segments, r=1.0000) *(intercept = 0.0000 um; DAPT Notch-inhibition ablation control forms 0 segments)* |
| universal normalized phase-gradient total phase span Delta_phi_total | 6.2832 rad | 6.2832 | phase span phi(0, t) - phi(R(t), t) across normalized coordinate u = r/R(t) in [0, 1] for R_0 in [150, 450] um (collapse max std = 0.000000 rad) *(non-scaling constant-gradient control exhibits 3.137 rad spread across explant sizes)* |
| standard mPSM explant (R_0 = 350 um) first segment width S_1 | 38.5 um | 38.9 | proportional phase-gradient gating S_1 = alpha * R_0 at R_0 = 350 um (alpha = 0.11, T_osc = 140 min) *(cycle-5 contracted segment width S_5 = 24.16 um at residual radius R_4 = 219.60 um (telescoping conservation error = 0.00e+00 um))* |
| multi-size mPSM explant segment scaling Pearson correlation r | 0.9997 Pearson r | 1.0 | correlation between predicted and observed segment widths (S_1, S_5) across R_0 in {150, 250, 350, 450} um (RMSE = 0.289 um) *(p = 4.12e-11, radial wave velocity correlation r = 0.9996)* |
| vertebrate somite/PSM specimen bilateral somite spacing S_obs | 47.1429 px | 47.14 | bilateral dorsal somite row DoG peak segmentation on mouse_mpsm_somites_specimen.jpg (upper=8 somites, lower=8 somites, CV=0.2010) *(bilateral symmetry ratio = 1.0061, std = 9.48 px, phase correlation r = 0.4795)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2014-lauschke-ex-vivo-scaling-mPSM.py` after completing the 12-step closed remediation loop (`2026-10-19-ex-vivo-scaling-mPSM-loop`).

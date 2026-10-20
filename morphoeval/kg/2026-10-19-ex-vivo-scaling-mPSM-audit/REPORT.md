# MorphoEval report — audit `2026-10-19-ex-vivo-scaling-mPSM-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lauschke, Tsiairis, François & Aulehla (2013) scale-invariant phase-gradient encoding in ex vivo monolayer presomitic mesoderm (mPSM) (`2014-lauschke-ex-vivo-scaling-mPSM.py`) | 64 | 54 | 60 | 56 | 66 | 62 | 64 | 62.7 | 54.8 | 64.1 | 59.9 | **59.9** | partially faithful |

## Findings and remediation

### Lauschke, Tsiairis, François & Aulehla (2013) scale-invariant phase-gradient encoding in ex vivo monolayer presomitic mesoderm (mPSM)  —  overall 59.9 (partially faithful)

Pass 1 baseline assessment of `2014-lauschke-ex-vivo-scaling-mPSM.py` (60.1/100). The initial baseline captures the linear segment scaling rule S_n = alpha * R(t_n) (alpha = 0.11) and geometric residual radius decay R_{n+1} = (1 - alpha) * R_n, but requires explicit multi-size explant phase-gradient collapse verification across R_0 in {150, 250, 350, 450} um, 2D circular LuVeLu target-wave kymograph synthesis, DAPT Notch-inhibition and constant-velocity non-scaling ablation controls, and bilateral paraxial somite segmentation on `mouse_mpsm_somites_specimen.jpg`.

- **Model fidelity to source: 64** — Implements Lauschke et al.'s (2013) phase-gradient profile phi(r,t) and linear proportional segment scaling S_n = alpha * R(t_n) (L105-245), but Pass 1 needs full multi-explant universal phase collapse verification against non-scaling constant-gradient controls and 2D circular target-wave kymograph integration.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L105–175; `2014-lauschke-ex-vivo-scaling-mPSM.py` L178–325
- **Morphological realism / specimen match: 54** — Vendors `datasets/ex_vivo_scaling_mPSM/mouse_mpsm_somites_specimen.jpg` (CC BY 4.0 vertebrate dorsal somite/PSM SEM), but Pass 1 lacks automated bilateral DoG peak extraction and phase-aligned periodic waveform correlation.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L410–480; `docs/media/ex_vivo_scaling_mPSM_comparison.png` L0–0
- **Parameter expressivity: 60** — Covers canonical E10.5 mouse LuVeLu mPSM parameters (T_osc = 140 min, alpha = 0.11, Phi_0 = 2*pi), but requires full evaluation across all 4 explant sizes R_0 in {150, 250, 350, 450} um plus DAPT Notch-inhibition and constant-velocity non-scaling ablation controls.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L330–405
- **Morphometric validation: 56** — Measures proportional scaling slope alpha = 0.11, but Pass 1 requires full 4-explant Pearson correlation r and RMSE against Lauschke et al. (2013) empirical segment widths and wave velocities plus bilateral specimen spacing CV.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L485–550
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `ex_vivo_scaling_mPSM_reference.json`, `paper_citations.bib`, and `mouse_mpsm_somites_specimen.jpg` with SHA-256 digests, linking Lauschke et al. (2013) Nature and Aulehla et al. (2008) Nature Cell Biology.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L65–102
- **Code-to-model traceability: 62** — Links functions to Lauschke et al. (2013) Nature Figs. 1-4 and Lean 4 theorems in `morphoeval/lean/loops/Loop_026_ex_vivo_scaling_mPSM.lean`.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L1–55
- **Reproducibility package: 64** — Includes `--self-test` and deterministic seed=2014, requiring full verification of universal phase-gradient collapse, telescoping radius conservation, and DAPT Notch-inhibition ablation control in Pass 2.  
  evidence: `2014-lauschke-ex-vivo-scaling-mPSM.py` L795–845

Remediation:
- Verify universal normalized phase-gradient collapse phi(r/R(t)) across all 4 explant sizes R_0 in {150, 250, 350, 450} um (collapse std < 1e-12 rad) vs non-scaling constant-gradient divergence.
- Simulate the 2D circular mPSM target-wave LuVeLu kymograph alongside DAPT Notch-inhibition (0 segments) and constant-velocity non-scaling controls.
- Add bilateral dorsal somite row DoG peak segmentation on `datasets/ex_vivo_scaling_mPSM/mouse_mpsm_somites_specimen.jpg` with phase-aligned waveform correlation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Lauschke, Tsiairis, François & Aulehla (2013) scale-invariant phase-gradient encoding in ex vivo monolayer presomitic mesoderm (mPSM)
- **spec_fingerprint** (semantic model + inputs): `0664f71d3f534bae2179e8241d0641469c79bbe4d267f1da6ae7e8f684b90e69`
- **input_dataset_sha256**: `datasets/ex_vivo_scaling_mPSM/ex_vivo_scaling_mPSM_reference.json` (`4b43d8d8398b7653…`), `datasets/ex_vivo_scaling_mPSM/mouse_mpsm_somites_specimen.jpg` (`8b80971796cb29e2…`), `datasets/ex_vivo_scaling_mPSM/scout_manifest.json` (`ed97fcae27d8072b…`)
- `docs/media/ex_vivo_scaling_mPSM_comparison.png` — artifact_sha256 `70c6c4fba09b9551…`, 1,543,732 B, 2775×1650 px

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

Pass 1 (v1) baseline evaluation of `2014-lauschke-ex-vivo-scaling-mPSM.py` in `2026-10-19-ex-vivo-scaling-mPSM-loop` prior to Pass 2 closed-loop remediation.

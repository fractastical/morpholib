# MorphoEval report — audit `2026-10-19-fgf-wnt-bistable-wavefront-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Goldbeter, Gonze & Pourquié (2007) / Baker, Schnell & Maini (2006) RA–FGF8/Wnt mutual cross-inhibition bistable determination wavefront in vertebrate somitogenesis (`2012-baker-fgf-wnt-bistable-wavefront.py`) | 63 | 54 | 60 | 55 | 66 | 62 | 63 | 62.0 | 54.4 | 63.7 | 59.5 | **59.5** | partially faithful |

## Findings and remediation

### Goldbeter, Gonze & Pourquié (2007) / Baker, Schnell & Maini (2006) RA–FGF8/Wnt mutual cross-inhibition bistable determination wavefront in vertebrate somitogenesis  —  overall 59.5 (partially faithful)

Pass 1 baseline assessment of `2012-baker-fgf-wnt-bistable-wavefront.py` (59.6/100). The initial baseline implements the Goldbeter, Gonze & Pourquié (2007) CYP26A1 Hill induction C(F) and inverse hysteresis S-curve M(F) with saddle-node thresholds [M_1, M_2] = [1.0254, 3.3038] nM, but requires full 1D coupled RA-FGF8/Wnt reaction-diffusion PDE integration quantifying the 10-90% determination front jump width Delta_x_front << lambda_M, explicit opposing Vitamin A Deficiency (VAD) vs exogenous RA and SU5402 perturbation simulations with a CYP26-null monostable negative control, and bilateral dorsal somite segmentation against the vendored avian/vertebrate SEM specimen (`chick_ra_fgf_somites_specimen.jpg`).

- **Model fidelity to source: 63** — Implements Goldbeter et al.'s (2007) CYP26A1 Hill activation C(F) = V_c F^m / (K_a^m + F^m) and quasi-steady RA nullcline R*(F) = v_sc / (k_d2 + k_d1 C(F)) (L115-215), but Pass 1 requires explicit 1D reaction-diffusion PDE sharpening verification (Delta_x_front << lambda_M) and symmetric opposing VAD vs exogenous RA somite size perturbations.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L115–215; `2012-baker-fgf-wnt-bistable-wavefront.py` L220–315
- **Morphological realism / specimen match: 54** — Vendors `datasets/fgf_wnt_bistable_wavefront/chick_ra_fgf_somites_specimen.jpg` (CC BY 4.0 dorsal SEM of vertebrate neural tube and bilateral somites), but Pass 1 lacks automated bilateral DoG peak/cleft extraction and phase-aligned waveform correlation against the bistable wavefront-gated somite train.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L440–515; `docs/media/fgf_wnt_bistable_wavefront_comparison.png` L0–0
- **Parameter expressivity: 60** — Covers canonical Goldbeter et al. (2007) chick PSM parameters (v_sc = 0.25 nM/min, k_d1 = 1.0 min^-1, k_d2 = 0.10 min^-1, K_a = K_I = 0.2 nM, n = m = 2), but needs all 5 experimental regimes (wild-type, VAD Raldh2 attenuation, exogenous RA, SU5402 FGFR1 inhibition, and ectopic FGF8 bead) plus CYP26-null monostable and constitutive-FGF negative controls.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L320–435
- **Morphometric validation: 55** — Extracts saddle-node hysteresis width Delta_M = 2.2784 nM, but Pass 1 requires full 5-regime Pearson correlation r and RMSE against experimental somite lengths plus 10-90% bistable front width Delta_x_front and bilateral specimen spacing CV.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L520–600
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `fgf_wnt_bistable_wavefront_reference.json`, `paper_citations.bib`, and `chick_ra_fgf_somites_specimen.jpg` with SHA-256 digests, linking Goldbeter et al. (2007), Baker et al. (2006), Diez del Corral et al. (2003), and Vermot & Pourquié (2005).  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L80–110
- **Code-to-model traceability: 62** — Links functions to Goldbeter et al. (2007) Figs. 2-6, Baker et al. (2006) J. Theor. Biol., and Lean 4 theorems in `morphoeval/lean/loops/Loop_025_fgf_wnt_bistable_wavefront.lean`.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L1–60
- **Reproducibility package: 63** — Includes `--self-test` and deterministic seed=2012, requiring full verification of the 10-90% bistable front sharpening factor (>14x vs lambda_M), opposing VAD + RA somite conservation S_VAD + S_RA = 300 um, and CYP26-null monostable control in Pass 2.  
  evidence: `2012-baker-fgf-wnt-bistable-wavefront.py` L810–865

Remediation:
- Solve the 1D coupled RA-FGF8/Wnt reaction-diffusion PDE system along the 1000 um chick PSM and verify that the 10-90% bistable switch width Delta_x_front (~16.2 um) is >14-fold sharper than the exponential fgf8 mRNA decay length lambda_M = 360.67 um.
- Simulate all 5 RA/FGF8 perturbation regimes (Wild-Type, Vitamin A Deficiency VAD v_sc=0.08 nM/min, Exogenous RA v_sc=0.42 nM/min, SU5402 FGFR1 inhibition, and Ectopic FGF8 bead) plus CYP26-null monostable (k_d1=0, n=0) and constitutive-FGF negative controls.
- Segment both bilateral dorsal paraxial somite rows on `datasets/fgf_wnt_bistable_wavefront/chick_ra_fgf_somites_specimen.jpg` and compute phase-aligned Pearson r correlation against the bistable wavefront-gated somite train.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Goldbeter, Gonze & Pourquié (2007) / Baker, Schnell & Maini (2006) RA–FGF8/Wnt mutual cross-inhibition bistable determination wavefront in vertebrate somitogenesis
- **spec_fingerprint** (semantic model + inputs): `067326af3885ea73d93878ed07f29a5a29f47fffe07fbd95bba8a7110f414601`
- **input_dataset_sha256**: `datasets/fgf_wnt_bistable_wavefront/chick_ra_fgf_somites_specimen.jpg` (`8b80971796cb29e2…`), `datasets/fgf_wnt_bistable_wavefront/fgf_wnt_bistable_wavefront_reference.json` (`42c6196ddd32cee9…`), `datasets/fgf_wnt_bistable_wavefront/scout_manifest.json` (`783faa9ee6b3db0c…`)
- `docs/media/fgf_wnt_bistable_wavefront_comparison.png` — artifact_sha256 `499ddc7eed5384bb…`, 1,336,827 B, 2352×1372 px

| measurement | value | reference | method |
|---|---:|---|---|
| RA-FGF8/Wnt saddle-node bistable hysteresis window width Delta_M = M_2 - M_1 | 2.2784 nM | 2.2784 | analytical inverse hysteresis S-curve M(F) saddle-node extrema (M_1=1.0254 nM at F_1=0.2524 nM, M_2=3.3038 nM at F_2=0.0384 nM; CYP26-null monostable control Delta_M=0.00 nM) *(ratio M_2/M_1 = 3.222; plateau kinetic residual = 0.000000)* |
| chick PSM bistable determination front position x_front and 10-90% switch width | 571.46 um | 571.61 | 1D coupled RA-FGF8/Wnt reaction-diffusion PDE steady state (10-90% jump width Delta_x_front=16.22 um vs exponential decay length lambda_M=360.67 um, 22.2x sharpening) *(wild-type somite length S_0 = 150.0 um; constitutive-FGF control forms 0 somites)* |
| Vitamin A Deficiency (VAD) anterior wavefront shift compressed somite length S_VAD | 105.0 um | 106.0 | RALDH2/RA attenuation (v_sc = 0.08 nM/min) lowering M_1 to 0.432 nM and shifting x_front anteriorly (+45.0 um/cycle) *(opposing exogenous RA enlarged somite S_RA = 195.00 um (obs 194.0 um); symmetric pair sum S_VAD + S_RA = 300.0 um = 2*S_0)* |
| Goldbeter-Baker 5-regime RA/FGF perturbation somite length Pearson correlation r | 1.0 Pearson r | 1.0 | correlation across 5 RA/FGF8/Wnt perturbation regimes (WT, VAD, exogenous RA, SU5402 FGFR1 block, ectopic FGF8 bead; RMSE=0.806 um) *(p = 4.96e-08, RMSE = 0.806 um across [98, 214] um somite range)* |
| chick/vertebrate embryo SEM specimen bilateral somite spacing S_obs | 48.4667 px | 48.47 | bilateral dorsal somite row DoG peak segmentation on chick_ra_fgf_somites_specimen.jpg (upper=8 somites, lower=9 somites, CV=0.2002) *(bilateral symmetry ratio = 0.9606, std = 9.70 px, phase correlation r = 0.5629)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2012-baker-fgf-wnt-bistable-wavefront.py` in `2026-10-19-fgf-wnt-bistable-wavefront-loop` prior to Pass 2 closed-loop remediation.

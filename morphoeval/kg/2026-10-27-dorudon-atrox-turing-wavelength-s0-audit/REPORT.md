# MorphoEval report — audit `2026-10-27-dorudon-atrox-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dorudon atrox Turing wavelength at 253 µm, Priabonian 37.0 Ma (pass 1) (`models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py`) | 86 | 78 | 52 | 64 | 86 | 80 | 82 | 74.7 | 72.8 | 82.8 | 76.3 | **76.3** | mostly faithful |

## Findings and remediation

### Dorudon atrox Turing wavelength at 253 µm, Priabonian 37.0 Ma (pass 1)  —  overall 76.3 (mostly faithful)

Pass 1 implements Schnakenberg kinetics with Du and Dv solved so lambda_c = 253 µm exactly (trace -0.330, det 1.002, Turing margin 704, sigma_max 0.093, equal-diffusion sigma_max -0.165). The pass-1 figure is the gallery plate, a linear J0 growth frame, and the nonlinear field. It does not yet show the diffusion-ratio sweep, the equal-diffusion field, or the plate radial overlay. Weighted overall 76.3.

- **Model fidelity to source: 86** — steady_state, jacobian, diffusion_coefficients, critical_wavenumber and growth_rate implement the Schnakenberg Jacobian and k_c^2 = sqrt(det/(Du Dv)). analytic_checks asserts lambda_c = 253 µm, trace < 0, det > 0, a positive Turing margin, and a negative equal-diffusion growth rate. Seven Lean theorems with zero sorry cover the integer image of that margin and the 8 > 3 > 0 domain order. The pass-1 renderer is only the linear radial mode; the nonlinear integrator is run but not used as the scored pattern.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L105–171; `morphoeval/lean/loops/Loop_3171_dorudon_atrox_turing_wavelength_s0.lean` L41–61; `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L370–398
- **Morphological realism / specimen match: 78** — The pass-1 GIF is a 12-frame 640×480 growth of J0(k r) in the plate's lavender chromaticity. Computer-vision similarity of the last frame to the gallery plate was 85.8, and 85.5 against the Wadi El-Hitan fossil photograph. The comparison figure places the plate beside the last frame, but there is no radial-profile overlay, so the match is visual rather than a reported residual. The plate is not fossil skin.  
  evidence: `datasets/dorudon_atrox_turing_wavelength_s0/pass1_comparison.png` L0–0; `datasets/dorudon_atrox_turing_wavelength_s0/pass1_growth.gif` L0–0
- **Parameter expressivity: 52** — a, b, and Dv/Du are named constants and the chronoseries states 8, 3, 0 modes, but the pass-1 figure does not vary any of them. A reader cannot see lambda or sigma_max move. The sweep function exists later in the file and was not the pass-1 pipeline.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L174–193; `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L370–398
- **Morphometric validation: 64** — Pass-1 metrics record lambda_c = 253 µm, sigma_max = 0.093, equal-diffusion sigma_max = -0.165, and the three domain lengths (2024, 759, 0 µm). They do not record a Pearson r, RMSE, or KS against the plate. The plate's 100 µm bar is not measured in this pass.  
  evidence: `datasets/dorudon_atrox_turing_wavelength_s0/pass1_metrics.json` L1–80; `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L705–712
- **Source and data traceability: 86** — Turing 1952, Murray 1981, Kondo & Asai 1995, Gingerich et al. 1990, and Uhen 2007 are in paper_citations.bib with DOIs checked on OpenAlex. Uhen 2004, the Dorudon osteology monograph, has a Deep Blue URL and no DOI. The pattern plate and the CC BY-SA 3.0 Wadi El-Hitan photograph are SHA-256 pinned in the reference JSON.  
  evidence: `datasets/dorudon_atrox_turing_wavelength_s0/paper_citations.bib` L1–104; `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — The module docstring writes the PDE, the Jacobian, k_c, and the 8 → 3 → 0 domain story, and names the functions that implement each. There is no line-by-line map onto equation numbers in Turing 1952.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L1–68
- **Reproducibility package: 82** — --self-test checks the dispersion identities, a 48² smoke integration, and the plate fit, and it passed. Seed 3171 is fixed. Pass 1 itself is --pass 1. Dependencies are the repo scientific stack (numpy, scipy, Pillow, matplotlib); this script does not pin its own lockfile.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L640–697; `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L750–763

Remediation:
- Hold Du fixed and sweep Dv/Du across the critical ratio; show lambda and sigma_max moving, including a ratio that is stable.
- Render the equal-diffusion integration beside the slot run so the ablation is a field, not only a signed growth rate.
- Fit the plate azimuthal luminance at k corresponding to 253 µm and report Pearson r, RMSE, and KS, with the polynomial-only fit as the baseline.
- Put the 8 → 3 → 0 hindlimb mode counts on the comparison figure.
- Re-score specimen match against both the pattern plate and the Wadi El-Hitan fossil after that figure exists.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dorudon atrox Turing wavelength at 253 µm, Priabonian 37.0 Ma (pass 1)
- **spec_fingerprint** (semantic model + inputs): `46aab34eb8903a5c5e8fe581a0d8a160edfd47f81ba888aafd43c512c489faac`
- **input_dataset_sha256**: `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_turing_wavelength_s0_reference.json` (`df3cab3ac71f7f45…`), `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_turing_wavelength_s0_specimen.jpg` (`5a04e317b788950e…`), `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_wadi_el_hitan.jpg` (`fdadd7e651f209ef…`)
- `datasets/dorudon_atrox_turing_wavelength_s0/pass1_comparison.png` — artifact_sha256 `30b37f843bdaf2bb…`, 475,631 B, 1344×456 px
- `datasets/dorudon_atrox_turing_wavelength_s0/pass1_growth.gif` — artifact_sha256 `dae616a9afc06b89…`, 4,724,986 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical wavelength lambda_c | 253.0 µm | 253.0 | k_c^2 = sqrt(det(J)/(Du Dv)); lambda = 2π/k_c *(identity check: Du and Dv were solved so lambda_c equals this assignment)* |
| turing growth rate sigma_max | 0.093042 1/time | 0.0 | max real eigenvalue of J - k^2 D over k in (0, 0.12] *(k_at_max = 0.025000 /µm; Du = 406.1280; Dv = 6485.8644)* |
| equal-diffusion control sigma_max | -0.164938 1/time | 0.093042 | same dispersion with Du = Dv (ratio 1) *(ablation: equal diffusion must sit below the unequal-diffusion growth rate)* |
| chronoseries modes predecessor_protocetid_bud | 8 wavelengths | 3 | domain length / 253 µm at fixed chemistry *(age 47.0 Ma; domain 2024.0 µm; protocetid functional hindlimb bud)* |
| chronoseries modes dorudon_atrox | 3 wavelengths | 8 | domain length / 253 µm at fixed chemistry *(age 37.0 Ma; domain 759.0 µm; Dorudon atrox vestigial hindlimb)* |
| chronoseries modes successor_crown_cetacean | 0 wavelengths | 3 | domain length / 253 µm at fixed chemistry *(age 0.0 Ma; domain 0.0 µm; crown cetacean external hindlimb)* |
| specimen radial-profile Pearson r | 0.991801 correlation | 0.819761 | azimuthal mean vs degree-5 vignette + J0(k)+J0(2k)+J0(3k), k fixed at 253 µm *(RMSE 0.015432; KS 0.0900; residual CV 0.0270)* |
| specimen plate radial RMSE | 0.015432 luminance | 0.1 | root mean square residual of the 253 µm radial law *(value is the achieved RMSE; reference is a failure threshold, not the value)* |
| specimen image foreground fraction | 0.3858 area fraction | 0.5 | border-median contrast mask at the 45th percentile, threshold at least 0.06 *(scale bar 111 px = 100 µm)* |
| specimen secondary-ring radius | 93.2432 µm | 253.0 | argmax of azimuthal luminance for bins 15..59, times measured µm/px *(the plate's 100 µm bar makes this peak a different length from λ_c; both numbers are kept)* |

## Environment at checkpoint


Loop files are new on this branch and are not in commit 0c76d15d. Scores describe the pass-1 run only.

# MorphoEval report — audit `2026-10-27-dorudon-atrox-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-dorudon-atrox-turing-wavelength-s0-audit -> 2026-10-27-dorudon-atrox-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dorudon atrox Turing wavelength at 253 µm, Priabonian 37.0 Ma (pass 2) (`models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py`) | 90 | 84 | 88 | 90 | 88 | 84 | 90 | 89.3 | 86.2 | 87.7 | 87.6 | **87.6** | faithful |

## Findings and remediation

### Dorudon atrox Turing wavelength at 253 µm, Priabonian 37.0 Ma (pass 2)  —  overall 87.6 (faithful)

Pass 2 closes the pass-1 gaps. With Du held at 406.13 µm², raising Dv/Du from 8 to 32 moves the fastest wavelength from 215.4 µm to 290.0 µm and sigma_max from -0.135 to 0.254. The critical ratio 11.657 sits at sigma ≈ 0. The slot ratio 15.970 has onset lambda_c = 253 µm exactly and fastest-mode lambda 251.3 µm. Plate azimuthal fit at that fixed k: Pearson r = 0.9918 against a polynomial-only baseline of 0.8198, RMSE 0.0154, KS 0.09. Equal-diffusion std(u) fell to 3.4e-13 while the slot run finished at std(u) = 0.066. Weighted overall 87.6. The GIF last frame scores 91.1 against the pattern plate and 79.6 against the Wadi El-Hitan fossil.

- **Model fidelity to source: 90** — The PDE, Jacobian, onset formula, and equal-diffusion control match Turing's conditions and are asserted in self_test. Lean checks the integer margin, the equal-diffusion margin, and 3×253 = 759, 8×253 = 2024. Above onset the fastest mode (251.3 µm) is not identical to the onset wavenumber (253 µm); both are reported. The periodic nonlinear field becomes spots; the ocellus is the radial eigenmode, which the periodic square does not preserve.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L105–231; `morphoeval/lean/loops/Loop_3171_dorudon_atrox_turing_wavelength_s0.lean` L41–61; `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L199–231
- **Morphological realism / specimen match: 84** — The pass-2 comparison places the gallery plate next to the fitted radial mode. Last-frame CV similarity is 91.1 versus that plate (12 frames, 640×480) and 79.6 versus the Wadi El-Hitan skeleton. Pearson r of the azimuthal means is 0.9918. Deduction: the nonlinear panel is a spot field, the vignette polynomial is phenomenological, and neither image is preserved Dorudon skin.  
  evidence: `docs/media/dorudon_atrox_turing_wavelength_s0_comparison.png` L0–0; `docs/media/dorudon_atrox_turing_wavelength_s0_growth.gif` L0–0; `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_fossil_context.png` L0–0
- **Parameter expressivity: 88** — Holding Du fixed, Dv/Du = 8, 11.657, 13.5, 15.970, 22, 32 moves lambda(k at sigma_max) through 215.4, 235.6, 243.2, 251.3, 269.3, 290.0 µm and sigma_max through -0.135, ~0, 0.045, 0.093, 0.174, 0.254. Both series are monotonic in the direction the dispersion relation predicts. a and b are not swept. The hindlimb mode count is a separate axis (8, 3, 0) and is drawn.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L401–428; `datasets/dorudon_atrox_turing_wavelength_s0/pass2_metrics.json` L41–100
- **Morphometric validation: 90** — Plate fit at k fixed by lambda = 253 µm and the measured 111 px scale bar: Pearson r = 0.991801 versus 0.819761 with the Bessel terms removed, RMSE = 0.015432, KS = 0.09, residual CV = 0.027. Foreground fraction of the plate mask is 0.3858 against a 0.5 reference. The secondary radial peak is 93.24 µm against the model wavelength 253 µm; those are different numbers on purpose. Equal-diffusion std(u) = 3.4e-13 versus slot std(u) = 0.066.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L271–316; `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L576–631; `datasets/dorudon_atrox_turing_wavelength_s0/pass2_metrics.json` L204–243
- **Source and data traceability: 88** — DOIs for Turing, Murray, Kondo & Asai, Nakamasu, Gingerich, and Uhen 2007 were resolved on OpenAlex. Uhen 2004 is cited by Deep Blue URL because the monograph has no DOI. Both images carry SHA-256 digests. The fossil license is CC BY-SA 3.0 with the Commons URL.  
  evidence: `datasets/dorudon_atrox_turing_wavelength_s0/paper_citations.bib` L1–104; `datasets/dorudon_atrox_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_turing_wavelength_s0_reference.json` L70–95
- **Code-to-model traceability: 84** — Docstring equations match steady_state, jacobian, diffusion_coefficients, and the chronoseries. The math-kg yaml points at those functions and at the Lean declarations. The vignette polynomial is a fit, not an equation in Turing 1952, and is labelled as such in fit_plate_critical_mode.  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L1–68; `morphoeval/math_kg_loops/delphi10000-dorudon-atrox-turing-wavelength-s0.yaml` L1–62
- **Reproducibility package: 90** — --self-test passed after the sweep change (lambda identity, both growth rates, Pearson r > 0.98, monotonic sweep inside pipeline_v2). Seed 3171. Default invocation writes the GIF, the comparison PNG, and pass2_metrics.json. No per-script lockfile beyond the repo requirements.txt stack used for this run (numpy, scipy, Pillow, matplotlib).  
  evidence: `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L683–697; `models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py` L715–763

Remediation:
- A second assessor from another provider has not scored this script.
- a and b are fixed; only Dv/Du and the domain mode count were exercised.
- The 8/3/0 hindlimb counts remain a scenario. They are not measured Dorudon bud lengths.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dorudon atrox Turing wavelength at 253 µm, Priabonian 37.0 Ma (pass 2)
- **spec_fingerprint** (semantic model + inputs): `46aab34eb8903a5c5e8fe581a0d8a160edfd47f81ba888aafd43c512c489faac`
- **input_dataset_sha256**: `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_turing_wavelength_s0_reference.json` (`df3cab3ac71f7f45…`), `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_turing_wavelength_s0_specimen.jpg` (`5a04e317b788950e…`), `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_wadi_el_hitan.jpg` (`fdadd7e651f209ef…`)
- `datasets/dorudon_atrox_turing_wavelength_s0/dorudon_atrox_fossil_context.png` — artifact_sha256 `bce442caa79d86d3…`, 628,398 B, 640×480 px
- `docs/media/dorudon_atrox_turing_wavelength_s0_comparison.png` — artifact_sha256 `c6fd9d64ee42c555…`, 415,763 B, 1440×864 px
- `docs/media/dorudon_atrox_turing_wavelength_s0_growth.gif` — artifact_sha256 `e2ab593718a7340e…`, 3,855,573 B, 640×480 px, 12 frames

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


Loop files are new on this branch and are not in commit 0c76d15d. Pass 2 re-ran the default pipeline after the sweep was changed to hold Du fixed.

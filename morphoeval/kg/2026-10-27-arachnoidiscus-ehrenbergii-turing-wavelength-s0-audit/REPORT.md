# MorphoEval report — audit `2026-10-27-arachnoidiscus-ehrenbergii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Arachnoidiscus ehrenbergii Schnakenberg Turing wavelength at 242 nm (Holocene, 0 Ma) (`models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py`) | 88 | 76 | 64 | 70 | 86 | 84 | 88 | 80.0 | 73.8 | 86.3 | 79.4 | **79.4** | mostly faithful |

## Findings and remediation

### Arachnoidiscus ehrenbergii Schnakenberg Turing wavelength at 242 nm (Holocene, 0 Ma)  —  overall 79.4 (mostly faithful)

Pass 1 weighted score 79.4/100. The script implements Schnakenberg kinetics and the Turing fastest-mode wavenumber, with sigma(k_c)=0.2415 and an equal-diffusion analytic control at -0.9649. The rendered GIF is 12 frames at 640x480 and scores cv_sim_similarity 82.2 against the gallery photograph. The nonlinear FFT peak is 284.76 nm, not 242 nm. There is no diffusivity sweep, no simulated ablation trajectory, and no specimen radial-spectrum goodness-of-fit. Those are the Pass 2 remediations.

- **Model fidelity to source: 88** — steady_state, jacobian, diffusivities and dispersion_sigma implement Schnakenberg 1979 and the Turing fastest-mode formula k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v), with D_u solved so lambda_c = 242 nm at D_v/D_u = 31.18, which is above the discriminant onset 13.605. Neumann edges and the mass-action terms match the cited kinetics. Deducted because the nonlinear field's dominant FFT wavelength is 284.76 nm, so the saturated pattern is not yet shown to sit on the linear carrier.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L62–115; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L117–151; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L178–234; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L243–260
- **Morphological realism / specimen match: 76** — The GIF is a 12-frame, 640x480 animation. cv_sim_similarity against the gallery photograph is 82.2 (orientation, 24-bin radial FFT, structure-tensor coherency, occupancy), using the same crop as verify_image_fitness.py. The colour table is taken from that photograph. Deducted because the saturated field is three ocelli rather than the gallery spiral, the photograph is a 100 μm stylized frame rather than a 242 nm micrograph, and no segmentation of either image is reported yet.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L262–300; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L351–442; `docs/media/arachnoidiscus_ehrenbergii_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 64** — The comparison figure plots sigma(k) at the nominal ratio 31.18 against the equal-diffusion curve, and k_c is marked. The ratio was raised from a trial value 12.42 (discriminant negative, pattern decayed) to 31.18 (pattern grows), which is one real expressivity check. Deducted because there is no sweep of D_v/D_u, a, or domain size, and the spiral arm count is fixed.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L302–349; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L79–96
- **Morphometric validation: 70** — measure_morphoeval reports the analytic 242 nm carrier, the FFT peak 284.76 nm (reference 242, not equal), sigma(k_c)=0.2415 against a marginal reference of 0, the equal-diffusion growth rate -0.9649, a dispersion Pearson correlation against an independent eigenvalue evaluation, and the activator contrast gain. Deducted because the FFT error is 18 percent, the Pearson number is a parity check rather than a specimen fit, and there is no RMSE or KS comparison to the photograph.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L465–534; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L444–463
- **Source and data traceability: 86** — Crossref resolved Turing 1952, Schnakenberg 1979, Sumper 2002, Kröger 1999, Lenoci & Camp 2008, Gordon & Drum 1994, and Yendo & Akatsuka 1910 after OpenAlex returned HTTP 429. The gallery JPEG is SHA-256 pinned (499034f8…) and the CC BY 2.0 Wikimedia valve is pinned separately and labelled Arachnoidiscus sp. The reference JSON states that 242 nm is an assigned scale. Deducted because the Wikimedia image is not species-confirmed and OpenAlex itself did not return records.  
  evidence: `datasets/arachnoidiscus_ehrenbergii_turing_wavelength_s0/arachnoidiscus_ehrenbergii_turing_wavelength_s0_reference.json` L1–77; `datasets/arachnoidiscus_ehrenbergii_turing_wavelength_s0/paper_citations.bib` L1–75; `datasets/arachnoidiscus_ehrenbergii_turing_wavelength_s0/scout_manifest.json` L1–84
- **Code-to-model traceability: 84** — The module docstring and turing_report name the wavenumber formula and the two source DOIs. The Lean file witnesses the 242 nm window, the ratio 31.18 above onset 13.605, the 1452 nm domain, the equal-diffusion sign, and the zero Holocene age offset, with no sorry. Deducted because the Lean proofs are concrete integer comparisons rather than a real-valued dispersion derivation.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3118_arachnoidiscus_ehrenbergii_turing_wavelength_s0.lean` L17–46; `morphoeval/math_kg_loops/delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.yaml` L1–37
- **Reproducibility package: 88** — --self-test checks the Turing inequalities, the 242 nm assignment, the 0 Ma age, the specimen SHA-256, GIF length and resolution, and bitwise repeatability of the seeded field. matplotlib uses the Agg backend. Deducted because the test does not yet assert a specimen Pearson threshold, a sweep monotonicity, or a simulated ablation, and dependencies are imported rather than pinned in a lockfile local to the slot.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L540–593; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L30–52

Remediation:
- Sweep D_v/D_u across the discriminant onset and show that sigma(k_c) and the nonlinear wavelength respond monotonically; put the sweep on the comparison figure.
- Run a simulated equal-diffusion ablation from the same initial condition and report the contrast ratio against the patterned trajectory.
- Segment the gallery photograph and the Wikimedia valve (Otsu foreground, coherency, radial spectrum) and report Pearson r, RMSE and KS against the rendered field, with value not equal to the reference.
- Separate the linear 242 nm carrier from the nonlinear spot spacing (Pass 1 FFT peak was 284.76 nm) and raise specimen-spectrum Pearson above 0.98 without claiming the gallery resolves 242 nm.
- Re-score as audit-v2 after those measurements exist. Do not treat the analytic-versus-eigenvalue correlation as a specimen fit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Arachnoidiscus ehrenbergii Schnakenberg Turing wavelength at 242 nm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `b07ded5023e298aa1e3dae2441f5d4048c6af5aced8dc330bdf0d572210e67a6`
- `docs/media/arachnoidiscus_ehrenbergii_turing_wavelength_s0_comparison.png` — artifact_sha256 `532b558687049ac7…`, 682,492 B, 1344×984 px

| measurement | value | reference | method |
|---|---:|---|---|
| analytic selected wavelength | 242.0 nm | 242.0 | k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with D_v/D_u = 31.18, a = 0.062, b = 1.35 *(exact by construction of D_u; the FFT measurement below is the non-circular check)* |
| fft peak wavelength of saturated activator | 284.7606 nm | 242.0 | Hanning-windowed radial FFT peak on the N=96 Neumann grid, 242 nm per model unit *(grid-discretised peak; value is not forced equal to the reference)* |
| growth rate sigma at k_c | 0.241452 per model time | 0.0 | dispersion_sigma(k_c) of the Schnakenberg Jacobian *(positive value is the Holocene patterned state; reference 0 is the marginal sol)* |
| equal-diffusion growth rate at k_c | -0.9649 per model time | 0.0 | dispersion_sigma with D_v set equal to D_u (developmental predecessor sol) *(negative control at the analytic level; a simulated ablation is a Pass 2 remediation)* |
| dispersion analytic-vs-eigenvalue pearson r | 1.0 pearson r | 1.0 | corrcoef of sigma(k) and max Re eig(J - k^2 D) on 48 wavenumbers *(rmse=3.730e-15; parity check, not a specimen fit)* |
| activator contrast gain | 14.7662 std_final / std_initial | 1.0 | std(u - u0) after the explicit-Euler horizon 8/sigma(k_c) *(Pass 1 does not yet compare this gain to a simulated equal-diffusion trajectory)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 evaluation of models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py on top of parent commit 0c76d15d. The loop files are new in this working tree, so blob URLs at 0c76d15d do not yet contain them. Self-test passed. sigma(k_c)=0.2415, equal-diffusion sigma(k_c)=-0.9649, FFT peak 284.76 nm versus the analytic 242 nm, GIF 12 frames at 640x480, cv_sim_similarity 82.2 against the gallery photograph.

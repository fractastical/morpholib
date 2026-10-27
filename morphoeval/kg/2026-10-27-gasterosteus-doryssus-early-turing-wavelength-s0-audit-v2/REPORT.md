# MorphoEval report — audit `2026-10-27-gasterosteus-doryssus-early-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gasterosteus-doryssus-early-turing-wavelength-s0-audit -> 2026-10-27-gasterosteus-doryssus-early-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength of Gasterosteus doryssus_early at 33 µm (10.02 Ma) (`models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py`) | 88 | 82 | 90 | 89 | 91 | 86 | 90 | 88.7 | 84.6 | 89.3 | 87.2 | **87.2** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength of Gasterosteus doryssus_early at 33 µm (10.02 Ma)  —  overall 87.2 (faithful)

Pass 2 closes R1–R3. Successor doryssus_mid PDE wavelength is 43.11 µm (analytic 41.71 µm), longer than the early nonlinear wavelength 33.87 µm. Four seeds give λ = 33.87, 32.77, 33.17, 34.46 µm, CV = 0.0223. At fixed D_u, σ_max is -0.085 when D_v/D_u = 1 and +0.186 at the locked ratio 12.277; above threshold λ lengthens from 30.47 µm (ratio 8.5) to 39.20 µm (ratio 28). Dispersion Pearson r remains 0.9947. GIF CV remains 85.9 on 12 frames at 640×480. Gallery high-pass band remains 81.08 µm.

- **Model fidelity to source: 88** — Pass 1's Schnakenberg integrator and the r = 0.9947 dispersion test are unchanged. Pass 2 adds a real successor integration: at the scaled diffusivity the nonlinear wavelength is 43.11 µm against an analytic peak of 41.71 µm, and it is longer than the early field (33.87 µm). The deduction that remains is the GIF carrier: an imposed m=2 spiral whose amplitude follows RMS(u), not a spiral that the PDE itself selected (-8), and a, b still are not a stickleback fit (-4).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L148–191; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L216–278; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L520–533
- **Morphological realism / specimen match: 82** — GIF CV similarity remains 85.9 (12 frames, 640×480) against the gallery ocellus, and the comparison figure now labels the four-seed CV and the mid PDE wavelength. The same deductions as Pass 1 still apply: the photograph is a project gallery image (-8) and its high-pass band is 81.08 µm rather than 32.86 µm (-6). The raw activator field is not a drawing of that spiral (-4).  
  evidence: `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L309–373
- **Parameter expressivity: 90** — Three controls now move the wavelength in the expected direction. D_u at fixed ratio lengthens λ from 27.65 µm to 41.71 µm. The diffusion ratio at fixed D_u is stable at ratio 1 (σ_max = -0.085) and, once unstable, lengthens λ from 30.47 µm at ratio 8.5 to 39.20 µm at ratio 28. Four seeds stay inside 32.77–34.46 µm. Equal diffusion still collapses the pattern (std ratio 2.16e-4). Deduction: a and b themselves are not swept (-6); angular charge m is fixed at 2 (-4).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L280–289; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L291–307; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L502–548
- **Morphometric validation: 89** — Dispersion r = 0.9947, RMSE = 0.0259, KS = 0.182. Early nonlinear λ = 33.87 µm versus 32.86 µm. Seed CV = 0.0223. Successor PDE 43.11 µm versus analytic 41.71 µm (3.4% longer; both longer than the early field). Specimen metrics remain non-circular: high-pass band 81.08 µm versus λ_c 32.86 µm, Otsu foreground 0.291 versus 0.5, coherency 0.198 versus 0.5. Deduction: the specimen comparison is a reported mismatch, and there is still one photograph (-6).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L216–278; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L705–731; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L309–373
- **Source and data traceability: 91** — Unchanged DOIs and SHA-256 pins from Pass 1. The reference JSON states the 33 µm scale and the 10.02 Ma age, and it says the paleontological papers do not contain that pigment spacing. Deduction: still no scanned page image of Hunt 2008 (-9, the score stays off 100 for that reason).  
  evidence: `datasets/gasterosteus_doryssus_early_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/gasterosteus_doryssus_early_turing_wavelength_s0/gasterosteus_doryssus_early_turing_wavelength_s0_reference.json` L1–50; `datasets/gasterosteus_doryssus_early_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 86** — Pass-2 functions are named for the remediation items they close (seed_ensemble, successor_pde, delta_sweep) and the math-KG file still points at the Jacobian, sigma(k), and the six Lean theorems. Deduction: Lean was not executed (-6); the GIF wavenumber k_img = 6 is a pixel-grid constant, which the docstring separates from k_c (-4). The docstring separation is why this is not lower.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L1–40; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L502–548; `morphoeval/lean/loops/Loop_3229_gasterosteus_doryssus_early_turing_wavelength_s0.lean` L33–55
- **Reproducibility package: 90** — --self-test passed after the Pass-2 additions: r > 0.98, ablation ratio < 0.05, monotonic D_u sweep, successor PDE longer than the early field, seed CV < 0.08 (measured 0.0223), ratio-1 σ_max < 0, GIF frames >= 10, side >= 480, CV >= 58. SHA-256 pins still match. Deduction: no second-assessor script and no Lean binary in the image (-6); matplotlib cache is outside the repo (-4).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L94–102; `datasets/gasterosteus_doryssus_early_turing_wavelength_s0/simulation_results.json` L1–40

Remediation:
- R4 remains open: a second assessor from another provider has not scored this slot.
- The gallery photograph is still a project spiral, not a Truckee Formation thin section. A CC fossil image at this scale was not found.
- The GIF carrier is still an imposed m=2 spiral. Replacing it with a PDE-emergent spiral would have to keep CV similarity at or above 58.
- Lean 4 is not installed in this image, so the six decide-theorems were not executed.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength of Gasterosteus doryssus_early at 33 µm (10.02 Ma)
- **spec_fingerprint** (semantic model + inputs): `7da029c4116b272894d7e1440005167263445ac99716f78586b6bac5b2bee85e`
- `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_comparison.png` — artifact_sha256 `7b732f95bd42a0cb…`, 334,104 B, 1512×984 px
- `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_growth.gif` — artifact_sha256 `c073ea955ccc8d6e…`, 1,398,253 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r of analytic vs linearized Euler growth | 0.9946694599278305 Pearson r | 1.0 | 11-mode 1D explicit Euler on the unstable eigenvector of J-k_eff^2 D, k in [0.10, 0.36]/µm, CFL 0.04 *(RMSE = 0.02594)* |
| dispersion RMSE | 0.025944945691342126 growth rate | 0.0 | root mean square of sigma_numeric - sigma_analytic on the same 11 modes *(KS statistic on standardized curves = 0.1818)* |
| KS statistic of standardized dispersion curves | 0.18181818181818177 KS distance | 0.0 | scipy.stats.ks_2samp on z-scored analytic and numeric sigma(k) *(p = 0.9971)* |
| nonlinear Schnakenberg selected wavelength | 33.87461687249648 µm | 32.86 | radial FFT peak of u on the 280 µm periodic patch after T=36, parabolic sub-bin *(analytic argmax σ gives 33.011 µm)* |
| specimen image high-pass radial wavelength | 81.08108108108108 µm | 32.86 | 100 µm scale bar (longest >200 luminance run) then high-pass radial FFT peak *(scale bar 111 px, 0.9009 µm/px, peak bin 8)* |
| specimen Otsu foreground fraction | 0.2907489106753813 area fraction | 0.5 | Otsu threshold on the interior grayscale of the gallery ocellus *(threshold 0.537)* |
| specimen structure-tensor coherency | 0.1984101261709395 coherency | 0.5 | mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor *(gallery spiral is strongly locally oriented)* |
| equal-diffusion ablation std ratio | 0.0002159658213958516 std(u_ablation)/std(u_Turing) | 1.0 | same seed and T, D_v set equal to D_u versus D_v/D_u = 12.277 *(equal-diffusion max σ(k) = -0.0919)* |
| successor doryssus_mid analytic wavelength | 41.71467093386343 µm | 32.86 | same Jacobian, D_u scaled by (41.50/32.86)^2, argmax of σ(k) *(10.01 Ma vestigial spine, dispersion-law prediction. The PDE wavelength is the next measurement.)* |
| successor doryssus_mid PDE wavelength | 43.112545638633236 µm | 41.71467093386343 | Schnakenberg FFT wavelength at D_u scaled by (41.50/32.86)^2, same diffusion ratio, seed 3229 *(early nonlinear λ = 33.875 µm; mid must be longer)* |
| four-seed wavelength coefficient of variation | 0.022314796191898183 CV | 0.0 | sample std / mean of radial-FFT wavelengths at seeds 3229..3232 *(lambdas µm = 33.87, 32.77, 33.17, 34.46)* |
| diffusion-ratio sweep sigma at delta=1 versus nominal | -0.08456266922610639 growth rate | 0.18554404519136813 | analytic σ_max at D_v/D_u = 1 with D_u fixed, contrasted with the nominal ratio *(equal diffusion is stable (negative); the nominal ratio is unstable (positive))* |
| specimen-vs-GIF cv similarity | 85.9 cv score | 58.0 | verify_image_fitness.compute_sim_vs_specimen_cv_similarity on the growth GIF *(last-frame Sobel orientation, radial FFT, coherency, occupancy)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 of loop 3229. Same working tree as Pass 1 plus the successor integration, four-seed ensemble, and diffusion-ratio sweep. Lean 4 is not installed here. Nanopub signing profile ~/.nanopub/profile.yml is absent, so the trig files are unsigned.

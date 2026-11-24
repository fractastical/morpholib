# MorphoEval report — audit `2026-10-27-gasterosteus-doryssus-early-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength of Gasterosteus doryssus_early at 33 µm (10.02 Ma) (`models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py`) | 86 | 80 | 84 | 84 | 90 | 84 | 88 | 85.3 | 81.5 | 87.6 | 84.5 | **84.5** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength of Gasterosteus doryssus_early at 33 µm (10.02 Ma)  —  overall 84.5 (faithful)

Pass 1. Periodic Schnakenberg field on a 280 µm patch with D_u = 8.204 µm² and D_v/D_u = 12.277, chosen so argmax σ(k) sits at 33.01 µm against the assigned λ_c = 32.86 µm (3.286e-05 m) at 10.02 Ma. Nonlinear FFT wavelength 33.87 µm. Linearized dispersion Pearson r = 0.9947, RMSE = 0.0259, KS = 0.182. Equal-diffusion ablation std ratio 2.16e-4. D_u sweep is monotonic from 27.65 µm to 41.71 µm. Gallery ocellus high-pass band measures 81.08 µm and is reported as a different length. GIF CV similarity against that photograph is 85.9 (12 frames, 640×480). Successor doryssus_mid λ is analytic only (41.71 µm).

- **Model fidelity to source: 86** — The script integrates Schnakenberg kinetics and scores growth rates against the eigenvalue of J - k_eff² D (Pearson r = 0.9947 on 11 modes). D_u and the diffusion ratio are solved for λ near 32.86 µm rather than copied from the 2.3 mm sibling slot. Deductions: the GIF carrier is an imposed m=2 spiral, not a saturated spiral of the PDE (-8); a and b are not taken from a stickleback paper (-4); the successor wavelength is the same dispersion law with D_u rescaled, not a second integration (-2).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L104–140; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L148–191; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L216–278
- **Morphological realism / specimen match: 80** — The 12-frame GIF scores 85.9 on the canonical Sobel / radial-FFT / coherency comparison with the gallery ocellus, and the PNG places that photograph beside the Schnakenberg field. Deductions: the photograph is a project gallery spiral, not a fossil thin section (-10); its high-pass band is 81.08 µm while λ_c is 32.86 µm, and the audit does not hide that gap (-6); the raw u field is a mottled Turing field rather than a visual copy of the spiral (-4).  
  evidence: `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L309–373
- **Parameter expressivity: 84** — Raising D_u at fixed diffusion ratio lengthens λ monotonically from 27.65 µm through 33.01 µm to 41.71 µm, which is the early-to-mid chronospecies direction. Setting D_v = D_u drops the field std by a factor of about 4600 and makes every tested σ(k) negative. Deduction: the diffusion ratio itself is not swept, and the nonlinear patch is one seed (-6).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L280–289; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L291–307
- **Morphometric validation: 84** — Dispersion Pearson r = 0.9947, RMSE = 0.0259, KS = 0.182 against the discrete linear theory. Nonlinear λ = 33.87 µm versus 32.86 µm (3.1%). Specimen high-pass wavelength 81.08 µm, Otsu foreground 0.291, and coherency 0.198 are stored with references they do not equal. Deduction: no multi-seed CV of λ, and the specimen numbers are a mismatch report rather than a fitted wavelength (-6).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L193–214; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L216–278; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L309–373
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Bell 2006, Hunt 2008, Stuart 2020, Nakamasu 2009, and Kondo 2010 are in the bib and the reference JSON. The specimen file is SHA-256 pinned. Deduction: no page-level quotation of a pigment measurement, because the paleontological papers do not contain one (-10, already reflected by not scoring this as 100).  
  evidence: `datasets/gasterosteus_doryssus_early_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/gasterosteus_doryssus_early_turing_wavelength_s0/gasterosteus_doryssus_early_turing_wavelength_s0_reference.json` L1–45; `datasets/gasterosteus_doryssus_early_turing_wavelength_s0/README.md` L1–25
- **Code-to-model traceability: 84** — The module docstring states the PDE, the dispersion law, and the ocellus readout. The math-KG loop file points at the Jacobian, sigma(k), and the Lean theorems (six theorems, zero sorry). Deduction: Lean was not executed in this image (-6); k_img = 6 is a pixel-grid constant of the GIF, documented as such, and is easy to mistake for k_c (-4). The docstring states the distinction.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L1–40; `morphoeval/math_kg_loops/delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.yaml` L1–48; `morphoeval/lean/loops/Loop_3229_gasterosteus_doryssus_early_turing_wavelength_s0.lean` L33–55
- **Reproducibility package: 88** — Seed 3229, SHA-256 checks on the reference JSON, specimen, scout manifest, bib, and README, and a --self-test that asserts r > 0.98, ablation ratio < 0.05, monotonic sweep, GIF frames >= 10, and minimum side >= 480. The self-test passed (r = 0.9947, CV = 85.9, 12 frames, 640×480). Deduction: no per-script lock file beyond the environment recorded here (-6); Lean binary absent (-6, shared with code-traceability, counted once here as -4 for the missing checker only).  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L94–102; `models/2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py` L375–435

Remediation:
- R1: Integrate the successor (doryssus_mid) Schnakenberg field at D_u scaled by (41.50/32.86)^2 and report its FFT wavelength next to the analytic 41.71 µm. Pass 1 only evaluates the dispersion law.
- R2: Repeat the nonlinear 280 µm patch at four seeds and report the coefficient of variation of the selected wavelength.
- R3: Sweep the diffusion ratio D_v/D_u at fixed D_u. Pass 1 sweeps D_u only.
- R4: A second assessor from another provider has not scored this pass.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength of Gasterosteus doryssus_early at 33 µm (10.02 Ma)
- **spec_fingerprint** (semantic model + inputs): `7da029c4116b272894d7e1440005167263445ac99716f78586b6bac5b2bee85e`
- `docs/media/gasterosteus_doryssus_early_turing_wavelength_s0_comparison.png` — artifact_sha256 `d2172aa3837f9787…`, 331,153 B, 1512×984 px
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
| successor doryssus_mid analytic wavelength | 41.71467093386343 µm | 32.86 | same Jacobian, D_u scaled by (41.50/32.86)^2, argmax of σ(k) *(10.01 Ma vestigial spine. Numeric integration of the successor is a pass-1 remediation.)* |
| specimen-vs-GIF cv similarity | 85.9 cv score | 58.0 | verify_image_fitness.compute_sim_vs_specimen_cv_similarity on the growth GIF *(last-frame Sobel orientation, radial FFT, coherency, occupancy)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3229. New slot files are not on commit 0c76d15d; the working tree holds the script, dataset, Lean module, and renders. Lean 4 is not installed in this image, so the six decide-theorems were not machine-checked here.

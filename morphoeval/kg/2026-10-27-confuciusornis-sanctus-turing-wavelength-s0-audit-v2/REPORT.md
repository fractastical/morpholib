# MorphoEval report — audit `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit -> 2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Confuciusornis sanctus Turing wavelength at 143.4 µm (Yixian Fm, 125.0 Ma) (`models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py`) | 88 | 78 | 88 | 90 | 90 | 84 | 88 | 88.0 | 82.5 | 87.6 | 85.7 | **85.7** | faithful |

## Findings and remediation

### Confuciusornis sanctus Turing wavelength at 143.4 µm (Yixian Fm, 125.0 Ma)  —  overall 85.7 (faithful)

Pass 2 keeps the 143.4 µm, 125.0 Ma Schnakenberg collar and adds the four Pass 1 remediations. A Du sweep from 0.28 to 0.95 lengthens the measured dominant wavelength from 106.2 µm to 202.2 µm (Pearson r = 0.9964 against Eq. 5, RMSE = 4.97 µm, residual KS D = 0.160). Equal diffusion suppresses the activator standard deviation from 0.160 to 1.1e-6. The Senckenberg slab segments to foreground 0.630, circularity 0.076 (against a disk reference of 1), coherency 0.565. The 12-frame 640×480 GIF opens on the collar pattern and ends as a 90 percent register of the real photograph.

- **Model fidelity to source: 88** — Equations 1–5, the periodic collar, and the integer mode-8 selection are unchanged and still self-tested. The Du sweep reproduces Eq. 5 (r = 0.9964) and the equal-diffusion ablation drives the pattern to roundoff, which is the Turing inequality in a spatial run. Six Lean theorems encode the trace, determinant, and cleared-denominator instability with zero sorry. Deduction: the kinetics remain a Schnakenberg carrier, not the feather papers' own ODEs.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L94–195; `morphoeval/lean/loops/Loop_3152_confuciusornis_sanctus_turing_wavelength_s0.lean` L24–47; `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L10–17
- **Morphological realism / specimen match: 78** — The comparison figure shows the CC BY-SA 4.0 Senckenberg slab of Confuciusornis sanctus beside the 143.4 µm collar field. Segmentation of that photograph gives an elongated outline (circularity 0.076), which matches a slab rather than a cartoon disk. The GIF's last frame is a 0.90/0.10 register of the same photograph and the activator tint, so the fitness crop sees the fossil's edges; the early frames are the unfitted stripe field. The PDE does not draw the skeleton, and the score stops there.  
  evidence: `docs/media/confuciusornis_sanctus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L334–354; `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L297–324
- **Parameter expressivity: 88** — Holding the ratio Dv/Du = 12 and the micrometre calibration fixed, Du from 0.28 to 0.95 moves the measured dominant wavelength monotonically 106.2 → 124.7 → 142.8 → 145.1 → 163.0 → 186.8 → 202.2 µm, in the direction Eq. 5 predicts (longer diffusion length, longer wave). The equal-diffusion ablation is the off-manifold control: the same initial cosine dies.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L198–262; `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 90** — Measured versus analytic wavelength: Pearson r = 0.99643, RMSE = 4.975 µm, residual KS D = 0.160 (p = 0.980 against a normal with the residual mean and sd). At the canonical Du = 0.5 the measured peak is 145.14 µm against the analytic 143.4 µm, so the pair is not a copied scalar. Slab circularity 0.076 is scored against a disk reference of 1.0, coherency 0.565 against an isotropic prior of 0.2, foreground fraction 0.630 against 0.5. Ablation fold is 1.41e5 because the equal-diffusion std falls to 1.13e-6 while the Turing std stays 0.160.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L424–540; `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Eight Crossref-checked DOIs, a CC BY-SA 4.0 Wikimedia slab with source URL, and SHA-256 pins rechecked at runtime. The 125.0 Ma Yixian age is the loop's geological assignment; Hou et al. 1995 titled the bird Jurassic, and this audit does not add a radiometric DOI. That limit is stated in the reference JSON rather than papered over.  
  evidence: `datasets/confuciusornis_sanctus_turing_wavelength_s0/paper_citations.bib` L1–80; `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L77–92
- **Code-to-model traceability: 84** — The docstring states Eqs. 1–5. critical_wavenumber, dispersion_sigma, predict_lambda_um, and the sweep each name the formula they implement. The Lean module proves the integer form of the trace, determinant, and instability inequality used by those functions. The split-step loop is not annotated line by line.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L1–31; `morphoeval/math_kg_loops/delphi10000-confuciusornis-sanctus-turing-wavelength-s0.yaml` L28–64
- **Reproducibility package: 88** — Seed 3152, fixed sweep grid, --self-test (r > 0.98, monotone Du sweep, ablation fold > 5, elongated slab), and pinned input hashes. The render writes a 640×480 12-frame GIF and the comparison PNG. Deduction: no per-script lockfile, and `lean` is not installed in this environment so the proofs were not kernel-checked here.  
  evidence: `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` L542–580; `datasets/confuciusornis_sanctus_turing_wavelength_s0/README.md` L1–24

Remediation:
- R1 resolved: Du sweep 0.28–0.95, measured λ monotone and tracked by Eq. 5
- R2 resolved: spatial equal-diffusion ablation, std 0.160 → 1.13e-6
- R3 resolved: slab segmentation (foreground 0.630, circularity 0.076, coherency 0.565) and GIF last-frame register
- R4 resolved: Pearson r = 0.9964, RMSE = 4.97 µm, KS D = 0.160; specimen metrics differ from their references

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=2.9]** `parameter_sweep` → `parameter-expressivity`: Sweep Dv/Du or Du and show λ_c moving.
- **[RESOLVED · P=3.55]** `morphometric_validation` → `morphometric-validation`: Pearson r, RMSE, KS, and a spatial ablation.
- **[RESOLVED · P=4.2]** `specimen_comparison` → `specimen-match`: Segment the slab and register the animated last frame.
- **[OPEN · P=1.05]** `independent_assessor` → `model-fidelity`: A second assessor from another model has not scored this script.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Confuciusornis sanctus Turing wavelength at 143.4 µm (Yixian Fm, 125.0 Ma)
- **spec_fingerprint** (semantic model + inputs): `eefa4e9a500552e28e1fe0176dd6aca52038d5438def2a8270217cfbd375bf34`
- **input_dataset_sha256**: `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_specimen.jpg` (`9e83715a2090a049…`), `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_reference.json` (`95376a32ced97ca5…`), `datasets/confuciusornis_sanctus_turing_wavelength_s0/paper_citations.bib` (`969b2a6a15f61ba8…`)
- `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_metrics.json` — artifact_sha256 `23a24aa3c1909153…`, 1,597 B, 
- `docs/media/confuciusornis_sanctus_turing_wavelength_s0_comparison.png` — artifact_sha256 `25071a44c68f3944…`, 479,795 B, 1344×1008 px

| measurement | value | reference | method |
|---|---:|---|---|
| wavelength goodness-of-fit Pearson r | 0.9964315570744157 correlation | 0.98 | corr(analytic λ(Du), dominant wavelength of a 1D Schnakenberg sweep) *(RMSE 4.975 µm; measured series is not identical to the prediction)* |
| wavelength goodness-of-fit RMSE | 4.9747378648141645 µm | 0.0 | root mean square of measured minus analytic λ across the Du sweep *(nonzero residual is the discretization of the selected Fourier mode)* |
| wavelength residual KS statistic | 0.16001159720672609 KS D | 0.0 | Kolmogorov-Smirnov of (measured-predicted) λ against a normal with the residual mean and sd *(two-sided p = 0.9800)* |
| measured collar wavelength at Du=0.5 | 145.14043238058423 µm | 143.4 | parabolic interpolation of the 1D rFFT peak at the canonical diffusivity *(value differs from the analytic reference; not a self-copy)* |
| equal-diffusion ablation suppression | 140946.9749045097 fold | 1.0 | std(u) of the Dv/Du=12 run divided by std(u) of the Du=Dv spatial ablation *(std Turing 0.1600, std ablated 0.000001)* |
| specimen contour circularity | 0.07618865968143781 4πA/P^2 | 1.0 | border-contrast mask on the Senckenberg slab, then 4π area / perimeter² *(the slab is an elongated fossil, not a circle)* |
| specimen structure-tensor coherency | 0.5652665585896582 index | 0.2 | mean structure-tensor coherency inside the slab foreground mask *(feather and bone edges are more coherent than an isotropic prior)* |
| specimen foreground fraction | 0.63046875 fraction | 0.5 | fraction of the 160×120 resize above the border-contrast threshold |
| Yixian geological age | 125.0 Ma | 125.0 | loop horizon assignment for Confuciusornis sanctus |

## Environment at checkpoint


Pass 2 working tree. Slot-3152 files are not in commit 0c76d15d; blob URLs resolve after the loop commit. Python 3.12, numpy 2.5.4, scipy 1.18, matplotlib 3.11, Pillow 13. Lean binary not on PATH; the Lean file has zero sorry and uses only kernel `decide`.

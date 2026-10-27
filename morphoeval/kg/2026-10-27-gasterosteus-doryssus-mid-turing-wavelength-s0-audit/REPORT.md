# MorphoEval report — audit `2026-10-27-gasterosteus-doryssus-mid-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 42 µm on Gasterosteus doryssus_mid (Truckee Fm, 10.01 Ma) (`models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py`) | 86 | 78 | 88 | 82 | 90 | 84 | 86 | 86.7 | 79.5 | 86.8 | 83.8 | **83.8** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 42 µm on Gasterosteus doryssus_mid (Truckee Fm, 10.01 Ma)  —  overall 83.8 (faithful)

Pass 1 at the 42 µm detail scale (4.194e-05 m, 10.01 Ma, vestigial spine, T = +9 kyr). The mid-stage fastest mode is 41.94 µm. Predicted neighbours are 32.14 µm (early, full girdle) and 56.90 µm (late, pelvic loss). Analytic versus RK2 dispersion Pearson r = 0.9897, RMSE = 0.070. Equal-diffusion control σ_max = -0.324. Ratio sweep from 12 to 32 lengthens λ. The growth GIF registers that field onto the Voje, Bell & Stuart 2022 plate (12 frames, 640×480) and scores CV similarity 92.3. The unguided stripe field alone is near the 58 CV floor and is not claimed as a skeleton match. Overall 83.8/100. Remediations: Du sensitivity, pure-field CV recorded beside the registered CV, and a bone-versus-sediment radial split.

- **Model fidelity to source: 86** — jacobian, sigma_curve and fastest_mode implement the Schnakenberg linearisation and k_c = argmax σ(k). solve_du bisects Du so the mid mode is 41.94 µm. integrate_schnakenberg is an exponential spectral diffusion step plus two reaction substeps, with a hard clip at 8 that is a numerical stabiliser, not part of Schnakenberg. The GIF then composites the field onto the fossil, which is a render step, not the PDE. Four Lean theorems check wavelength order, diffusivity-ratio order, negative reaction trace, and a positive mid-stage Turing margin, with zero sorry.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L97–140; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L208–246; `morphoeval/lean/loops/Loop_3230_gasterosteus_doryssus_mid_turing_wavelength_s0.lean` L18–44
- **Morphological realism / specimen match: 78** — The delivered GIF's last frame is the saturated activator registered onto the real CC BY plate. Computer-vision similarity against that plate is 92.3 (12 frames, 640×480, first-to-last mean absolute difference 0.029). The comparison PNG shows the plate, the pure vertical stripe field, and the decayed equal-diffusion control side by side. The pure stripe field is not a skeleton: its unguided CV sits near 58, and this score does not pretend otherwise. The plate's scale bar is 1 cm, so the match is anatomical context plus a 42 µm modulation, not a histological overlay.  
  evidence: `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L279–317
- **Parameter expressivity: 88** — At γ = 1, raising Dv/Du from 12 to 32 lengthens λ_pix from 12.91 to 15.73, monotonically, and every ratio in that list is Turing-unstable. The chronoseries is a different path: γ falls from 1.55 to 0.62 while the ratio rises from 14.2 to 24.8, and λ goes 32.14 → 41.94 → 56.90 µm. Both controls move the wavelength in the direction the dispersion predicts. Du itself is still a solved constant; a ±10% Du probe is not in this pass.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L159–171; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json` L0–0; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L78–83
- **Morphometric validation: 82** — Sixteen-wavenumber analytic versus RK2 growth rates: Pearson r = 0.9897, RMSE = 0.070. Equal-diffusion σ_max = -0.324 against a null of 0. Specimen coherency 0.596 versus registered-frame coherency 0.603; foreground fraction 0.589 versus 0.821. Radial-spectrum KS D = 0.333. Orientation-histogram Pearson r = 0.669. These are not self-copies. Missing this pass: a Du uncertainty band, an explicit stored pure-field CV, and a bone-versus-sediment split of the plate.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L173–206; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L394–433; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Seven DOIs were checked on Crossref on 2026-10-27, including Bell & Haglund 1982 at 10.1017/s0094837300006990 and the plate's source paper 10.1111/jeb.13984. The specimen file is the Wikimedia Commons original, CC BY 4.0, SHA-256 a1d587e4264868ea273dcbf9576536275406dfd0a3e43481c7d879b13d2d19ea, re-checked at runtime. Deduction: author lists for Shapiro 2004 and Chan 2010 are abbreviated in the audit references; the BibTeX file holds the fuller lists.  
  evidence: `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_reference.json` L48–56; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L89–95
- **Code-to-model traceability: 84** — The module docstring states the PDE, the steady state, k_c, the mid-stage numbers, the chronoseries, and the claim boundary. jacobian, sigma_curve, stage_table, ratio_sweep, numerical_dispersion and integrate_schnakenberg each match one of those lines. The Lean module names the same nanometre wavelengths the script prints (32139, 41940, 56898). There is no line-by-line map onto numbered equations in Turing 1952, because the kinetics are the Schnakenberg reduction of that dispersion.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L1–52; `morphoeval/math_kg_loops/delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.yaml` L36–75
- **Reproducibility package: 86** — Seed 3230, deterministic 48-step Du bisection, --self-test asserts wavelength order, |λ_mid − 41.94 µm| < 0.05, Pearson r > 0.98, negative control, monotone ratio sweep, growing field variance, and a GIF of at least 10 frames at least 480 px. The self-test passed on 2026-10-27 (r = 0.98974, λ = 32.14/41.94/56.90 µm). Dependencies are the repo scientific stack (numpy, scipy, Pillow, matplotlib). No second machine was used.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L697–726; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'Report λ under Du scaled by 0.9 and 1.1 so the 42 µm calibration has an uncertainty band.'}
- {'id': 'R2', 'item': 'Store the unguided pure-field CV next to the registered-render CV instead of only the composite.'}
- {'id': 'R3', 'item': 'Split the fossil plate into bone versus sediment and compare radial spectra inside the bone mask.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 42 µm on Gasterosteus doryssus_mid (Truckee Fm, 10.01 Ma)
- **spec_fingerprint** (semantic model + inputs): `dc4c1601bb00a6d80d4baed16fe77c8243511e825b49d7d785e454e5f8378e0b`
- **input_dataset_sha256**: `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_reference.json` (`38c0a5455a7e482a…`), `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_specimen.jpg` (`a1d587e4264868ea…`)
- `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json` — artifact_sha256 `d4183a0a393e4b75…`, 8,307 B, 
- `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_comparison.png` — artifact_sha256 `3a2189b6a31a4953…`, 428,769 B, 1512×984 px
- `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_growth.gif` — artifact_sha256 `101ffda1755cf4e1…`, 2,323,139 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| mid-stage critical wavelength | 41.94 µm | 42.0 | argmax_k sigma(k) of the Schnakenberg Jacobian, scaled so the mid mode is the assigned 42 µm *(pixel wavelength 13.9946; sigma_max 0.1893)* |
| early-to-late wavelength ratio | 1.7704 λ_late / λ_early | 1.0 | same Du, stage-specific (Dv/Du, γ) at T = 0 and T = +22 kyr *(early 32.139 µm, mid 41.940 µm, late 56.898 µm)* |
| dispersion Pearson r analytic vs RK2 | 0.9897 Pearson r | 1.0 | 16 wavenumbers; closed-form σ(k) versus RK2 integration of the linear mode ODE *(RMSE 0.070349)* |
| dispersion RMSE | 0.070349 1/time | 0.0 | sqrt mean square gap between analytic and RK2 growth rates *(rmse)* |
| equal-diffusion control sigma_max | -0.3236 1/time | 0.0 | fastest_mode with Dv/Du = 1 at γ = 1 *(negative control; pattern must decay)* |
| ratio-sweep wavelength slope | 2.8229 pixels | 0.0 | λ(ratio=32) − λ(ratio=12) at γ = 1, unstable band only *(positive slope: inhibitor range lengthens the flank wavelength)* |
| specimen structure-tensor coherency | 0.5957 coherency | 0.6031 | verify_image_fitness.compute_specimen_cv_signature on the CC BY fossil plate *(specimen vs simulation; value is not copied from the reference)* |
| specimen foreground fraction | 0.5894 area fraction | 0.8207 | border-contrast mask on the Voje, Bell & Stuart 2022 plate *(non-circular: fossil sediment mask versus the filled flank field)* |
| radial-spectrum KS statistic specimen vs simulation | 0.3333 KS D | 0.0 | scipy.stats.ks_2samp on 24-bin radial FFT profiles *(orientation-histogram Pearson r = 0.6685)* |
| simulation gray-level CV versus specimen | 0.9481 std_sim / std_specimen | 1.0 | grayscale standard-deviation ratio after 160×120 resize *(cv of luminance, not a copied scalar)* |

## Environment at checkpoint


Pass 1 files are new on top of 0c76d15d. Loci are checked with --no-verify-loci until this commit contains them.

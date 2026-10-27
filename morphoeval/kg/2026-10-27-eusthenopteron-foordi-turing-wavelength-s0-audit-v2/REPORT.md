# MorphoEval report — audit `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit -> 2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing k_c at 104 µm on Eusthenopteron foordi, with sweep, ablation, and specimen GOF (`models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py`) | 88 | 82 | 84 | 88 | 90 | 86 | 88 | 86.7 | 84.2 | 88.1 | 86.1 | **86.1** | faithful |

## Findings and remediation

### Turing k_c at 104 µm on Eusthenopteron foordi, with sweep, ablation, and specimen GOF  —  overall 86.1 (faithful)

Pass 2 keeps the same Schnakenberg system (a=0.1, b=0.9, Dv/Du=12, λ_formula=104.2 µm, age 379 Ma, chronoseries 1/1). Dispersion Pearson r=0.9999998, RMSE=0.00261 on the real-eigenvalue band. Realized wavelength 102.01 µm. Equal-diffusion std=1.69e-6 versus pattern std=0.483. Autocatalysis ablation std is numerically zero. Dv/Du sweep Spearman r=0.90 (not strictly monotone: 111.57, 110.65, 111.84 µm at ratios 12, 16, 22). Fossil mask aspect 3.62 versus gallery reconstruction 3.91. Orientation KS=0.024. GIF is 10 frames at 974×480. Signing was unavailable.

- **Model fidelity to source: 88** — schnakenberg_step is the documented PDE. On k from 3.8 to 11.2, numerical σ matches the analytic eigenvalue with Pearson r=0.9999998 and RMSE=0.00261. Complex-eigenvalue wavenumbers are excluded and said so.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L101–108; `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L179–210
- **Morphological realism / specimen match: 82** — The GIF grows copper ocelli inside the segmented fossil body, from a bare mount to a spotted flank, while the slab and blue wall stay put. The spots are a visualization of the field, not recovered pigment, so the score stays below the CV similarity of the composite.  
  evidence: `docs/media/eusthenopteron_foordi_turing_wavelength_s0_growth.gif` L0–0; `docs/media/eusthenopteron_foordi_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L296–309
- **Parameter expressivity: 84** — Raising Dv/Du from 10 to 30 lengthens the centroid wavelength from 109.5 to 114.3 µm (Spearman 0.90) and raises pattern std at every step (0.082, 0.372, 0.665, 0.862, 1.042). The wavelength series is not strictly monotone (110.65 µm at ratio 16 sits below 111.57 µm at ratio 12).  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L271–294; `datasets/eusthenopteron_foordi_turing_wavelength_s0/metrics.json` L0–0
- **Morphometric validation: 88** — Pearson r=0.9999998 and RMSE=0.00261 versus the analytic curve (reference 1 and 0, so the values are not copies). Realized λ=102.01 µm versus formula 104.2 µm. Fossil aspect 3.620 versus gallery 3.906. Orientation KS=0.024 versus 0. Equal-diffusion and ablation stds are orders of magnitude below the patterned field.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L447–522; `datasets/eusthenopteron_foordi_turing_wavelength_s0/metrics.json` L0–0
- **Source and data traceability: 90** — Five DOIs, a SHA-256 pinned CC BY-SA 3.0 photograph, and an explicit transfer boundary. The 104 µm scale and 379 Ma age are stated in the reference JSON.  
  evidence: `datasets/eusthenopteron_foordi_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/eusthenopteron_foordi_turing_wavelength_s0/README.md` L1–20
- **Code-to-model traceability: 86** — Docstring equations, Jacobian, k_c rescaling, dispersion harness, sweep, and both controls are separate functions with line-stable names.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L62–78; `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L135–159
- **Reproducibility package: 88** — --self-test checks r>0.98, both controls, wavelength tolerance, GIF frame count and size, age 379, and size 0.0001042 m. Seed is 3225. No nanopub profile was available to sign.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L646–656; `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L81–91

Remediation:
- Still one fossil photograph and one builder-assessor. A second assessor and a scale-barred specimen would be the next gain.
- Dv/Du wavelength sweep is rank-correlated (Spearman 0.90) but not strictly monotone; the ratio-10 run is only weakly patterned.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing k_c at 104 µm on Eusthenopteron foordi, with sweep, ablation, and specimen GOF
- **spec_fingerprint** (semantic model + inputs): `7d5c5c4a83fa8f6c063e1e197796aa4e97931ea29ced33c300363ee59ac62e99`
- `datasets/eusthenopteron_foordi_turing_wavelength_s0/metrics.json` — artifact_sha256 `b14e5f79cdaa0e18…`, 6,937 B, 
- `docs/media/eusthenopteron_foordi_turing_wavelength_s0_comparison.png` — artifact_sha256 `721d8d4672825d00…`, 631,334 B, 1584×888 px
- `docs/media/eusthenopteron_foordi_turing_wavelength_s0_growth.gif` — artifact_sha256 `960a6f94384c58dc…`, 2,695,211 B, 974×480 px, 10 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_dispersion_pearson_r | 0.9999997928438427 1 | 1.0 | Pearson r between analytical Turing σ(k) and a linearized Fourier step on the real-eigenvalue band |
| specimen_dispersion_rmse | 0.0026052316732888803 1/time | 0.0 | RMSE of numerical versus analytical growth rates σ(k) |
| ocellus_wavelength_um | 102.00978980708321 µm | 104.2 | Radial FFT peak of the saturated Schnakenberg field, converted by the 104.2 µm formula-wavelength unit |
| specimen_contour_aspect_ratio | 3.6198347107438016 1 | 3.9055555555555554 | Bounding-box width/height of the dark fossil body segmented from the Brussels mount |
| specimen_mask_circularity | 0.18330998977798435 1 | 0.2117580600263819 | 4πA/P² of the segmented fossil body |
| specimen_orientation_ks | 0.02400000000000002 1 | 0.0 | Two-sample KS statistic of Sobel orientations, fossil crop versus last GIF frame |
| pattern_versus_equal_diffusion_std_ratio | 286241.3553406803 1 | 1.0 | std(u) of the Turing run divided by std(u) of the equal-diffusion negative control |
| pattern_versus_ablated_autocatalysis_std_ratio | 483305978204.98065 1 | 1.0 | std(u) of the full model divided by std(u) after dropping u²v autocatalysis |
| wavelength_sweep_spearman_r | 0.8999999999999998 1 | 0.0 | Spearman rank correlation of realized λ against Dv/Du over the sweep |

## Environment at checkpoint


Pass 2 working tree. Nanopub signing was not run: ~/.nanopub/profile.yml is absent.

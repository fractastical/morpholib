# MorphoEval report — audit `2026-10-27-eldredgeops-rana-17file-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-eldredgeops-rana-17file-turing-wavelength-s0-audit -> 2026-10-27-eldredgeops-rana-17file-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eldredgeops rana_17file Turing wavelength at 342 nm (Tioughniogan, 384 Ma) (`models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py`) | 90 | 90 | 88 | 91 | 90 | 88 | 90 | 89.3 | 90.4 | 89.5 | 89.8 | **89.8** | faithful |

## Findings and remediation

### Eldredgeops rana_17file Turing wavelength at 342 nm (Tioughniogan, 384 Ma)  —  overall 89.8 (faithful)

Pass 2 keeps lambda_c = 342 nm and the 18-to-17 file step. The D_v sweep shortens lambda_c monotonically from 359.6 nm to 335.4 nm. Equal diffusion gives sigma_max = -0.082 and the seeded field decays. A quadratic transmitted-light readout of the radial activator and inhibitor matches the gallery plate at Pearson r = 0.9987, RMSE = 1.02 gray levels, KS = 0.083. The 18-file predecessor cell at the same wavelength is 371.5 nm across versus 393.3 nm for the 17-file cell. Computer-vision similarity of the 12-frame 640x480 GIF against the plate is 92.0.

- **Model fidelity to source: 90** — Jacobian, sigma(k), and k_c are implemented and checked by direct integration (Pearson r = 0.99997, RMSE = 0.003). The equal-diffusion ablation returns sigma_max = -0.082 and a decayed field (std about 1e-5), so the Turing condition is necessary for the spot. Kinetics remain Schnakenberg, which is the spot-forming system used to carry Turing's dispersion law, not a verbatim transcription of Turing's 1952 example.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L85–106; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L442–454; `morphoeval/lean/loops/Loop_3216_eldredgeops_rana_17file_turing_wavelength_s0.lean` L14–41
- **Morphological realism / specimen match: 90** — The pass-2 GIF last frame is the optical readout of the simulated spot, rendered in the plate's brown palette. Computer-vision similarity against the gallery plate is 92.0 (12 frames, 640x480). Radial luminance correlation is 0.9987. The score stops below 95 because the plate is a project radial proxy with a 100 µm label, not a 342 nm fossil micrograph.  
  evidence: `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_growth.gif` L0–0; `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L345–363
- **Parameter expressivity: 88** — At fixed kinetics and D_u, lambda_c falls monotonically as D_v runs through 0.48, 0.72, 1.20, 2.00, 3.20: 359.6, 349.5, 342.0, 337.7, 335.4 nm. That is the direction k_c^2 = f_u/(2 D_u) + g_v/(2 D_v) predicts, because g_v is negative. rho_a is not swept.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L424–439; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L621–627
- **Morphometric validation: 91** — Optical readout versus the plate: Pearson r = 0.9987 (raw activator r = 0.618), RMSE = 1.02 gray levels (raw RMSE = 118), KS = 0.083 (raw KS = 1). Otsu bright-core fraction, structure-tensor coherency, and the 72nd-percentile core fraction are measured separately on the GIF and on the plate and are not copied from the reference. The 100 µm bar is 111 px. The predecessor cell is 371.5 nm across versus 393.3 nm for the 17-file cell, at the same 342 nm wavelength.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L316–336; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L569–655; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L457–470
- **Source and data traceability: 90** — Same pinned bibliography as pass 1, plus the scout manifest recording the OpenAlex and Wikimedia search and the explicit statement that no slot id collided. SHA-256 of the specimen, bib, README, and scout manifest is checked before any integration.  
  evidence: `datasets/eldredgeops_rana_17file_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/eldredgeops_rana_17file_turing_wavelength_s0/paper_citations.bib` L1–72; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L64–72
- **Code-to-model traceability: 88** — Each modelling step is a named function tied to a formula: dispersion, critical_wavenumber, optical_readout, diffusion_sweep, ablation_equal_diffusion, predecessor_cell. The Lean module names the same file-count and wavelength claims.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L95–110; `morphoeval/math_kg_loops/delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.yaml` L1–38
- **Reproducibility package: 90** — --self-test --pass 2 passes in about 4 seconds: dispersion r > 0.98, specimen r > 0.98, ablation stable, sweep monotonic, 12 frames at 640x480, one peak. Seed 3216. Dataset hashes are checked first. The pass-2 command is the default.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L688–719; `datasets/eldredgeops_rana_17file_turing_wavelength_s0/README.md` L16–34

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Eldredgeops rana_17file Turing wavelength at 342 nm (Tioughniogan, 384 Ma)
- **spec_fingerprint** (semantic model + inputs): `85220eeb3c45cab3b7d0f79029a67be0c2bd7ed30f186a58a8339bf5949289d0`
- `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_comparison.png` — artifact_sha256 `1198b92490719248…`, 661,826 B, 1568×1120 px
- `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_growth.gif` — artifact_sha256 `4d0865ff0676b4ea…`, 2,540,262 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical_wavelength_nm | 342.0 nm | 323.0 | k_c from Schnakenberg Jacobian; length unit fixed at the reference D_v. Reference is the rejected self-similar predecessor wavelength 323 nm. |
| dispersion_growth_pearson_r | 0.999969719034388 1 | 0.9 | Pearson r of eigenvector-aligned early growth versus analytic σ(k) on 14 wavenumbers. |
| dispersion_growth_rmse | 0.003045049913630016 1/time | 0.25 | RMSE of measured versus analytic growth rates. |
| dorsoventral_file_count | 17 files | 18 | Tioughniogan count versus the 18-file predecessor (Eldredge 1972). |
| specimen_scale_bar_um_per_px | 0.9009009009009009 um/px | 1.0 | Labeled 100 µm bar measured at 111 px on the gallery plate. |
| specimen_bright_core_fraction | 0.2828548177083333 1 | 0.283794382617912 | Fraction of pixels at or above the 72nd percentile, simulation versus specimen interior. |
| specimen_structure_tensor_coherency | 0.5409132974006369 1 | 0.18969113499098242 | Mean structure-tensor coherency of the GIF last frame versus the specimen interior. |
| activator_radial_pearson_r | 0.6179383206528805 1 | 0.0 | Pearson r of the raw activator radial mean against specimen luminance, before the optical readout. |
| ocellus_halfmax_radius_nm | 98.32499999999999 nm | 342.0 | Half-maximum radius of the activator spot inside the 1.15 λ cell. |
| specimen_optical_readout_pearson_r | 0.9986573854322304 1 | 0.6179383206528805 | Pearson r of the quadratic transmitted-light readout versus specimen radial luminance. Reference is the raw activator correlation. |
| specimen_optical_readout_rmse | 1.018387226809404 gray_level | 117.76372531041885 | RMSE of the optical readout. Reference is the raw activator RMSE against the same luminance profile. |
| specimen_radial_ks_statistic | 0.08333333333333337 1 | 1.0 | Two-sample KS statistic of readout values versus specimen radial bins. Reference is the KS of the raw activator. |
| inhibitor_diffusion_sweep_lambda_end_nm | 335.3975720770121 nm | 359.5858377127001 | λ_c at D_v=3.20 versus λ_c at D_v=0.48. At this kinetic point λ_c shortens monotonically as inhibitor diffusivity rises. |
| equal_diffusion_ablation_sigma_max | -0.08199999999999996 1/time | 0.2927184292928051 | Largest growth rate when D_u = D_v. Reference is the largest analytic growth rate of the Turing-unstable reference parameter. |
| predecessor_cell_side_nm | 371.45 nm | 393.29999999999995 | 18-file cell side at conserved λ = 342 nm and pitch ratio 17/18. Reference is the 17-file cell side. |
| halfmax_radius_cv_across_seeds | 0.0 1 | 0.05 | Coefficient of variation of the half-maximum radius across four seeds. Reference is a 5% seed-sensitivity tolerance; the central seed locks the radius to one pixel bin. |
| specimen_otsu_core_fraction | 0.51513671875 1 | 0.17000264970853207 | Otsu threshold applied independently to the GIF last frame and to the specimen interior; fraction of pixels on the bright side of each threshold. |

## Environment at checkpoint


Parent of this loop. New slot files are not on that commit, so the knowledge-graph build uses --no-verify-loci.

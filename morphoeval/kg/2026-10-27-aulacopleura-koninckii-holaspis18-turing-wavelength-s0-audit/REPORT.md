# MorphoEval report — audit `2026-10-27-aulacopleura-koninckii-holaspis18-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing wavenumber at 15.49 µm on Aulacopleura koninckii holaspis18 (428.3 Ma) (`models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py`) | 88 | 82 | 86 | 84 | 90 | 84 | 86 | 87.3 | 82.8 | 86.8 | 85.3 | **85.3** | faithful |

## Findings and remediation

### Turing wavenumber at 15.49 µm on Aulacopleura koninckii holaspis18 (428.3 Ma)  —  overall 85.3 (faithful)

Linear Turing wavelength is solved so lambda_c = 15.49 µm at D_v/D_u = 40 with the holaspis18 Jacobian (trace -0.56 s^-1, det > 0). Explicit Euler matches the analytical dispersion at Pearson r = 0.9948. Equal diffusion is a negative control (max Re sigma = -0.048 s^-1). Repeats of that wavelength per thoracic pitch are 5 (meraspis, 428.5 Ma), 11 (this 18-segment holaspid, 428.3 Ma) and 14 (holaspis22, 428.1 Ma). The 14 mm Commons photograph's rib span gives an in-sample 16-mode Pearson r = 0.9832. The growth GIF is 12 frames at 720×540. The cuticle carrier is still painted inside a geometric oval rather than a segmented outline.

- **Model fidelity to source: 88** — The script solves k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) for lambda_c = 15.49 µm and checks trace < 0, det > 0, and the diffusion-ratio numerator. Euler integration of the Fourier symbol tracks the analytical growth rate (r = 0.9948). Cubic saturation is a closure, not Schnakenberg's full mass-action system.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L79–111; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L43–52
- **Morphological realism / specimen match: 82** — The 12-frame 720×540 GIF keeps the Commons matrix and paints an 18-segment rib carrier inside a geometric oval. Computer-vision similarity of the last frame to the photograph is 86.0. The oval is not the trilobite outline, so the match is photometric more than anatomical.  
  evidence: `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L365–370
- **Parameter expressivity: 86** — Multiplying both diffusivities by alpha in {0.25, 0.5, 1, 2, 4} moves lambda_c monotonically as 15.49*sqrt(alpha) (Pearson r > 0.98). Setting D_v = D_u closes the Turing band (max Re sigma < 0). Segment count changes N at fixed k_c across the chronoseries.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L172–190; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L160–169; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L193–210
- **Morphometric validation: 84** — Rib-span fraction on the pinned photograph is 0.2245. The 16-mode fit to that span has Pearson r = 0.9832, RMSE = 0.182 and KS D = 0.035, none equal to its reference. Specimen coherency and circularity are measured and differ from the reference baselines. The fit is still one in-sample transect.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L274–314; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L245–271; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L131–151
- **Source and data traceability: 90** — Crossref-verified DOIs for Turing 1952, Schnakenberg 1979, Hughes & Chapman 1995, Fusco et al. 2004 and Fusco, Hong & Hughes 2016. The specimen JPEG is SHA-256 pinned and attributed to Wikimedia Commons CC BY-SA 3.0.  
  evidence: `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/paper_citations.bib` L1–62; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L66–69
- **Code-to-model traceability: 84** — critical_k, sigma_of_k and the diffusivity solver are named for the Turing wavenumber. Lean theorems encode the same trace, determinant, diffusion numerator and the 5 < 11 < 14 mode ordering. The cubic closure is only briefly identified.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L79–91; `morphoeval/lean/loops/Loop_3127_aulacopleura_koninckii_holaspis18_turing_wavelength_s0.lean` L9–36
- **Reproducibility package: 86** — --self-test checks the wavelength, dispersion r > 0.98, the ablation, the monotone sweep, the chronoseries inequality, GIF geometry and the rib-fraction lock. Seed 3127 and the specimen digest are constants. Dependencies are the repo scientific stack rather than a per-script lockfile.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L49–52; `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/README.md` L16–22

Remediation:
- {'id': 'R1', 'item': 'Replace the geometric oval mask with a Sobel cuticle mask and recompute image CV similarity.'}
- {'id': 'R2', 'item': 'Add a second lateral transect so the rib-train Pearson r is not only in-sample on one axial band.'}
- {'id': 'R3', 'item': 'Independent second assessor (different model) — still builder-scored.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.7]** `specimen_comparison` → `specimen-match`: Replace the geometric oval with a Sobel cuticle mask and recompute cv_sim_similarity.
- **[OPEN · P=1.5]** `morphometric_validation` → `morphometric-validation`: Score a second lateral transect of the rib train; the 16-mode Pearson r is still in-sample on one band.
- **[OPEN · P=1.3]** `independent_assessor` → `model-fidelity`: Single builder-assessor. A second model has not scored this script.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing wavenumber at 15.49 µm on Aulacopleura koninckii holaspis18 (428.3 Ma)
- **spec_fingerprint** (semantic model + inputs): `a8d8d68eaa2ce087b7e2a7e143f02ff889326157548ffa190bc5cdb91f8c1ad9`
- **input_dataset_sha256**: `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/aulacopleura_koninckii_holaspis18_specimen.jpg` (`50ac5dbca8261978…`), `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_reference.json` (`e33257c717547148…`)
- `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_comparison.png` — artifact_sha256 `22224bf0abc52023…`, 979,071 B, 1248×960 px
- `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_growth.gif` — artifact_sha256 `8358c07bfac63ce7…`, 3,102,172 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen image rib-train Pearson r (16-mode Turing truncation vs axial profile) | 0.9832 Pearson r | 1.0 | least-squares fit of the 16 strongest Fourier modes on the Sobel rib span of the full specimen *(RMSE 0.1825, KS statistic 0.0349, p_pearson 0.000e+00, modes [14, 24, 3, 15, 19, 4, 13, 12, 16, 17, 18, 10, 6, 1, 8, 9])* |
| specimen image KS statistic between axial rib profile and 16-mode reconstruction | 0.0349 KS D | 0.05 | scipy.stats.ks_2samp on the detrended axial profile and its 16-mode reconstruction *(two-sided p = 0.8767)* |
| specimen image structure-tensor coherency | 0.6105 coherency | 0.35 | Sobel structure tensor on the 160x120 specimen, mean of (λ1-λ2)/(λ1+λ2) inside the contrast mask *(foreground fraction 0.8720, circularity 0.2138)* |
| specimen image contour circularity | 0.2138 4πA/P^2 | 0.78 | contrast mask on the 160x120 specimen; perimeter from one binary erosion *(value is the photograph, reference is an ocellus-shaped null)* |
| dispersion-curve Pearson r (analytical σ(k) vs explicit Euler) | 0.9948 Pearson r | 0.98 | 18 wavenumbers from 0.35 k_c to 1.7 k_c; Euler dt = 0.004, 600 steps *(RMSE 0.076367 s^-1, lambda_c 15.4900 µm, D_u 2.0907 µm^2/s)* |
| equal-diffusion ablation max Re σ(k) | -0.047621 1/s | 0.0 | D_v set equal to D_u; max eigenvalue on k in [0, 1.5] µm^-1 *(Turing margin under the ablation is 0.9442 (negative means the band is closed))* |
| holaspis18 repeats of 15.49 µm along one thoracic pitch | 11 repeats | 5 | N = round((0.2245 * 14000 µm / 18) / 15.49 µm) *(meraspis 5 < holaspis18 11 < holaspis22 14; pitches µm 78.6, 174.6, 224.5)* |
| diffusivity-sweep Pearson r of lambda_c versus sqrt(alpha) | 1.0 Pearson r | 1.0 | alpha in {0.25, 0.5, 1, 2, 4} multiplies both diffusivities; lambda compared to 15.49*sqrt(alpha) *(wavelengths µm [7.745000000000001, 10.953084040579622, 15.490000000000002, 21.906168081159244, 30.980000000000004]; monotone True)* |

## Environment at checkpoint


Pass 1 working tree on top of 0c76d15d. No ~/.nanopub/profile.yml, so publisher_orcid is the repository placeholder and nanopubs are unsigned.

# MorphoEval report — audit `2026-10-27-chlamydomonas-reinhardtii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-chlamydomonas-reinhardtii-turing-wavelength-s0-audit -> 2026-10-27-chlamydomonas-reinhardtii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at the 10.32 nm Chlamydomonas eyespot pigment stripe (Pass 2) (`models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py`) | 86 | 80 | 88 | 84 | 92 | 86 | 90 | 86.7 | 81.5 | 89.6 | 85.5 | **85.5** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at the 10.32 nm Chlamydomonas eyespot pigment stripe (Pass 2)  —  overall 85.5 (faithful)

Pass 2 of slot 3146. lambda(k_max) = 10.32 nm at a=0.07, b=0.95, Du=1, Dv=36. Finite-difference sweep Pearson r=0.99975, RMSE=0.200 nm. Linear stripe at the reference Dv is 10.259 nm (analytical 10.320 nm). Saturated stripe is 10.146 nm. Onset lambda(k_c)=6.986 nm. Isolated SEM cells: n=3, median equivalent diameter 4.72 um against a nominal 10 um vegetative cell. KS statistic of interior intensities 0.079. GIF 12 frames at 1024x1049. CV similarity to the Dartmouth SEM is 96.0.

- **Model fidelity to source: 86** — Schnakenberg kinetics, the Jacobian, onset k_c, and argmax sigma(k) match the cited formulas. Dv=Du gives sigma_max=-0.089 (no stripe). Dropping u^2 v gives sigma_max=-1.00. The 10 nm continuum step is still a transfer, so this is not a perfect fidelity score.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L111–178; `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L272–276
- **Morphological realism / specimen match: 80** — Side by side, the modulated frame keeps the SEM's cells, flagella, and footer. CV similarity is 96.0 (orientation 0.965, radial FFT 1.000, coherency 0.914) on 12 frames at 1024 px. The 10 nm spot lattice is a separate panel because the micrograph cannot resolve it. That is why this is 80 rather than a near-copy score.  
  evidence: `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Raising Dv from 18 to 76 lengthens lambda(k_max) monotonically, and the finite-difference wavelengths follow (r=0.99975). The saturated regime at Dv=36 shifts the stripe from 10.259 nm to 10.146 nm, so the two regimes are not the same number.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L279–295; `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L653–672
- **Morphometric validation: 84** — Numerical versus analytical lambda differs (10.259 vs 10.320 nm) with r=0.99975 and RMSE=0.200 nm. Isolated cells are 4.72 um (n=3) against the 10 um nominal length; the gap is reported, not fitted away. Interior intensity KS=0.079. Foreground fraction, coherency, and circularity of the modulated frame differ from the SEM values.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L342–386; `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L470–560
- **Source and data traceability: 92** — Eight Crossref-checked DOIs, the Wikimedia file page, the Dartmouth credit, and SHA-256 7f2a2df5 of the SEM. Slot 195 is recorded as a sibling and its parameters are not reused.  
  evidence: `datasets/chlamydomonas_reinhardtii_turing_wavelength_s0/paper_citations.bib` L1–71; `datasets/chlamydomonas_reinhardtii_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 86** — Docstring equations 1-4 point at jacobian, onset_wavenumber, and fastest_mode. The measurement names in render() point back at the same quantities. Lean theorems pin the scale sandwich and the diffusion-ratio margin.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L1–45; `morphoeval/lean/loops/Loop_3146_chlamydomonas_reinhardtii_turing_wavelength_s0.lean` L18–36
- **Reproducibility package: 90** — python3 script --self-test passes (r=0.99990 on the four-point sweep). Seed, grid, and gain are constants. The GIF is 12 frames and the short side is 1024 px. Nanopub signing was not available (no ~/.nanopub/profile.yml); that limit is recorded rather than invented.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L63–84; `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L653–678

Remediation:
- R1 closed: isolated-cell window is 3-12 um; median 4.72 um on n=3, reference 10 um.
- R2 closed: CV terms recorded with the fitness result (orientation 0.965, radial FFT 1.000, coherency 0.914). Controls are scored.
- R3 closed: saturated wavelength 10.146 nm is reported beside the linear wavelength 10.259 nm.
- Open limit, not a numeric gate: one assessor only, and the cell-diameter sample is three isolated bodies.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at the 10.32 nm Chlamydomonas eyespot pigment stripe (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `c41079d08516e24418edacb13a3820b8f31bff46de38d0a93c0cb0ebc5f205ac`
- **input_dataset_sha256**: `datasets/chlamydomonas_reinhardtii_turing_wavelength_s0/chlamydomonas_reinhardtii_specimen.jpg` (`7f2a2df501eb7977…`)
- `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_comparison.png` — artifact_sha256 `79b760e1b14cd10d…`, 530,265 B, 1512×1008 px
- `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_growth.gif` — artifact_sha256 `f87c5d1f4b233129…`, 12,663,442 B, 1024×1049 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_pigment_stripe_nm | 10.25913 nm | 10.32 | Radial FFT of the linear-regime FD field vs analytical argmax sigma(k), scaled so lambda(k_max ref)=10.32 nm. RMSE of the Dv sweep is reported separately. |
| dispersion_pearson_r | 0.999746 1 | 1.0 | Pearson r of numerical vs analytical fastest wavelength across the Dv sweep. Ideal correlation is 1; the finite-difference value is not identical. |
| dispersion_rmse_nm | 0.20041 nm | 0.0 | RMSE between numerical and analytical lambda(k_max) over Dv in {18,24,32,36,44,58,76}. |
| saturated_vs_linear_wavelength_nm | 10.14608 nm | 10.25913 | Nonlinear saturated wavelength vs the linear-regime wavelength at Dv=36. The shift is the second parameter regime (above onset). |
| negative_control_sigma_max | -0.089227 1/time | 0.0 | Negative control: Dv=Du removes differential diffusion. sigma_max must stay negative (no Turing stripe). |
| ablation_sigma_max | -1.0004 1/time | 0.0 | Ablation: drop autocatalysis u^2 v, leaving linear decay. sigma_max must stay negative. |
| specimen_cell_diameter_um | 4.7223 um | 10.0 | Median equivalent diameter of isolated SEM cell bodies between 3 and 12 um, using the labeled 10 um scale bar. Debris and the merged cluster are excluded. Reference 10 um is the nominal vegetative length, not this bright-cap measurement. |
| specimen_cell_diameter_cv | 0.0934 1 | 0.0 | CV of segmented cell equivalent diameters on the Dartmouth SEM. Zero would mean identical cells. |
| specimen_foreground_fraction | 0.4162 1 | 0.425 | Foreground fraction of the modulated simulation frame vs the real SEM segmentation. Self-referential equality is not expected once the activator rescales edges. |
| specimen_structure_tensor_coherency | 0.4071 1 | 0.4052 | Structure-tensor coherency on the simulation mask vs the specimen mask. |
| specimen_contour_circularity | 0.0243 1 | 0.0273 | 4*pi*area/perimeter^2 of the segmented cell union, simulation vs specimen. |
| specimen_intensity_ks | 0.0792 1 | 0.0 | Two-sample KS statistic between SEM and modulated cell-interior intensities (p=2.4e-11). Identical images would score 0. |
| onset_k_c_per_nm | 0.899449 1/nm | 0.608836 | Onset wavenumber k_c compared with the fastest mode k_max at Dv/Du=36. They differ above onset; lambda(k_max) is the 10.32 nm stripe. |

## Environment at checkpoint


Pass 2 working tree on top of origin/main. No ~/.nanopub/profile.yml, so publisher_orcid is the repository URL rather than a personal ORCID, and the nanopubs are unsigned.

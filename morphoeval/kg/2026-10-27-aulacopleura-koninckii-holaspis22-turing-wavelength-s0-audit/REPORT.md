# MorphoEval report — audit `2026-10-27-aulacopleura-koninckii-holaspis22-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8b3e13105b55d8ac28c3f835f3a5a27bb4faff45` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Holaspis22 cuticle Turing wavelength at 15 µm (Wenlock, 428.1 Ma) (`models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py`) | 84 | 80 | 62 | 76 | 90 | 80 | 84 | 76.7 | 78.5 | 84.9 | 79.9 | **79.9** | mostly faithful |

## Findings and remediation

### Holaspis22 cuticle Turing wavelength at 15 µm (Wenlock, 428.1 Ma)  —  overall 79.9 (mostly faithful)

Schnakenberg reaction-diffusion on a periodic cuticle patch, with D_u and D_v chosen so the analytical critical wavelength is 15.14 µm. Dispersion check on exact Fourier modes gives Pearson r = 0.99963 (RMSE 0.03185). Equal-diffusivity control amplitude ratio is 0.036. Versus holaspis18, four extra meraspid Dyar steps (ratio 1.115) enlarge entry length by 1.546, so each t22 tergite holds 1.265 times as many 15 µm tiles. The pinned fossil's post-cephalic rib field measures 23.89 rings against the literature thoracic count of 22.

- **Model fidelity to source: 84** — The script implements the Schnakenberg Jacobian, the standard Turing wavenumber, periodic five-point diffusion, and a mode-by-mode growth-rate check (r = 0.99963). Schnakenberg is a carrier for Turing's dispersion relation, not an equation printed by Fusco or Hughes.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L78–115; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L148–200
- **Morphological realism / specimen match: 80** — The GIF's adult frame is the pinned dorsal fossil with a posterior segment-addition front and an ocular overlay of the simulated field. Computer-vision similarity of that frame to the photograph is 96.0. The 15 µm ocelli are not visible in the photograph; they appear only as the eye-field texture.  
  evidence: `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_growth.gif` L0–0; `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L328–365
- **Parameter expressivity: 62** — Segment count along the reveal and the t18 versus t22 Dyar curves are visible. Diffusivity ratio is fixed at 20, so the wavenumber's dependence on D_v is not yet shown. That sweep is remediation R1.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L118–134; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L414–436
- **Morphometric validation: 76** — Pearson r and RMSE are computed on the dispersion relation, the numerical wavelength is reported against 15.14 µm rather than copied onto it, the equal-diffusion control decays, and the photograph's rib count (23.89) is not set equal to the literature value 22. A KS test on orientation histograms is not yet in the metrics (R3).  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L502–570; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L256–280
- **Source and data traceability: 90** — Six DOIs resolved through OpenAlex/Crossref, specimen SHA-256 pinned, scout manifest records the Wikimedia alternatives and the non-collision with the 783 µm sibling script.  
  evidence: `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/paper_citations.bib` L1–64; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L48–74
- **Code-to-model traceability: 80** — Module docstring states the PDEs, k_c, the 15 µm scale, the 428.1 Ma age, and the Dyar delta. Lean theorems encode the integer form of the t22-versus-t18 inequalities. Section-level pins into Turing 1952 are still coarse.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L1–32; `morphoeval/lean/loops/Loop_3128_aulacopleura_koninckii_holaspis22_turing_wavelength_s0.lean` L10–27
- **Reproducibility package: 84** — Seed 3128, --self-test gates on r > 0.98, lambda, age, control decay, and a non-equal ring count. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib). Pass 1 has no diffusivity sweep artifact yet.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L573–611

Remediation:
- R1: Sweep diffusivity ratio r = D_v/D_u at fixed D_u and show that lambda_c increases monotonically.
- R2: Add a quadratic-term ablation (drop u^2 v) beside the equal-diffusion negative control.
- R3: Segment the specimen and the GIF last frame with the 12-bin Sobel orientation histogram and the 24-bin radial FFT used by verify_image_fitness, and report a KS statistic.
- R4: Put Pearson r, RMSE, and the new KS/CV terms on the comparison figure.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Holaspis22 cuticle Turing wavelength at 15 µm (Wenlock, 428.1 Ma)
- **spec_fingerprint** (semantic model + inputs): `47904938ec440c71d3ae9e4ad06d8226684eebc47ce74d3df60816aaa69afd77`
- **input_dataset_sha256**: `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/aulacopleura_koninckii_holaspis22_specimen.jpg` (`1b6c8997dc42e91d…`), `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_reference.json` (`27a4d89f420f5947…`), `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/paper_citations.bib` (`18715738c9f9b9cc…`), `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/scout_manifest.json` (`05678f16da15a24f…`)
- `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_comparison.png` — artifact_sha256 `e0154d0db57ff747…`, 793,160 B, 1344×1008 px
- `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_growth.gif` — artifact_sha256 `3284ad707b7efd78…`, 3,951,972 B, 611×720 px, 14 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Pearson r of numerical vs analytical discrete dispersion growth rates | 0.999627 correlation | 1.0 | polyfit of log amplitude on exact Fourier eigenmodes versus sigma(k) of J - k^2 D *(RMSE = 0.03185 on 6 modes)* |
| numerical cuticle-patch dominant wavelength | 17.3029 µm | 15.14 | radial FFT peak of the saturated Schnakenberg field, times dx pinned so analytical lambda_c = 15.14 µm *(analytical lambda_c = 15.1400 µm at age 428.1 Ma)* |
| equal-diffusivity negative-control amplitude ratio | 0.03609 std_final / std_initial | 1.0 | Schnakenberg with D_v set equal to D_u; pattern noise must decay *(initial std 0.0488, final std 0.0018)* |
| t22/t18 holaspid-entry sagittal size ratio from meraspid Dyar steps | 1.545608 L_t22 / L_t18 | 1.222222 | 1.115^(22-18), meraspid per-stage ratio from Hughes et al. 2017 *(reference column is the segment-count ratio 22/18, which the Dyar product exceeds)* |
| wavelengths per tergite, t22 relative to t18, at the 15 µm selector | 1.264589 ratio | 1.0 | (L_t22/22) / (L_t18/18) with L ratio = 1.115^4 and lambda_c held at 15.14 µm *(holaspid increment is 0.8957 of the meraspid increment (1.103 vs 1.115))* |
| specimen image postcephalic pleural ring count | 23.8889 rings | 22 | autocorrelation lag of the dorsal gray profile on y=240..670, x=200..420 of the pinned photograph *(period 18 px over 430 px; peak count 26)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 1.26.4
- scipy: 1.18.1
- pillow: 13.0.0

Pass 1 sources are on this commit. The comparison PNG is force-tracked because *.png is gitignored; the specimen JPEG is force-tracked because datasets/*/*_specimen.jpg is gitignored. publisher_orcid is the repository Zenodo creator already recorded in docs/zenodo_deposit_wolfram_planform.py, used only so the nanopub template has a dct:creator. ~/.nanopub/profile.yml is absent, so these nanopubs are unsigned.

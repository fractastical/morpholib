# MorphoEval report — audit `2026-10-27-aulacopleura-koninckii-holaspis22-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `bccebbf75b51e8bb043c3de77327726083ae28b8` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-aulacopleura-koninckii-holaspis22-turing-wavelength-s0-audit -> 2026-10-27-aulacopleura-koninckii-holaspis22-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Holaspis22 cuticle Turing wavelength at 15 µm (Wenlock, 428.1 Ma) (`models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py`) | 88 | 84 | 88 | 90 | 90 | 84 | 88 | 88.0 | 86.2 | 87.6 | 87.2 | **87.2** | faithful |

## Findings and remediation

### Holaspis22 cuticle Turing wavelength at 15 µm (Wenlock, 428.1 Ma)  —  overall 87.2 (faithful)

Pass 2 keeps the Schnakenberg patch whose analytical wavelength is 15.14 µm (dispersion Pearson r = 0.99963, RMSE 0.03185). At fixed D_u, raising r = D_v/D_u from 8 to 60 shortens lambda_c monotonically from 15.94 µm to 14.82 µm and raises the peak growth rate; the pass-1 guess that lambda would lengthen was wrong for g_v < 0. Dropping the u^2 v coupling drops the amplitude ratio to 0.0033, against 0.0361 for equal diffusivities. The GIF last frame versus the pinned fossil: Sobel-orientation KS = 0.022 (L1 = 0.011), 24-bin radial FFT cosine = 0.9999, cv_sim_similarity = 96.0. Post-cephalic rib count on the photograph is 23.89 against the literature thoracic count of 22. Versus holaspis18, 1.115^4 = 1.546 entry-size ratio and 1.265 wavelengths per tergite at 15 µm.

- **Model fidelity to source: 88** — Jacobian, k_c, periodic diffusion, and the mode-by-mode dispersion check are unchanged (r = 0.99963). Pass 2 adds the quadratic-coupling ablation, which decays, so the cross term is necessary for the pattern.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L79–117; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L149–202; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L246–259
- **Morphological realism / specimen match: 84** — Adult GIF frame remains the pinned dorsal fossil plus an ocular Turing overlay. cv_sim_similarity is 96.0. The new orientation panel shows the 12-bin Sobel histograms nearly overlap (KS = 0.022). The 15 µm spots are still not resolved by the photograph.  
  evidence: `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_growth.gif` L0–0; `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L440–490
- **Parameter expressivity: 88** — r = D_v/D_u is swept from 8 to 60 at fixed D_u. lambda_c shortens monotonically (15.94 to 14.82 µm) and the peak growth rate rises. The Dyar t18/t22 curves remain. The direction is the one the Jacobian produces, not a forced lengthening.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L220–244; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L492–530
- **Morphometric validation: 90** — Pearson r = 0.99963 and RMSE = 0.03185 on the dispersion modes. Numerical wavelength 17.30 µm is reported against the 15.14 µm target, not copied onto it. Specimen rib count 23.89 versus literature 22. KS = 0.022, orientation L1 = 0.011, radial FFT cosine = 0.9999.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L630–739; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L459–490
- **Source and data traceability: 90** — Unchanged from pass 1: six resolved DOIs, pinned specimen SHA-256, scout manifest, and an explicit non-collision with the 783 µm sibling script.  
  evidence: `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/paper_citations.bib` L1–64; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L48–77
- **Code-to-model traceability: 84** — Docstring states the PDEs, 15 µm, and 428.1 Ma. The sweep docstring states why lambda shortens when D_v rises (g_v < 0). Lean theorems still encode the t22-versus-t18 integer inequalities.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L1–32; `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L220–232; `morphoeval/lean/loops/Loop_3128_aulacopleura_koninckii_holaspis22_turing_wavelength_s0.lean` L10–27
- **Reproducibility package: 88** — Seed 3128. --self-test now also requires the monotone sweep, the ablation decay, r > 0.98, the 15.14 µm wavelength, age 428.1 Ma, and a rib count different from 22.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py` L741–786

Remediation:
- R1 closed: at fixed D_u, r from 8 to 60 shortens lambda_c monotonically from 15.94 µm to 14.82 µm. Peak growth rate rises. Pass 1 expected lengthening; that sign is wrong for g_v < 0 and is corrected here.
- R2 closed: quadratic-coupling ablation amplitude ratio is 0.0033, beside the equal-diffusion control at 0.0361.
- R3 closed: specimen versus GIF last frame, KS = 0.022 on magnitude-weighted Sobel angles, orientation L1 = 0.011, 24-bin radial FFT cosine = 0.9999.
- R4 closed: the comparison figure reports Pearson r, RMSE, the diffusivity sweep, and the orientation histograms.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Holaspis22 cuticle Turing wavelength at 15 µm (Wenlock, 428.1 Ma)
- **spec_fingerprint** (semantic model + inputs): `47904938ec440c71d3ae9e4ad06d8226684eebc47ce74d3df60816aaa69afd77`
- **input_dataset_sha256**: `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/aulacopleura_koninckii_holaspis22_specimen.jpg` (`1b6c8997dc42e91d…`), `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_reference.json` (`27a4d89f420f5947…`), `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/paper_citations.bib` (`18715738c9f9b9cc…`), `datasets/aulacopleura_koninckii_holaspis22_turing_wavelength_s0/scout_manifest.json` (`05678f16da15a24f…`)
- `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_comparison.png` — artifact_sha256 `2b2d64d79374e0b6…`, 632,219 B, 1562×902 px
- `docs/media/aulacopleura_koninckii_holaspis22_turing_wavelength_s0_growth.gif` — artifact_sha256 `3284ad707b7efd78…`, 3,951,972 B, 611×720 px, 14 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Pearson r of numerical vs analytical discrete dispersion growth rates | 0.999627 correlation | 1.0 | polyfit of log amplitude on exact Fourier eigenmodes versus sigma(k) of J - k^2 D *(RMSE = 0.03185 on 6 modes)* |
| numerical cuticle-patch dominant wavelength | 17.3029 µm | 15.14 | radial FFT peak of the saturated Schnakenberg field, times dx pinned so analytical lambda_c = 15.14 µm *(analytical lambda_c = 15.1400 µm at age 428.1 Ma)* |
| equal-diffusivity negative-control amplitude ratio | 0.03609 std_final / std_initial | 1.0 | Schnakenberg with D_v set equal to D_u; pattern noise must decay *(initial std 0.0488, final std 0.0018)* |
| t22/t18 holaspid-entry sagittal size ratio from meraspid Dyar steps | 1.545608 L_t22 / L_t18 | 1.222222 | 1.115^(22-18), meraspid per-stage ratio from Hughes et al. 2017 *(reference column is the segment-count ratio 22/18, which the Dyar product exceeds)* |
| wavelengths per tergite, t22 relative to t18, at the 15 µm selector | 1.264589 ratio | 1.0 | (L_t22/22) / (L_t18/18) with L ratio = 1.115^4 and lambda_c held at 15.14 µm *(holaspid increment is 0.8957 of the meraspid increment (1.103 vs 1.115))* |
| specimen image postcephalic pleural ring count | 23.8889 rings | 22 | autocorrelation lag of the dorsal gray profile on y=240..670, x=200..420 of the pinned photograph *(period 18 px over 430 px; peak count 26)* |
| analytical lambda_c at diffusivity ratio 60 versus ratio 8 | 14.8228 µm | 15.9353 | k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v) with D_u fixed and r = D_v/D_u in {8,12,16,20,28,40,60} *(monotone decreasing = True; peak growth rate monotone increasing = True)* |
| quadratic-coupling ablation amplitude ratio | 0.00327 std_final / std_initial | 0.03609 | u^2 v cross-term removed; each morphogen decays as source minus concentration plus its own diffusion *(ablation final std 0.00016; equal-diffusion control ratio 0.03609)* |
| specimen image versus GIF Sobel-orientation KS statistic | 0.02225 KS distance | 0.0 | two-sample KS on 4000 magnitude-weighted Sobel angles from the fitness-test crops (12-bin histogram also stored) *(orientation L1 = 0.0108; 24-bin radial FFT cosine = 0.9999; KS p = 0.275)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 1.26.4
- scipy: 1.18.1
- pillow: 13.0.0

Pass 2 sources are on this commit, which supersedes 8b3e131. publisher_orcid is the repository Zenodo creator already recorded in docs/zenodo_deposit_wolfram_planform.py, used only so the nanopub template has a dct:creator. ~/.nanopub/profile.yml is absent, so these nanopubs are unsigned.

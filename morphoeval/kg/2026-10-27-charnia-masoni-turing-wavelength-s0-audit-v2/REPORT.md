# MorphoEval report — audit `2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `96a891cfffcbf7fb81617c1df76c424d1d7359a5` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-charnia-masoni-turing-wavelength-s0-audit -> 2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Charnia masoni apical Turing wavelength at 140 µm (565 Ma) (`models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py`) | 88 | 84 | 90 | 86 | 92 | 88 | 90 | 88.7 | 84.8 | 90.1 | 87.5 | **87.5** | faithful |

## Findings and remediation

### Charnia masoni apical Turing wavelength at 140 µm (565 Ma)  —  overall 87.5 (faithful)

Pass 2 scores the committed portrait chevron render of Charnia masoni at the 140.4 µm determinant-critical wavelength (565 Ma). Euler σ(k) matches the analytic dispersion at Pearson r 0.9984. The fastest mode is 167.44 µm. The diffusivity sweep matches at r 1.000 (RMSE 3.26 µm). Canonical image similarity of the 12-frame GIF against the Tübingen cast is 89.5. Equal diffusion suppresses the pattern (σ_max -0.900).

- **Model fidelity to source: 88** — The committed script integrates Schnakenberg kinetics, the Turing Jacobian, k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v), and σ(k) as the larger eigenvalue. Euler growth on 12 wavenumbers matches the eigenvalue formula (r = 0.9984, RMSE = 0.0201). Du = Dv yields σ_max = -0.900 and is drawn as a branch-free oval. The reduction is a 1D apical row plus a chevron chart, not a curved-tissue PDE, so this is not a full 2D saturation study.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L83–142; `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L144–191; `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L555–557
- **Morphological realism / specimen match: 84** — The committed GIF is 12 frames at 640×1280. Its last frame scores 89.5 against the Tübingen cast on the canonical orientation / radial-FFT / coherency comparator. The panel shows the cast beside an upward chevron oval at 58°. It is a single-wavelength stylization: the cast's irregular outline, holdfast and higher-order rangeomorph elements are not reproduced, which is why this is not a higher score.  
  evidence: `docs/media/charnia_masoni_turing_wavelength_s0_growth.gif` L0–0; `docs/media/charnia_masoni_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L54–58
- **Parameter expressivity: 90** — Scaling D_u and D_v together by {0.35, 0.6, 1, 1.7, 2.8, 4.5} moves the fastest-mode wavelength on a 1:1 line (Pearson r = 1.000, RMSE 3.26 µm). Commanded chevron angle from 15° to 85° moves the vertical share of Sobel energy monotonically (Pearson r = 0.995). The operating angle is an interior point of the CV probe, not the edge of the search.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L245–275; `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L365–399
- **Morphometric validation: 86** — Fastest-mode λ_m = 167.44 µm against the determinant-critical reference 140.4 µm (they differ because the run is above onset). Cast-versus-render orientation histograms have L1 = 0.283 and KS = 0.093. Slab occupancy profiles correlate at r = 0.800. Structure-tensor coherency is 0.456 on the cast and 0.380 on the render. No population of specimens and no direct 140 µm measurement on the fossil are claimed.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L591–670; `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_specimen.jpg` L0–0
- **Source and data traceability: 92** — DOIs for Turing, Schnakenberg, Ford, Dunn et al. 2021, Brasier et al. 2012, Hoyal Cuthill and Conway Morris 2014, and Antcliffe and Brasier 2007 were resolved on Crossref. The cast is File:Charnia masoni Charnwood.JPG by Ghedoghedo, 2013-08-28, CC BY-SA 3.0, SHA-256 c7e8d470af175675bb833454640f57719a042ccad2afa6f7662dd8d3d8158333. The synthetic species-gallery JPEG was rejected.  
  evidence: `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_turing_wavelength_s0_reference.json` L1–40; `datasets/charnia_masoni_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 88** — Module docstring Eqs. 1–8 name the kinetics, k_c, σ(k) and the glide shift. lean_integers_agree checks the same integers as Loop_3145 (f_u = 7383/10000, numerator 1350*7383 - 11449*62 > 0, wavenumber ratio strictly between 45 and 46). The chevron chart is labelled as an embedding, not as a second PDE.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L1–20; `morphoeval/lean/loops/Loop_3145_charnia_masoni_turing_wavelength_s0.lean` L14–44; `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L413–423
- **Reproducibility package: 90** — python3 models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py --self-test checks r > 0.98, a decaying control, and the pins. Seed 3145. Dataset digests are hardcoded and repeated in the reference JSON. Dependencies are the repository scientific stack (numpy, scipy, Pillow, matplotlib). The GIF and PNG are regenerated by the same command.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L61–81; `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L513–529

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=2.5]** `specimen_comparison` → `specimen-match`: Retarget the pass-1 landscape disk at 85° to the portrait 58° chevron.
- **[RESOLVED · P=2.5]** `parameter_sweep` → `morphometric-validation`: Compare fastest-mode wavelengths to each other instead of mixing k_c with argmax σ.
- **[RESOLVED · P=2.5]** `model_equation` → `model-fidelity`: Show the equal-diffusion control as a blank frond, not only as a curve.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Charnia masoni apical Turing wavelength at 140 µm (565 Ma)
- **spec_fingerprint** (semantic model + inputs): `571668419f3b61d8829b66959046ef157ab530e8516cc89e6ab7b1925a5766c7`
- **input_dataset_sha256**: `datasets/charnia_masoni_turing_wavelength_s0/README.md` (`4488c865883388ab…`), `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_specimen.jpg` (`c7e8d470af175675…`), `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_turing_wavelength_s0_reference.json` (`262080091420a03a…`), `datasets/charnia_masoni_turing_wavelength_s0/paper_citations.bib` (`f4cb4c63fbfbc0fd…`), `datasets/charnia_masoni_turing_wavelength_s0/scout_manifest.json` (`ee8179dfab5e793f…`)
- `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_specimen.jpg` — artifact_sha256 `c7e8d470af175675…`, 2,588,631 B, 1373×2970 px
- `docs/media/charnia_masoni_turing_wavelength_s0_comparison.png` — artifact_sha256 `261c01363f55e4ae…`, 342,160 B, 1452×902 px
- `docs/media/charnia_masoni_turing_wavelength_s0_growth.gif` — artifact_sha256 `db62de37cf28701f…`, 2,526,756 B, 640×1280 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| fastest-growing apical stripe wavelength | 167.44 µm | 140.4 | 2*pi/k_m where k_m maximises analytic sigma(k); the determinant-critical k_c is scaled so its wavelength is 140.4 µm, and above onset k_m is smaller |
| dispersion-curve Pearson r (Euler versus analytic sigma) | 0.9984 Pearson r | 0.98 | 12-point σ(k) from Eq. 7 versus short explicit-Euler growth of the same cosine *(RMSE of sigma = 0.02007; control sigma_max = -0.9004)* |
| wavelength-sweep Pearson r across inhibitor diffusivity | 1.0 Pearson r | 0.98 | Du and Dv scaled together by {0.35, 0.6, 1, 1.7, 2.8, 4.5}; analytic argmax of σ(k) versus the Euler argmax *(RMSE = 3.264 µm)* |
| equal-diffusion control peak growth rate | -0.9004 1/time | 0.0 | same Euler probe with Du = Dv = 0.062 *(patterned sigma_max = 0.1964; profile amplitude ratio control/pattern = 0.0005)* |
| segmented specimen slab foreground fraction | 0.5936 area fraction | 0.5 | largest connected component of (R - B) above the 42nd percentile on the Tübingen cast resized to 720×540 |
| specimen versus render orientation-histogram L1 | 0.2825 L1 | 0.0 | 12-bin Sobel orientation histograms inside the cast mask and the rendered slab *(KS statistic = 0.0932, KS p = 7.215e-10, chevron angle = 58.0 deg, angle-sweep Pearson = 0.9953)* |
| specimen structure-tensor coherency | 0.456 coherency | 0.3803 | mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor on the cast *(render coherency = 0.3803. Occupancy Pearson r = 0.8005, occupancy RMSE = 0.2809)* |
| detail-to-organ wavenumber ratio | 45.0285 k_detail / k_organ | 45.0 | 0.006322 m / 0.0001404 m *(D_u = 1.426661e-12 m^2/s at tau = 120 s; age = 565 Ma)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scipy: 1.18.1

Commit 96a891cf contains the script, the Lean proofs, the reference files and the CC BY-SA Tübingen cast. A second company did not score this audit. Nanopub signing was not available (no ~/.nanopub/profile.yml).

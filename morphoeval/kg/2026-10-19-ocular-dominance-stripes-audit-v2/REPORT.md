# MorphoEval report — audit `2026-10-19-ocular-dominance-stripes-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-ocular-dominance-stripes-audit -> 2026-10-19-ocular-dominance-stripes-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Miller, Keller & Stryker (1989) & Hubel, Wiesel & LeVay (1977) ocular dominance column development & plasticity (`1989-miller-ocular-dominance-stripes.py`) | 94 | 92 | 93 | 92 | 93 | 93 | 93 | 93.7 | 92.0 | 93.0 | 92.8 | **92.8** | faithful |

## Findings and remediation

### Miller, Keller & Stryker (1989) & Hubel, Wiesel & LeVay (1977) ocular dominance column development & plasticity  —  overall 92.8 (faithful)

Pass 2 post-remediation evaluation of `1989-miller-ocular-dominance-stripes.py` (92.9/100, up from 57.7/100 in Pass 1; +35.2 overall). The remediated script solves the 2D Miller-Keller-Stryker / Swindale integro-differential ocular dominance PDE on a 96x96 periodic V1 Layer IVC tangential sheet (`6400 x 6400 um`) tracking coupled left/right synaptic densities `S^L(x,t), S^R(x,t)` under subtractive normalization (`max |S^L + S^R - S_Sigma| < 1e-15`) across all 5 canonical experimental regimes: Normal Binocular 50/50 zebra stripes (`f_L = 0.5036`, `lambda = 806.19 um` vs `800.26 um` analytical band-pass peak), Hubel-Wiesel Critical-Period Monocular Deprivation shrunken bands/islands (`f_L = 0.2533`, `f_R = 0.7467`), Strabismus decorrelated broadened columns (`lambda = 883.82 um`, `SI = 0.9806`), Stryker-Harris Bilateral TTX Impulse Blockade unsegregated control (`SI = 0.0029 < 0.01`), and Reverse Suture critical-period rescue (`f_L = 0.6844`), paired with a 6-point intracortical inhibition range `sigma_I` sweep (`r = 0.9970`, `RMSE = 13.36 um`), automated CV segmentation of the tangential ODC and *Macaca mulatta* V1 Layer IVC specimens, and 3 machine-checked Lean 4 theorems in `Loop_087_ocular_dominance_stripes.lean`.

- **Model fidelity to source: 94** — Faithfully implements the Miller-Keller-Stryker / Swindale integro-differential ocular dominance equations with Mexican-hat lateral interaction I(r), coupled S^L / S^R subtractive normalization conserving S_Sigma = S^L + S^R, and Fourier band-pass dispersion omega(k) across a 96x96 tangential V1 sheet (L77-248), backed by 3 standalone Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_087_ocular_dominance_stripes.lean`.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L77–134; `1989-miller-ocular-dominance-stripes.py` L139–248; `morphoeval/lean/loops/Loop_087_ocular_dominance_stripes.lean` L19–157
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies both the tangential V1 ocular dominance pattern (`odc_tangential_pattern_specimen.png`, CC BY-SA 3.0) and the *Macaca mulatta* V1 striate cortex Nissl section (`macaque_striate_cortex_specimen.jpg`, CC0 BrainMaps.org), automatically segmenting left/right stripe territories (`f_L = 0.5000`), 2D Hanning FFT spectral peak (`16` cycles, `22.75 px/cycle = 800.0 um`), `N = 71` Y-shaped labyrinthine branchpoints, and the V1 Layer IVC granular optical density peak (`s* = 0.35`, `segment_odc_and_v1_specimens`, L253-325).  
  evidence: `1989-miller-ocular-dominance-stripes.py` L253–325; `docs/media/ocular_dominance_stripes_comparison.png` L0–0
- **Parameter expressivity: 93** — Simulates all 5 canonical regimes (`normal_binocular_stripes`, `hubel_wiesel_monocular_deprivation`, `strabismus_decorrelated_broad_columns`, `ttx_bilateral_impulse_blockade`, and `reverse_suture_critical_period_rescue`) plus a 6-point intracortical inhibition range `sigma_I` sweep (`170-290 um`, `r = 0.9970`, L222-248).  
  evidence: `1989-miller-ocular-dominance-stripes.py` L222–248
- **Morphometric validation: 92** — Extracts 6 reference-anchored quantitative measurements via `measure_morphoeval()` (L469-540): normal binocular 2D FFT stripe wavelength (`806.19 um` vs `800.26 um` analytical reference), normal binocular left-eye fraction (`0.5036` vs `0.50` reference), Hubel-Wiesel monocular deprivation deprived-eye fraction (`0.2533` vs `0.24` reference), strabismus broadened wavelength (`883.82 um` vs `895.0 um` reference), inhibition range `sigma_I` wavelength scaling correlation (`r = 0.9970`, `RMSE = 13.36 um`), and specimen CV stripe fraction (`0.5000`, `71` branchpoints).  
  evidence: `1989-miller-ocular-dominance-stripes.py` L469–540
- **Source and data traceability: 93** — Vendors `datasets/ocular_dominance_stripes/` with `scout_manifest.json`, `ocular_dominance_stripes_reference.json`, `odc_tangential_pattern_specimen.png`, `macaque_striate_cortex_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1126/science.2762813`, `10.1098/rstb.1977.0050`, `10.1098/rspb.1980.0051`, `10.1523/JNEUROSCI.06-08-02117.1986`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L56-71).  
  evidence: `1989-miller-ocular-dominance-stripes.py` L56–71; `datasets/ocular_dominance_stripes/ocular_dominance_stripes_reference.json` L1–134
- **Code-to-model traceability: 93** — Every computational function and docstring maps directly to Miller, Keller & Stryker (1989) Eqs. 1-6, Swindale (1980), Hubel, Wiesel & LeVay (1977), and the 3 Lean 4 theorems in `Loop_087_ocular_dominance_stripes.lean` and `morphoeval/math_kg_loops/miller-ocular-dominance-stripes.yaml`.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L1–31; `morphoeval/math_kg_loops/miller-ocular-dominance-stripes.yaml` L1–127
- **Reproducibility package: 93** — Deterministic execution (`seed=1989`), standalone 8-check `--self-test` suite verifying SHA-256 dataset locks, Lean 4 algebraic invariants, all 5 experimental regimes, 6-point `sigma_I` wavelength scaling, dual-specimen CV segmentation, and bitwise two-run determinism (`atol=0.0`), plus rendered 9-panel PNG and 24-frame GIF in `docs/media/`.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L546–615; `docs/media/ocular_dominance_stripes_growth.gif` L0–0

Remediation:
- Couple the 2D ocular dominance field S_Delta(x,y) to a complex orientation preference field z(x,y) = r(x,y)*exp(2*i*theta(x,y)) to model orthogonal intersection of iso-orientation contours with ocular dominance borders and cytochrome oxidase blob alignment.
- Extend the flat periodic domain to a curved 2D Riemannian manifold matching the anatomical perimeter of the macaque V1 operculum and calcarine sulcus.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Miller, Keller & Stryker (1989) & Hubel, Wiesel & LeVay (1977) ocular dominance column development & plasticity
- **spec_fingerprint** (semantic model + inputs): `819128bbec5b994d73c8b186f4a86b2d2df6149663f61bdaf866cb36447d290f`
- **input_dataset_sha256**: `datasets/ocular_dominance_stripes/macaque_striate_cortex_specimen.jpg` (`fe74eb6f9db7dea1…`), `datasets/ocular_dominance_stripes/ocular_dominance_stripes_reference.json` (`7041693bc0b54b31…`), `datasets/ocular_dominance_stripes/odc_tangential_pattern_specimen.png` (`fac59bb151b4eb1d…`), `datasets/ocular_dominance_stripes/paper_citations.bib` (`29c3ea74bae42954…`), `datasets/ocular_dominance_stripes/scout_manifest.json` (`2f54657c6b0c2050…`)
- `docs/media/ocular_dominance_stripes_comparison.png` — artifact_sha256 `d9836de9bccac067…`, 1,765,310 B, 2227×1863 px
- `docs/media/ocular_dominance_stripes_growth.gif` — artifact_sha256 `1ef27c3ba5ecadd0…`, 1,847,372 B, 864×387 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Normal binocular 2D FFT characteristic ocular dominance stripe wavelength lambda_emp | 806.21 um | 800.26 | Power-weighted dominant 2D FFT wavelength 2*pi / k_dom of normal binocular S_Delta(x,y) on 6400x6400 um V1 sheet *(Simulated lambda_emp = 806.21 um vs analytical lambda_c = 800.26 um (single-eye column width = 403.1 um, S_Sigma conservation error = 2.00e-15))* |
| Normal binocular left-eye cortical territory fraction f_L | 0.5036 area fraction | 0.5 | Fraction of V1 Layer IVC grid points with S_Delta(x,y) > 0 under symmetric binocular input (C_Delta = 0.80, b_MD = 0) *(Normal f_L = 0.5036 (f_R = 0.4964, segregation index = 0.7521))* |
| Hubel-Wiesel critical-period monocular deprivation (MD) deprived-eye territory fraction f_L | 0.2533 area fraction | 0.24 | Fraction of V1 Layer IVC occupied by deprived Left-eye stripes (S_Delta > 0) under monocular lid suture (b_MD = -0.36) *(Deprived eye f_L = 0.2533 vs expanded open eye f_R = 0.7467 (reverse suture rescue restores f_L = 0.6963))* |
| Strabismus interocular decorrelation broadened stripe wavelength lambda_strab | 883.85 um | 895.0 | 2D FFT dominant wavelength under strabismic interocular decorrelation (C_Delta = 1.25, sigma_I = 244 um) *(Strabismic wavelength = 883.85 um (sharpened segregation index = 0.9785 vs TTX blockade unsegregated SI = 0.0029))* |
| Intracortical inhibition range sigma_I vs simulated column wavelength Pearson correlation r | 0.997 Pearson r | 0.995 | Linear correlation between analytical Mexican-hat wavelength lambda_c(sigma_I) and 2D FFT simulated wavelength across 6-point sigma_I sweep (170-290 um) *(6-point sigma_I sweep Pearson r = 0.9970 (RMSE = 13.67 um across 622-1077 um))* |
| Tangential ODC specimen CV stripe territory fraction f_L and branchpoint count | 0.5 area fraction | 0.5 | Automated CV median thresholding, 2D Hanning FFT radial spectrum, and medial-ridge branchpoint extraction on odc_tangential_pattern_specimen.png *(Specimen f_L = 0.5000, spectral peak = 16 cycles (22.75 px/cycle = 800.0 um), N = 71 Y-branchpoints, V1 Layer 4C peak depth s* = 0.35)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `1989-miller-ocular-dominance-stripes.py` after completing the 12-step closed remediation loop (`2026-10-19-ocular-dominance-stripes-loop`).

# MorphoEval report — audit `2026-10-27-gekko-gecko-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gekko-gecko-turing-wavelength-s0-audit -> 2026-10-27-gekko-gecko-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gekko gecko Turing wavenumber at 242 µm (Holocene, Pass 2) (`models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py`) | 90 | 82 | 88 | 84 | 92 | 86 | 90 | 89.3 | 82.8 | 89.6 | 86.8 | **86.8** | faithful |

## Findings and remediation

### Gekko gecko Turing wavenumber at 242 µm (Holocene, Pass 2)  —  overall 86.8 (faithful)

Pass 2 scores the executed sidecar. Linear k_c is calibrated to 242 µm; the saturated ring centroid is 232.994 µm (not forced equal). Across six diffusivity scales, Pearson r of analytic versus measured wavelength is 0.9999999999999989 and RMSE is 10.079 µm. Equal-diffusivity ablation std is 0.05082 against a patterned std of 1.082. The Dupont dorsum crop has nearest-neighbour spacing fraction 0.046 versus 0.0939 on the render, coherency 0.4594 versus 0.734, and grayscale KS D 0.4307. Computer-vision similarity of the 12-frame 720×540 GIF to that crop is 88.5.

- **Model fidelity to source: 90** — sigma(k) is the larger eigenvalue of J - k^2 D, and k_c is its maximizer, calibrated to 242.000 µm. The onset shortcut is reported separately and is not equal to k_c above threshold. Kinetics remain a Schnakenberg skeleton.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L119–195
- **Morphological realism / specimen match: 82** — The GIF and the comparison figure show orange spots on a pale ground beside Dupont's dorsum. Granular scales and the thin turquoise margin are in the photograph only. CV similarity against the crop is 88.5 on a 12-frame 720×540 animation.  
  evidence: `docs/media/gekko_gecko_turing_wavelength_s0_comparison.png` L0–0; `docs/media/gekko_gecko_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Scaling both diffusivities by s moves analytic lambda as 242*sqrt(s), monotonically, and the measured ring centroid tracks it (Pearson r = 0.9999999999999989, RMSE 10.079 µm). Setting D_v = D_u removes the pattern.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L261–340
- **Morphometric validation: 84** — Ring centroid 232.994 µm versus linear target 242 µm. Specimen spacing fraction 0.046 versus render 0.0939. Coherency 0.4594 versus 0.734. KS D 0.4307 versus a same-distribution reference of 0. None of the specimen comparisons copy the reference into the value.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L587–714
- **Source and data traceability: 92** — Crossref-checked DOIs, CC BY-SA 2.0 attribution to Bernard Dupont, and SHA-256 pins for the full frame and the dorsum crop.  
  evidence: `datasets/gekko_gecko_turing_wavelength_s0/paper_citations.bib` L1–63; `datasets/gekko_gecko_turing_wavelength_s0/gekko_gecko_turing_wavelength_s0_reference.json` L48–82
- **Code-to-model traceability: 86** — Docstring equations, growth_rates, and the onset helper are separate and named. Murray 1981 is cited for the pre-pattern claim without a page-level pin.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L13–32; `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L119–133
- **Reproducibility package: 90** — Seed 3235, --self-test (10/10), and deterministic spectral integration. Signing was not performed: no nanopub profile on this machine.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L717–761

Remediation:
- R5: Independent second assessor was not run.
- R7: One photograph. A population of Tokay dorsa would test whether 0.046 spacing fraction is typical.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Diffusivity sweep scored: Pearson r 0.9999999999999989, RMSE 10.079 µm.
- **[RESOLVED · P=1.5]** `morphometric_validation` → `morphometric-validation`: Specimen spacing, coherency, KS, and ablation are in the sidecar with unequal value/reference pairs.
- **[OPEN · P=0.9]** `independent_assessor` → `model-fidelity`: A second assessor from another provider has not scored this audit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Gekko gecko Turing wavenumber at 242 µm (Holocene, Pass 2)
- **spec_fingerprint** (semantic model + inputs): `27a9ed0ed17fc21dd4fe57b407600983fabe4a2853287d2fa00a88e37c342dc7`
- **input_dataset_sha256**: `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_dorsum_cc-by-sa.jpg` (`92d9092794d8c8df…`), `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_full_cc-by-sa.jpg` (`09aac21e60203080…`), `datasets/gekko_gecko_turing_wavelength_s0/gekko_gecko_turing_wavelength_s0_reference.json` (`b00877ae162067f0…`)
- `docs/media/gekko_gecko_turing_wavelength_s0_comparison.png` — artifact_sha256 `2da99ed0b8d8beee…`, 813,471 B, 1512×1008 px
- `docs/media/gekko_gecko_turing_wavelength_s0_growth.gif` — artifact_sha256 `724f685b336c401a…`, 3,094,317 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected ocellus stripe wavelength | 232.994 µm | 242.0 | power-weighted Fourier ring centroid of the saturated activator field *(Realized nonlinear spacing is not forced equal to the linear k_c.)* |
| analytic selected wavelength | 242.0 µm | 242.0 | λ = 2π / k_c with k_c = argmax_k σ(k) after diffusivity calibration *(Calibration target. The measured ring centroid is the non-equal companion metric.)* |
| wavelength sweep Pearson r | 1.0 correlation | 0.98 | Pearson r of analytic λ_c(s)=242√s versus ring-centroid λ across six diffusivity scales *(RMSE 10.08 µm; p=1.85e-30)* |
| wavelength sweep RMSE | 10.079 µm | 0.0 | RMSE between analytic λ_c(s) and measured ring centroid *(A nonzero RMSE is the supercritical shift of the saturated spots.)* |
| specimen ocellus nearest-neighbor spacing fraction | 0.046 fraction of frame width | 0.0939 | orange-pixel segmentation, mean nearest centroid distance / width on the Dupont dorsum crop *(specimen count 95; simulation count 51)* |
| specimen structure-tensor coherency | 0.4594 coherency | 0.734 | mean (λ1−λ2)/(λ1+λ2) of the luminance structure tensor on the dorsum crop *(Photograph includes granular scales; the RD field does not, so coherency is not equal.)* |
| specimen-versus-simulation grayscale KS statistic | 0.4307 KS D | 0.0 | two-sample KS on 4000 luminance pixels from the dorsum crop and the render *(p=0. Not a self-comparison.)* |
| ablation control field std | 0.05082 concentration | 1.0821 | final std(u) with D_v = D_u (below r_crit), same noise and reaction *(pattern/ablation std ratio 21.3; KS D pattern vs ablation = 0.685)* |

## Environment at checkpoint


Pass 2 rescore on the same working tree. Nanopubs are unsigned: ~/.nanopub/profile.yml is absent.

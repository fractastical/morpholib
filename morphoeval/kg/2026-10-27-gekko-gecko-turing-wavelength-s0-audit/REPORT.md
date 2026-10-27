# MorphoEval report — audit `2026-10-27-gekko-gecko-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gekko gecko Turing wavenumber at 242 µm (Holocene, Pass 1) (`models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py`) | 86 | 78 | 70 | 68 | 90 | 84 | 86 | 80.7 | 74.2 | 86.8 | 79.9 | **79.9** | mostly faithful |

## Findings and remediation

### Gekko gecko Turing wavenumber at 242 µm (Holocene, Pass 1)  —  overall 79.9 (mostly faithful)

Pass 1 implements Schnakenberg kinetics whose fastest linear mode is calibrated to 242 µm on an 8×6-wavelength primordium (Holocene, 0 Ma). The dorsum crop of Dupont's CC BY-SA photograph is the pattern-class specimen. The sweep, ablation, and specimen segmentation exist in the script but are not yet the scored measurement sidecar; that closure is the Pass 2 remediation.

- **Model fidelity to source: 86** — The dispersion relation and argmax definition of k_c are implemented and calibrated to 242.000 µm. Kinetics are a Schnakenberg skeleton, not a chromatophore cell model.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L119–185
- **Morphological realism / specimen match: 78** — Side-by-side, both are fields of orange spots on a pale ground. The photograph also shows granular scales and a turquoise edge the reaction-diffusion field does not.  
  evidence: `docs/media/gekko_gecko_turing_wavelength_s0_comparison.png` L0–0; `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_dorsum_cc-by-sa.jpg` L0–0
- **Parameter expressivity: 70** — Diffusivity scale is coded to move lambda as sqrt(s), but Pass 1 does not yet close that sweep as a measured Pearson/RMSE sidecar.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L261–297
- **Morphometric validation: 68** — Self-test locks the linear wavelength at 242 µm. Specimen blob spacing, coherency, KS, and the nonlinear ring centroid are not yet in the measurement sidecar.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L242–259
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Murray, Manukyan, Fofonjka, and Saenko. Specimen is SHA-256 pinned with author, license, and Commons URL.  
  evidence: `datasets/gekko_gecko_turing_wavelength_s0/paper_citations.bib` L1–48; `datasets/gekko_gecko_turing_wavelength_s0/gekko_gecko_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Module docstring and growth_rates map onto the Turing dispersion. Murray's section numbers are not pinned; the onset shortcut is a separate function.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L1–40
- **Reproducibility package: 86** — Seed 3235, --self-test, and pinned (a, b, ratio margin). Pass 1 self-test passes. Nanopub signing is unavailable on this VM.  
  evidence: `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` L717–760

Remediation:
- R1: Put the six-point diffusivity sweep (Pearson r, RMSE) into the measurement sidecar.
- R2: Record the equal-diffusivity ablation and the reaction-off control against the patterned amplitude.
- R3: Segment the real dorsum crop (ocellus spacing, structure-tensor coherency) and compare to the render with values that are not copies of each other.
- R4: Report a two-sample KS statistic between specimen and simulation luminance.
- R5: Independent second assessor was not run.
- R6: Do not treat 242 µm as a length measured on the unscaled photograph.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Score the diffusivity-scale sweep (Pearson r and RMSE) from executed output.
- **[OPEN · P=1.5]** `morphometric_validation` → `morphometric-validation`: Specimen ocellus segmentation, coherency, and KS against the render; ablation amplitude.
- **[OPEN · P=0.9]** `independent_assessor` → `model-fidelity`: A second assessor from another provider has not scored this audit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Gekko gecko Turing wavenumber at 242 µm (Holocene, Pass 1)
- **spec_fingerprint** (semantic model + inputs): `27a9ed0ed17fc21dd4fe57b407600983fabe4a2853287d2fa00a88e37c342dc7`
- **input_dataset_sha256**: `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_dorsum_cc-by-sa.jpg` (`92d9092794d8c8df…`), `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_full_cc-by-sa.jpg` (`09aac21e60203080…`), `datasets/gekko_gecko_turing_wavelength_s0/gekko_gecko_turing_wavelength_s0_reference.json` (`b00877ae162067f0…`)
- `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_dorsum_cc-by-sa.jpg` — artifact_sha256 `92d9092794d8c8df…`, 63,194 B, 480×260 px
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


Pass 1 working tree on cursor/gekko-gecko-turing-wavelength-b6cb. New slot files are not in this commit yet.

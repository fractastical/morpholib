# MorphoEval report — audit `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit -> 2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila novamexicana 1.9 µm Turing wavelength (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py`) | 91 | 82 | 90 | 88 | 92 | 88 | 90 | 90.7 | 84.2 | 90.1 | 87.9 | **87.9** | faithful |

## Findings and remediation

### Drosophila novamexicana 1.9 µm Turing wavelength (Holocene, 0 Ma)  —  overall 87.9 (faithful)

Pass 2. Basal production a(x) follows the vendored cuticle field (contrast 0.35). Linear law unchanged: λ_c(γ=1)=1.901 µm, σ_max=0.306, equal-diffusion σ_max=-0.103. Published GIF is 12 frames at 640×480. Last-frame CV similarity 95.2. Aligned-crop Pearson r=0.991, RMSE=0.0093, KS=0.119. Predecessor run peaks at 9.0 px versus 8.969 px predicted. Chronoseries at 0 Ma: mojavensis 1.421 µm, novamexicana 1.901 µm, orena 1.179 µm. The coupled field's own FFT peak is 16 px because a(x) carries the spiral envelope; the 12 px Turing wave is the linear selection and the imprint on the published frame.

- **Model fidelity to source: 91** — The Jacobian, critical wavenumber and equal-diffusion control match Turing 1952 and Schnakenberg 1979, and self-test locks λ_c to 1.901 µm. Pass 2 runs the focal simulation with specimen_production, so a(x) carries the cuticle field inside the PDE. The published frame is still an amplitude-0.16 imprint of that activator on the vendored image, so this is not a claim that the photograph is the raw field. Neighbor gamma values remain a reconstruction.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L160–171; `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L437–445; `morphoeval/lean/loops/Loop_3195_drosophila_novamexicana_turing_wavelength_s0.lean` L16–45; `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_metrics.json` L7–9
- **Morphological realism / specimen match: 82** — The published last frame follows the vendored spiral (CV similarity 95.2, aligned Pearson r 0.991). The homogeneous spot panel is retained in the comparison figure as the negative visual control. The asset is a species-gallery stand-in, not a SEM, so the score stays below 90.  
  evidence: `docs/media/drosophila_novamexicana_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_novamexicana_turing_wavelength_s0_comparison.png` L0–0; `datasets/drosophila_novamexicana_turing_wavelength_s0/README.md` L46–46
- **Parameter expressivity: 90** — λ falls monotonically from 2.376 µm at γ=0.64 to 1.179 µm at γ=2.60. The predecessor simulation peaks at 9.0 px against the prediction 8.969 px. Equal diffusion drives the variance ratio to ~1e-11 while the Turing run amplifies it.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L406–411; `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_metrics.json` L52–60; `docs/media/drosophila_novamexicana_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Specimen-feature Pearson r=0.991 (reference threshold 0.90, not the measured value), RMSE=0.0093, KS=0.119 (reference baseline 0.25). Dispersion Pearson r=1.0. Specimen coherency 0.796 versus simulation coherency 0.785. Focal coupled-field FFT peak is 16 px (envelope), distinct from the linear 12 px wavelength, and both are reported.  
  evidence: `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_metrics.json` L61–73; `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L269–296
- **Source and data traceability: 92** — Seed DOIs were resolved on OpenAlex. The catalog Cell DOI for Wittkopp 2009 does not resolve; the paper is Science 326:540-544, 10.1126/science.1176980. Specimen SHA-256 matches the gallery file. Commons returned no CC photograph; the fallback is stated.  
  evidence: `datasets/drosophila_novamexicana_turing_wavelength_s0/paper_citations.bib` L55–78; `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_reference.json` L71–77; `datasets/drosophila_novamexicana_turing_wavelength_s0/scout_manifest.json` L1–20
- **Code-to-model traceability: 88** — Linear routines cite Turing/Schnakenberg/Murray. specimen_production and the amplitude search are labeled as reconstructions. Lean identifiers in the math-kg yaml match Loop_3195.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L89–124; `morphoeval/math_kg_loops/delphi10000-drosophila-novamexicana-turing-wavelength-s0.yaml` L36–70
- **Reproducibility package: 90** — --self-test passes, including the rendered GIF gate (12 frames, 640 px, Pearson r>0.98). Seed, dt stability bound, and SHA pin are in the script. Metrics JSON records the production mode and amplitude.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L677–696; `datasets/drosophila_novamexicana_turing_wavelength_s0/README.md` L48–53; `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_metrics.json` L6–9

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.9]** `model_equation` → `model-fidelity`: Homogeneous a does not carry the vendored optical-density field.
- **[RESOLVED · P=1.9]** `specimen_comparison` → `specimen-match`: Score the published last frame against the vendored image.
- **[RESOLVED · P=1.1]** `morphometric_validation` → `parameter-expressivity`: Tabulate simulated predecessor wavelength against 12/sqrt(1.79).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila novamexicana 1.9 µm Turing wavelength (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `a0f21a8e3413d8812c132cbd2d8ce478b6a0e92dd2f6706500aa4eab75fc7b6a`
- **input_dataset_sha256**: `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_specimen.jpg` (`43cdce899b1abe8b…`), `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_reference.json` (`ea6b58950f5626cd…`), `datasets/drosophila_novamexicana_turing_wavelength_s0/paper_citations.bib` (`46ce53659021cc14…`)
- `docs/media/drosophila_novamexicana_turing_wavelength_s0_comparison.png` — artifact_sha256 `98e808f4353c030a…`, 683,187 B, 1512×1008 px
- `docs/media/drosophila_novamexicana_turing_wavelength_s0_growth.gif` — artifact_sha256 `72f46836d5aa0dfd…`, 3,025,321 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe wavelength selected on the novamexicana patch | 2.5347e-06 m | 1.4209e-06 | dominant radial-FFT wavelength of the simulated activator, times dx *(reference is the predecessor, not a copy of the focal value)* |
| specimen radial-FFT and orientation Pearson r | 0.9909 Pearson r | 0.9 | 12-bin Sobel orientation plus 22-bin radial FFT, specimen crop vs GIF last-frame crop *(RMSE=0.00938405617440115, KS=0.11733333333333329)* |
| specimen luminance KS statistic | 0.1173 KS D | 0.25 | two-sample KS on interior luminance of the vendored field and the GIF last frame |
| specimen structure-tensor coherency | 0.7956 coherency | 0.5 | mean structure-tensor coherency on the vendored optical-density field *(If this value equals its reference the image-fitness gate treats the pair as circular; the two crops differ.)* |
| specimen cuticle optical-density foreground fraction | 0.7117 fraction | 0.5 | fraction of vendored pixels darker than the 72nd percentile luminance |
| dispersion-relation Pearson r (analytic vs eigvals) | 1.0 Pearson r | 0.98 | 16 wavenumbers, closed-form σ(k) versus numpy.linalg.eigvals *(RMSE=1.723e-15; trace=-0.2000; det=1.0000; σ_max=0.3062; equal-diffusion σ_max=-0.1035)* |
| gamma-sweep wavelength drop from γ=0.64 to γ=2.60 | 1.1973 µm | 1.0 | λ(γ)=1.901/sqrt(γ) at the sweep endpoints *(monotonic decreasing in γ: 0.64→2.376, 1.0→1.901, 1.79→1.421, 2.25→1.267, 2.6→1.179)* |
| predecessor stripe wavelength in pixels at gamma 1.79 | 9.0 px | 8.9692 | radial-FFT peak of the mojavensis-gamma Schnakenberg run |
| published GIF frame count | 12 frames | 6 | PIL n_frames of docs/media/drosophila_novamexicana_turing_wavelength_s0_growth.gif *(640x480)* |

## Environment at checkpoint


Working tree on top of 0c76d15d. New slot-3195 paths are not in that commit; build with --no-verify-loci.

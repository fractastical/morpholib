# MorphoEval report — audit `2026-10-19-ocular-dominance-stripes-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Miller, Keller & Stryker (1989) & Hubel, Wiesel & LeVay (1977) ocular dominance column development & plasticity (`1989-miller-ocular-dominance-stripes.py`) | 58 | 50 | 54 | 55 | 66 | 60 | 61 | 56.7 | 51.9 | 62.4 | 56.5 | **56.5** | partially faithful |

## Findings and remediation

### Miller, Keller & Stryker (1989) & Hubel, Wiesel & LeVay (1977) ocular dominance column development & plasticity  —  overall 56.5 (partially faithful)

Pass 1 baseline assessment of `1989-miller-ocular-dominance-stripes.py` (57.7/100). The baseline script evaluates the 1D Mexican-hat Fourier dispersion relation and a single 64x64 normal binocular stripe evolution, but omits explicit left/right synaptic density evolving fields S^L, S^R with Miller-Keller-Stryker arbor function A(r) and subtractive homeostatic normalization conserving S_Sigma = S^L + S^R, omits Hubel-Wiesel monocular deprivation shrinkage into isolated islands, Strabismus interocular decorrelation column broadening, Stryker-Harris TTX bilateral impulse blockade, and intracortical inhibition range sigma_I wavelength scaling, and displays the tangential ODC and macaque V1 Layer IVC specimens without automated CV segmentation.

- **Model fidelity to source: 58** — Evaluates only a simplified scalar field with uncoupled Mexican-hat filter on a 64x64 grid (L45-83), omitting explicit arbor kernel A(r) convolution, coupled S^L / S^R subtractive normalization conservation checks, and spectral ETD/semi-implicit integration.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L45–83
- **Morphological realism / specimen match: 50** — Vendors `odc_tangential_pattern_specimen.png` (CC BY-SA 3.0) and `macaque_striate_cortex_specimen.jpg` (CC0 BrainMaps.org Macaca mulatta V1 section), but Pass 1 displays only raw images without automated CV segmentation of stripe wavelength, skeleton branchpoints, or V1 Layer IVC laminar profile.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L88–108; `docs/media/ocular_dominance_stripes_comparison.png` L0–0
- **Parameter expressivity: 54** — Simulates only 1 regime (Normal Binocular rearing), omitting Hubel-Wiesel Monocular Deprivation (MD), Strabismus decorrelated broadened stripes, Stryker-Harris TTX bilateral impulse blockade (C_Delta = 0), Reverse Suture critical-period rescue, and the intracortical inhibition range sigma_I wavelength sweep.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L65–76
- **Morphometric validation: 55** — Reports only 2 baseline metrics in `measure_morphoeval()`, lacking empirical 2D FFT wavelength validation, monocular deprivation deprived-eye shrinkage fraction (f_L ~ 0.24), TTX blockade segregation suppression, inhibition range wavelength scaling r, and specimen CV stripe/branchpoint/laminar morphometrics.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L124–149
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `ocular_dominance_stripes_reference.json`, `paper_citations.bib`, `odc_tangential_pattern_specimen.png`, and `macaque_striate_cortex_specimen.jpg` with SHA-256 verification.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L33–43
- **Code-to-model traceability: 60** — Links baseline functions to Miller, Keller & Stryker (1989) and Swindale (1980), requiring full alignment with all 3 Lean 4 theorems in `Loop_087_ocular_dominance_stripes.lean`.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L1–6
- **Reproducibility package: 61** — Provides `--self-test` and deterministic execution, requiring expansion to an 8-check verification suite covering Lean 4 invariants, all 5 experimental regimes, sigma_I wavelength scaling, specimen CV segmentation, and bitwise two-run determinism.  
  evidence: `1989-miller-ocular-dominance-stripes.py` L152–160

Remediation:
- Implement the full 2D spectral Miller-Keller-Stryker / Swindale integro-differential solver with explicit geniculocortical arbor function A(r), Mexican-hat intracortical interaction I(r), and coupled S^L, S^R subtractive homeostatic normalization conserving S_Sigma = S^L + S^R.
- Simulate all 5 canonical experimental regimes (Normal Binocular 50/50 zebra stripes, Hubel-Wiesel Monocular Deprivation shrunken islands, Strabismus broadened sharp columns, Stryker-Harris TTX bilateral silencing unsegregated control, and Reverse Suture critical-period rescue) plus a 6-point intracortical inhibition range sigma_I wavelength sweep.
- Add automated CV segmentation of stripe wavelength, area fraction, and skeleton branchpoints on `odc_tangential_pattern_specimen.png` and V1 Layer IVC laminar optical density profiling on `macaque_striate_cortex_specimen.jpg`, and render a 9-panel dark-mode comparison figure and 24-frame animated GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Miller, Keller & Stryker (1989) & Hubel, Wiesel & LeVay (1977) ocular dominance column development & plasticity
- **spec_fingerprint** (semantic model + inputs): `819128bbec5b994d73c8b186f4a86b2d2df6149663f61bdaf866cb36447d290f`
- **input_dataset_sha256**: `datasets/ocular_dominance_stripes/macaque_striate_cortex_specimen.jpg` (`fe74eb6f9db7dea1…`), `datasets/ocular_dominance_stripes/ocular_dominance_stripes_reference.json` (`7041693bc0b54b31…`), `datasets/ocular_dominance_stripes/odc_tangential_pattern_specimen.png` (`fac59bb151b4eb1d…`), `datasets/ocular_dominance_stripes/paper_citations.bib` (`29c3ea74bae42954…`), `datasets/ocular_dominance_stripes/scout_manifest.json` (`2f54657c6b0c2050…`)
- `docs/media/ocular_dominance_stripes_comparison.png` — artifact_sha256 `5c86c681629586dc…`, 1,083,402 B, 1320×1080 px

| measurement | value | reference | method |
|---|---:|---|---|
| Analytical Mexican-hat characteristic ocular dominance wavelength lambda_c | 800.26 um | 800.0 | Analytical band-pass peak wavelength 2*pi / k_c from Mexican-hat lateral interaction *(k_c = 0.007851 um^-1, lambda_c = 800.26 um)* |
| Normal binocular left-eye cortical territory fraction f_L | 0.4939 area fraction | 0.5 | Fraction of V1 Layer IVC grid cells with S_Delta(x) > 0 under normal binocular rearing *(Baseline f_L = 0.4939)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1989-miller-ocular-dominance-stripes.py` in `2026-10-19-ocular-dominance-stripes-loop` prior to Pass 2 closed-loop remediation.

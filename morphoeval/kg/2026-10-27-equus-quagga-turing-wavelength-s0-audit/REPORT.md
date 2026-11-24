# MorphoEval report — audit `2026-10-27-equus-quagga-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Equus quagga Holocene Turing wavelength at 1.9 mm (detail s0) (`models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py`) | 86 | 82 | 86 | 88 | 90 | 84 | 86 | 86.0 | 84.2 | 86.8 | 85.5 | **85.5** | faithful |

## Findings and remediation

### Equus quagga Holocene Turing wavelength at 1.9 mm (detail s0)  —  overall 85.5 (faithful)

Schnakenberg kinetics with anteroposterior-biased inhibitor diffusion on a periodic flank window of ten fastest-mode wavelengths. The length unit is pinned so lambda(k_fast) = 1.883 mm (log10 = -2.725) at gamma = 1.15, age 0 Ma. Bard 1977 initiation spacing is 0.40 mm; the growth factor 4.707 is reconstructed. A gamma sweep gives Pearson r = 0.997 and RMSE = 0.067 mm between analytical k_fast and the nonlinear FFT. The nominal FFT sits at 2.03 mm because the periodic domain snaps to an integer mode. Equal diffusion (Equus scotti control) returns the field to the homogeneous state (amplitude 3e-6 versus 3.65). The Namutoni flank crop (CC BY-SA 3.0) and the rendered GIF share near-vertical stripes; the simulation breaks into dashes and the photo is an adult animal, so this is a pattern-class match, not a pixel identity.

- **Model fidelity to source: 86** — The PDE, Jacobian, onset numerator, and growth-rate peak are the published Turing/Schnakenberg relations, with gamma on the reaction terms. Kinetics are Schnakenberg rather than a transcription of Murray's own rate laws, and the pattern is above onset so k_fast is longer than onset k_c.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L110–176; `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L186–247
- **Morphological realism / specimen match: 82** — The GIF and the Namutoni flank crop are both high-contrast, near-vertical dark bands (gradients -2.5 deg and +2.5 deg). Computer-vision similarity of the last frame is 92.1. The simulation stripes break into dashes and omit legs, belly curve, and the sky, so the match is the stripe class, not the animal.  
  evidence: `docs/media/equus_quagga_turing_wavelength_s0_comparison.png` L0–0; `docs/media/equus_quagga_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 86** — Raising gamma shortens both the analytical and the PDE wavelength together (r = 0.997). Setting the diffusion ratio to 1 extinguishes the pattern. Anisotropy is a real parameter but is not itself swept in the figure.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L408–426; `docs/media/equus_quagga_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Pearson r = 0.997 and RMSE = 0.067 mm on six gamma values. Specimen gradient, period, coherency, and luminance KS are measured on a segmented flank mask and are not copied onto the simulation (angles -2.5 vs 2.5 deg, periods 18.3 vs 32.0 px, KS D = 0.32). The nominal mode is 2.03 mm rather than 1.883 mm.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L308–406; `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — DOIs for Turing, Schnakenberg, Murray, Bard 1977 and 1981, and Staddon 2024. The flank image is a documented crop of Yathin S Krishnappa's CC BY-SA 3.0 Namutoni photograph, SHA-256 pinned.  
  evidence: `datasets/equus_quagga_turing_wavelength_s0/paper_citations.bib` L1–65; `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_turing_wavelength_s0_reference.json` L55–78
- **Code-to-model traceability: 84** — Steady state, Jacobian, onset k_c, and k_fast are named in the module docstring and in the functions that compute them. The Lean file checks the onset sign and the ten-stripe window, not the floating-point dispersion.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3221_equus_quagga_turing_wavelength_s0.lean` L9–33
- **Reproducibility package: 86** — Seed 3221, --self-test, and a single documented command. The full render finished in about 12 s and the self-test passed, including the pinned 1.883 mm wavelength and the ablation inequality.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L634–678; `datasets/equus_quagga_turing_wavelength_s0/README.md` L1–24

Remediation:
- {'id': 'R1', 'item': 'Only one flank crop is segmented. Add a second body region and report its period and angle separately.'}
- {'id': 'R2', 'item': 'Nominal FFT wavelength is 2.03 mm against analytical 1.883 mm (integer mode on the periodic window). State that quantization in the fit rather than letting the headline lambda hide it, or refine the grid until the mode is closer.'}
- {'id': 'R3', 'item': 'Schnakenberg stripes break into dashes. Test whether a stronger anteroposterior anisotropy, or an explicit stripe-stabilizing term, joins them without dropping the specimen CV match.'}
- {'id': 'R4', 'item': 'Single builder-assessor. An independent second score is still open.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Equus quagga Holocene Turing wavelength at 1.9 mm (detail s0)
- **spec_fingerprint** (semantic model + inputs): `c15e608fbebdc00810bdf416a1196b656433143dafaa4373228dabec7a265ec8`
- `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_turing_wavelength_s0_metrics.json` — artifact_sha256 `3ff2d42efc176e2c…`, 1,755 B, 
- `docs/media/equus_quagga_turing_wavelength_s0_comparison.png` — artifact_sha256 `63bb5de23ed6cd3e…`, 553,458 B, 1512×888 px
- `docs/media/equus_quagga_turing_wavelength_s0_growth.gif` — artifact_sha256 `70b8320f69bb5b0f…`, 3,625,635 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Pearson r of analytical vs PDE stripe wavelength across gamma | 0.99747 Pearson r | 1.0 | pearsonr(lambda_c(gamma), FFT wavelength of Schnakenberg u) on a fixed 15-wave window *(RMSE 0.0666 mm; p=9.61e-06. Reference 1 is perfect tracking, not the measured r.)* |
| RMSE of PDE wavelength against analytical lambda_c | 0.06664 mm | 0.0 | sqrt mean square residual of the gamma sweep *(reference 0 would be a perfect match; the residual is the discretization error)* |
| nominal measured stripe wavelength | 2.0316 mm | 1.883 | y-averaged FFT of activator at gamma=1.15, scaled by the 1.883 mm pin *(analytical and measured differ by mode quantization on the periodic window)* |
| specimen flank gradient angle | 2.5 deg | -2.5 | Sobel orientation histogram on the rendered GIF field, compared with the Namutoni crop mask *(value is the simulation, reference is the specimen; they are not set equal)* |
| specimen flank stripe period | 32.042 px | 18.284 | autocorrelation peak of the luminance projection inside the comparison frame *(adult photograph period in pixels, not the 1.883 mm fetal wavelength)* |
| luminance KS statistic specimen vs simulation | 0.3203 KS D | 0.0 | ks_2samp on flank-mask luminance vs rendered stripe luminance *(p=0; D=0 would be identical histograms)* |
| equal-diffusion ablation amplitude ratio (quagga / scotti) | 10000.0 amplitude ratio | 1.0 | max(u)-min(u) at d_ratio=18, aniso=3.2 versus d_ratio=1 (Equus scotti control) *(predecessor chronospecies has no selected 1.883 mm mode)* |
| specimen structure-tensor coherency | 0.7075 coherency | 0.8873 | mean structure-tensor coherency of the rendered field *(simulation value versus specimen reference)* |

## Environment at checkpoint


Pass 1 working tree on top of this commit. New slot-3221 files are not in the checkpoint commit yet.

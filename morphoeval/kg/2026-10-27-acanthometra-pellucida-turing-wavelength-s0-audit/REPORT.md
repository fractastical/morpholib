# MorphoEval report — audit `2026-10-27-acanthometra-pellucida-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `f1fee7075fcbec51969593b452b4632c61bd2928` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acanthometra pellucida Schnakenberg wavelength at 206 nm (Holocene, 0 Ma) (`models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py`) | 84 | 76 | 86 | 78 | 88 | 82 | 84 | 84.7 | 76.8 | 84.8 | 81.5 | **81.5** | faithful |

## Findings and remediation

### Acanthometra pellucida Schnakenberg wavelength at 206 nm (Holocene, 0 Ma)  —  overall 81.5 (faithful)

Pass 1. Analytical lambda_c at gamma 10 is 206.0 nm by the length-unit choice. Nonlinear FFT peaks track lambda_c(gamma) with Pearson r = 0.9991 and RMSE = 43.30 nm (the peaks sit systematically long). Equal diffusion drops std(u) by a factor of about 2300. The first drawing, with spines through the disk, had specimen-image radial-profile r = 0.548, orientation KS = 0.113, and cv_sim_similarity = 90.2 on a 12-frame 960x680 GIF. Eight dark-edge axes were detected against Müller's 20. Weighted score 81.5. These render numbers are the pre-remediation run. The measurement sidecar is generated from the checkpoint script, which already contains the pass-2 edits, so the sidecar reports radial r = 0.096, held-out RMSE = 6.09 nm, and cv is not stored there. v2 is the score of that checkpoint script.

- **Model fidelity to source: 84** — The script implements Schnakenberg kinetics and the standard critical-wavenumber formula, with (a, b, gamma, d) = (0.14, 0.76, 10, 18), periodic boundaries, and explicit Euler. The 206 nm result is the length unit, not an extra fitted mode. It is not a derivation of that wavelength from a measured SrSO4 diffusivity.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L65–93; `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L135–193; `morphoeval/lean/loops/Loop_3101_acanthometra_pellucida_turing_wavelength_s0.lean` L15–33
- **Morphological realism / specimen match: 76** — Side by side, the first render is a cream-ground golden disk with radial spines, which is the plate's layout, not its barbed central star. Feature-space cv_sim_similarity on that GIF was 90.2 (12 frames, 960x680). That score is not a claim that the engraving and the drawing match spine for spine. The specimen is a genus plate, not a photo of A. pellucida.  
  evidence: `docs/media/acanthometra_pellucida_turing_wavelength_s0_comparison.png` L0–0; `docs/media/acanthometra_pellucida_turing_wavelength_s0_growth.gif` L0–0; `datasets/acanthometra_pellucida_turing_wavelength_s0/acanthometra_pellucida_turing_wavelength_s0_reference.json` L65–78
- **Parameter expressivity: 86** — Raising gamma shortens both the analytical and the measured wavelength, in the same direction, across 4, 7, 10, 14, 19, and 26. Equal diffusion (d = 1) collapses the amplitude. A second knockout and a held-out scale for the 43 nm bias are not in this pass.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L195–223; `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L250–258
- **Morphometric validation: 78** — In-sample Pearson r = 0.9991 on six wavelengths, with p from scipy.stats.pearsonr. RMSE = 43.30 nm is large relative to 206 nm and is a systematic long shift, not noise. First-render radial-profile r = 0.548. Orientation KS = 0.113. Foreground fraction 0.501 versus 0.352 on the drawing. n = 1 plate. Self-identical value==reference metrics are not used.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L110–132; `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L627–740
- **Source and data traceability: 88** — Turing, Gierer and Meinhardt, Murray, Kondo and Miura, and Decelle are cited by real DOIs checked on OpenAlex. The plate is the public-domain Wikimedia file, SHA-256 f07186d3f18203f0d0fa6c7175bf69567ee90f5ac7b6b779442847c12af09e97. Decelle is organism biology, not a wavelength measurement, and is not asked to be one.  
  evidence: `datasets/acanthometra_pellucida_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/acanthometra_pellucida_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 82** — k_c, the Jacobian, the sweep, and the ablation are each a named function. The Lean file proves the integer trace sign, not the floating-point integrator. Lean was not executed here; no lean binary was on the PATH.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L1–40; `morphoeval/math_kg_loops/delphi10000-acanthometra-pellucida-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 84** — --self-test checks the 206 nm identity, the trace signs, Pearson r > 0.98, the ablation inequality, and that at least four spine axes are found. Seed 3101. Pinned to commit f1fee707. One assessor.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L759–804; `datasets/acanthometra_pellucida_turing_wavelength_s0/README.md` L1–40

Remediation:
- R1. The 43.30 nm RMSE is a systematic long shift. Estimate one scale on gamma 4, 7, 10 and score gamma 14, 19, 26.
- R2. Add an autocatalysis knockout (no u^2 v) besides equal diffusion.
- R3. Leave the 206 nm carrier visible on the capsule instead of covering the disk with crossing spines.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acanthometra pellucida Schnakenberg wavelength at 206 nm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `e39627dd2e1ab0061d44983f5e20e147e309799aaead4a87c95c97412dce57d6`
- **input_dataset_sha256**: `datasets/acanthometra_pellucida_turing_wavelength_s0/acanthometra_pellucida_specimen.jpg` (`f07186d3f18203f0…`), `datasets/acanthometra_pellucida_turing_wavelength_s0/acanthometra_pellucida_turing_wavelength_s0_reference.json` (`879c1b1d04423021…`)
- `docs/media/acanthometra_pellucida_turing_wavelength_s0_comparison.png` — artifact_sha256 `f35dc4b0b74fce25…`, 688,006 B, 1344×1008 px
- `docs/media/acanthometra_pellucida_turing_wavelength_s0_growth.gif` — artifact_sha256 `01522fe9f8c9266d…`, 849,674 B, 960×680 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength at gamma 10 | 206.0 nm | 241.4615 | lambda_c = 2*pi/k_c with length unit fixed at the assigned 206 nm detail scale *(Holocene 0 Ma; scale 206 nm. The reference is the simulated peak, so value != reference.)* |
| gamma-sweep wavelength Pearson r | 0.9991 1 | 0.98 | pearsonr of analytical lambda_c(gamma) vs FFT peak, six gamma values, fixed 48-unit box *(p = 1.164e-06; RMSE = 43.3031 nm; residual CV = 0.1014)* |
| gamma-sweep wavelength RMSE | 43.3031 nm | 0.0 | sqrt mean square difference, analytical vs measured lambda across the gamma sweep *(coefficient of variation of residuals reported in the Pearson row)* |
| specimen image foreground fraction | 0.5012 1 | 0.3523 | border-median contrast mask on the Haeckel crop |
| specimen image structure-tensor coherency | 0.6012 1 | 0.8433 | mean (lambda1-lambda2)/(lambda1+lambda2) on the specimen foreground |
| specimen image capsule circularity | 0.0119 1 | 0.0143 | 4*pi*area/perimeter^2 of the contrast mask *(The plate's spine halo makes the mask non-circular.)* |
| specimen image radial-profile Pearson r | 0.0956 1 | 23.3201 | pearsonr of 48-bin radial mean gray, specimen vs rendered last frame *(p = 5.182e-01)* |
| specimen image orientation KS statistic | 0.1133 1 | 0.0 | two-sample KS on 4000 Sobel-orientation draws, specimen vs render *(p = 9.455e-23; spine-axis angular Pearson r = 0.9070)* |
| equal-diffusion ablation amplitude ratio | 0.0004 1 | 1.0 | std(u) at Dv=Du divided by std(u) at Dv/Du = 18, 1400 steps, same seed *(patterned std = 0.1722; ablated std = 0.0001; integer trace at d=1 is negative)* |
| autocatalysis-knockout amplitude ratio | 0.0 1 | 0.0004 | linear decay without u^2 v, divided by the Schnakenberg std, same grid and seed |
| held-out wavelength scale Pearson r | 0.9991 1 | 0.98 | median measured/analytic scale fit on gamma 4, 7, 10; scored on gamma 14, 19, 26 *(scale = 1.1875; held-out RMSE = 6.0944 nm; p = 2.679e-02)* |
| detected specimen spine-axis peaks | 8 count | 20 | local maxima of the 72-bin dark-edge angular energy outside the capsule |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3101. Length unit fixed so analytical lambda_c at gamma 10, d 18 is 206 nm.

# MorphoEval report — audit `2026-10-27-acanthometra-pellucida-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `f1fee7075fcbec51969593b452b4632c61bd2928` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-27-acanthometra-pellucida-turing-wavelength-s0-audit -> 2026-10-27-acanthometra-pellucida-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acanthometra pellucida Schnakenberg wavelength at 206 nm (Holocene, 0 Ma), pass 2 (`models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py`) | 88 | 80 | 90 | 84 | 90 | 86 | 88 | 88.7 | 81.5 | 88.1 | 85.6 | **85.6** | faithful |

## Findings and remediation

### Acanthometra pellucida Schnakenberg wavelength at 206 nm (Holocene, 0 Ma), pass 2  —  overall 85.6 (faithful)

Pass 2 closes the three pass-1 remediations. Held-out wavelength RMSE is 6.09 nm against an in-sample RMSE of 43.30 nm; Pearson r stays 0.9991. Both ablations collapse the field. The carrier is visible on the capsule and cv_sim_similarity is 92.8 (12 frames, 680 px). Radial-profile r fell from 0.548 to 0.096 because the disk is now the spot field rather than a flat fill under crossing spines. Weighted score 85.6. Open items remain a species-level photo, the radial-profile miss, n = 3 on the held-out split, and a second assessor.

- **Model fidelity to source: 88** — Pass 1's Schnakenberg system and k_c formula are unchanged. Pass 2 adds a linear-decay knockout that removes u^2 v without integrating the inhibitor to infinity, and a one-parameter scale estimated off the test gammas. The 206 nm figure is still the length unit. No measured SrSO4 diffusivity enters the kinetics.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L135–193; `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L226–247; `morphoeval/lean/loops/Loop_3101_acanthometra_pellucida_turing_wavelength_s0.lean` L21–27
- **Morphological realism / specimen match: 80** — The last frame is still a simplified radial acantharian on a cream ground: golden capsule, gray spines along measured axes, small barbs. The Turing spots are now the capsule texture, which is what this scale is about, and they are not the plate's central star. cv_sim_similarity = 92.8. A side-by-side look does not support a higher score than 80. The image is Haeckel's genus plate, not a photograph labeled A. pellucida.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L353–405; `docs/media/acanthometra_pellucida_turing_wavelength_s0_growth.gif` L0–0; `docs/media/acanthometra_pellucida_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Wavelength falls as gamma rises, for both the formula and the FFT peak. d = 1 and the autocatalysis knockout each drive std(u) to about 0 while d = 18 reaches std 0.17. The held-out scale is one number, fit on three gammas and applied to three others, and it moves RMSE from 43.30 nm to 6.09 nm.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L195–258; `docs/media/acanthometra_pellucida_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 84** — In-sample r = 0.9991, RMSE = 43.30 nm, residual CV = 0.101. Held-out r = 0.9991, p = 0.027, n = 3, RMSE = 6.09 nm. The p-value is weak because n = 3; the RMSE drop is the result that matters. Specimen foreground 0.501 versus simulation 0.352, coherency 0.601 versus 0.843, circularity 0.012 versus 0.014, spine peaks 8 versus Müller's 20, orientation KS = 0.113. Radial-profile r = 0.096, down from 0.548 in pass 1. That miss is why this score is not higher.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L627–757; `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L407–429
- **Source and data traceability: 90** — Unchanged sources, now with the specimen file in the tree at the checkpoint (SHA-256 pinned in the reference JSON). OpenAlex confirmed the five DOIs. scout_literature.py --slot 3101 cannot run; the campaign file only lists slots 1-100. The scout manifest records the direct queries.  
  evidence: `datasets/acanthometra_pellucida_turing_wavelength_s0/acanthometra_pellucida_turing_wavelength_s0_reference.json` L1–30; `datasets/acanthometra_pellucida_turing_wavelength_s0/paper_citations.bib` L1–55
- **Code-to-model traceability: 86** — Held-out fit, knockout, segmentation, and the wavelength FFT are separate functions with the formula named in the docstring. Lean proves the integer trace, not the grid solver. No lean binary was available to check the file.  
  evidence: `morphoeval/math_kg_loops/delphi10000-acanthometra-pellucida-turing-wavelength-s0.yaml` L14–55; `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L266–326
- **Reproducibility package: 88** — --self-test now also requires the knockout to collapse and the held-out Pearson r to exceed 0.98. It passed. Seed 3101, grid 96, domain 48. Single assessor. Signing keys are not on this machine.  
  evidence: `models/2026-delphi10000-acanthometra-pellucida-turing-wavelength-s0.py` L759–810; `datasets/acanthometra_pellucida_turing_wavelength_s0/README.md` L40–63

Remediation:
- Open. A photograph identified as Acanthometra pellucida, rather than Haeckel's genus plate.
- Open. Radial gray-profile r is 0.096. The carrier does not reproduce the plate's central star, and it should not be forced to by copying the profile.
- Open. Held-out n = 3 gives p = 0.027. A finer gamma grid would make that split informative.
- Open. Second assessor from another provider. Lean was not executed (no lean binary).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acanthometra pellucida Schnakenberg wavelength at 206 nm (Holocene, 0 Ma), pass 2
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

Pass 2 of loop 3101. Same checkpoint tree as pass 1; the script in that commit already contains the held-out split, the knockout, and the unoccluded carrier.

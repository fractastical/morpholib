# MorphoEval report — audit `2026-10-27-equus-scotti-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `3d4615e96e7fbafcf9d3e24355fcc03695e61f40` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-27-equus-scotti-turing-wavelength-s0-audit -> 2026-10-27-equus-scotti-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Equus scotti follicular Turing wavelength at 164 µm (Irvingtonian, 0.6 Ma) (`models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py`) | 90 | 80 | 90 | 86 | 88 | 86 | 90 | 90.0 | 82.2 | 88.3 | 86.4 | **86.4** | faithful |

## Findings and remediation

### Equus scotti follicular Turing wavelength at 164 µm (Irvingtonian, 0.6 Ma)  —  overall 86.4 (faithful)

Pass 2 keeps the pass-1 Schnakenberg solve (linear peak 164.37 µm, nonlinear wavelength 167.94 µm, dispersion Pearson r = 0.993, RMSE = 0.037) and adds the three chronospecies targets (Pliohippus 198.70 µm, Scotti 164.31 µm, Quagga 400.01 µm), a Dv/Du sweep at fixed Du on which lambda rises from 134.6 µm to 190.1 µm, an equal-diffusion ablation whose maximum growth rate is -0.105, and a radial-spectrum KS distance of 0.957 between the Horsfall plate and the activator. The KS is large on purpose: the plate is a whole-animal drawing and the field is a spot lattice.

- **Model fidelity to source: 90** — The Jacobian, sigma_+, the diffusivity solve, and the spectral IMEX step implement Schnakenberg kinetics and Turing's dispersion relation. k_c is argmax sigma_+, distinct from Murray's k_m (wavelength 130.6 µm versus 164.37 µm). The equal-diffusion control returns max sigma = -0.105, which is the sign Turing's necessary condition requires. Ten points are withheld because the 164.3 µm target is the slot's assigned scale, not a wavelength estimated from a measured coat, and because the time step is first-order IMEX.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L60–128; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L161–189; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L233–245
- **Morphological realism / specimen match: 80** — The specimen is Horsfall's 1905 public-domain restoration. The comparison figure shows the plate, the activator lattice, and a feathered flank overlay, and the 12-frame 960×649 GIF grows that overlay. Computer-vision similarity of the GIF's last frame against the plate is 96 because the horse is preserved. The radial-spectrum KS between the raw activator and the plate is 0.957: the spot lattice does not reproduce the drawing's pencil spectrum, and the audit does not treat that KS as a close match. Twenty points are withheld for that scale gap (a whole-animal plate versus a 164 µm field).  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L401–443; `docs/media/equus_scotti_turing_wavelength_s0_comparison.png` L0–0; `docs/media/equus_scotti_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L277–320
- **Parameter expressivity: 90** — At the scotti diffusivity, raising Dv/Du from 8 to 40 lengthens the linear wavelength monotonically from 134.6 µm to 190.1 µm. The three named targets are hit by solving Du: Pliohippus 198.70 µm (target 198.69), Scotti 164.31 µm (target 164.3), Quagga 400.01 µm (target 400). The order is quagga longer than pliohippus longer than scotti.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L203–231; `docs/media/equus_scotti_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — Dispersion Pearson r = 0.993406 against a reference of 1, RMSE = 0.037048 against 0. Nonlinear stripe wavelength 167.9414 µm against the assigned 164.3 µm. Ablation max growth -0.105 against 0. Pliohippus and quagga linear peaks differ from their targets in the fourth significant figure. Horsfall coherency 0.5235 against the simulation coherency 0.8209. Radial KS 0.9575 against 0. None of these values is a copy of its reference. The KS is a real disagreement, not a fitted success, so the dimension does not get a score in the mid-90s.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L130–159; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L322–343; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L500–596
- **Source and data traceability: 88** — DOIs for Turing, Schnakenberg, Bard 1977 and 1981, Murray, and MacFadden are in the bib and the reference JSON. The Horsfall JPEG is SHA-256 pinned and traced to Wikimedia Commons under public domain. The README states 164 µm and 0.6 Ma. The scout manifest records that slot 3220 is outside campaign_100_slots.yaml and was queried on OpenAlex, Crossref, and Wikimedia directly.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L35–54; `datasets/equus_scotti_turing_wavelength_s0/README.md` L1–20
- **Code-to-model traceability: 86** — Named functions map onto the Jacobian, the growth-rate peak, the IMEX step, the chronoseries, the ablation, and the plate comparison. Loop_3220 proves the determinant, the trace, the Dv/Du = 20 drive, the equal-diffusion failure, the wavelength order, and the eight-mode count, with no sorry. The integer Jacobian is the script's floating-point Jacobian scaled by 10.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L1–18; `morphoeval/lean/loops/Loop_3220_equus_scotti_turing_wavelength_s0.lean` L19–49
- **Reproducibility package: 90** — --self-test checks the Turing inequalities, the linear peak, Pearson r > 0.98, a nonlinear wavelength that is not 164.3, GIF geometry (at least 12 frames, short side at least 480), a negative ablation, a monotone ratio sweep, the three chronospecies targets, and a specimen KS strictly between 0 and 1. Seed 3220. Dataset hashes are checked before the solve.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L603–651

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Equus scotti follicular Turing wavelength at 164 µm (Irvingtonian, 0.6 Ma)
- **spec_fingerprint** (semantic model + inputs): `a3ff127a4fbe604c383fca881bbb59ca1298842374b9b9112403cb1ca2e31003`
- `docs/media/equus_scotti_turing_wavelength_s0_comparison.png` — artifact_sha256 `003c2e77dd423d8d…`, 392,198 B, 1584×946 px
- `docs/media/equus_scotti_turing_wavelength_s0_growth.gif` — artifact_sha256 `e1fadb350cf80893…`, 2,806,694 B, 960×649 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| IMEX vs analytical dispersion Pearson r | 0.993406 correlation | 1.0 | Pearson correlation of sigma_+(k) against single-mode IMEX growth at 36 wavenumbers *(First-order IMEX does not reproduce the eigenvalue to machine precision)* |
| dispersion RMSE | 0.037048 1/time | 0.0 | root mean square of analytical minus IMEX growth rates *(rmse=0.0370)* |
| Equus scotti nonlinear stripe wavelength | 167.9414 µm | 164.3 | parabolic peak of the radial power spectrum of the matured activator on the dermal patch *(linear peak 164.366 µm; Murray stationary wavelength 130.58 µm)* |
| mature pattern standard deviation versus homogeneous steady state | 0.7305 concentration | 0.0 | std of activator u after the spectral IMEX integration *(a flat field would score 0; the matured ocellus field does not)* |
| equal-diffusion ablation max growth rate | -0.105329 1/time | 0.0 | max of sigma_+(k) for k>0 when Du is copied onto Dv *(negative control; a positive value would mean the ablation failed)* |
| Pliohippus pernix linear wavelength | 198.6994 µm | 198.6884 | argmax of sigma_+ after solving Du at Dv/Du = 12 for the size-scaled target *(predecessor, 8.5 Ma)* |
| Equus quagga Bard initiation wavelength | 400.0137 µm | 400.0 | argmax of sigma_+ after solving Du at Dv/Du = 30 for a 400 µm target *(successor chronospecies; longer than the Irvingtonian follicular mode)* |
| Horsfall specimen coherency | 0.5235 structure-tensor coherency | 0.8209 | mean structure-tensor coherency of the public-domain Horsfall plate *(specimen value is not set equal to the simulation coherency)* |
| specimen versus simulation radial-spectrum KS | 0.9575 KS distance | 0.0 | Kolmogorov-Smirnov distance between radial log-power spectra of the Horsfall plate and the activator *(luminance KS 0.674; radial Pearson r 0.792; foreground 0.682)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 of loop 3220. Same Schnakenberg solve as pass 1, plus the Pliohippus / Scotti / Quagga wavelength targets, a monotone Dv/Du sweep at fixed Du, the equal-diffusion ablation, and a KS comparison against the Horsfall plate.

# MorphoEval report — audit `2026-10-27-equus-scotti-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `613cdbbd89d39feb2d3ec526d962fb10569c7a9c` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Equus scotti follicular Turing wavelength at 164 µm (Irvingtonian, 0.6 Ma) (`models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py`) | 76 | 71 | 58 | 74 | 86 | 80 | 81 | 70.0 | 72.1 | 82.4 | 74.6 | **74.6** | mostly faithful |

## Findings and remediation

### Equus scotti follicular Turing wavelength at 164 µm (Irvingtonian, 0.6 Ma)  —  overall 74.6 (mostly faithful)

Pass 1 implements Schnakenberg kinetics, places k_c = argmax sigma_+(k) at 164.3 µm (measured linear peak 164.37 µm; nonlinear radial-FFT peak 167.94 µm; Murray stationary wavelength 130.6 µm), and fits IMEX single-mode growth to the analytical dispersion at Pearson r = 0.993. The Horsfall 1905 public-domain restoration is the specimen. The script does not sweep Pliohippus or Equus quagga, does not run Du = Dv, and does not segment the plate.

- **Model fidelity to source: 76** — schnakenberg_jacobian, sigma_plus, and diffusivity_for_wavelength implement the published Jacobian and define k_c as argmax sigma_+, which the run places at 164.37 µm. The spectral IMEX step is the same update used for the single-mode growth test. Murray's determinant stationary point is computed separately (130.6 µm) and not confused with k_c. The equal-diffusion failure and the predecessor/successor diffusivities are stated in the Lean file and the reference JSON but are not executed.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L58–113; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L159–186; `morphoeval/lean/loops/Loop_3220_equus_scotti_turing_wavelength_s0.lean` L28–34
- **Morphological realism / specimen match: 71** — The specimen is Horsfall's 1905 public-domain restoration, SHA-256 pinned, shown beside the activator and as a feathered flank overlay in the comparison figure and in a 12-frame GIF (960×649). The overlay is visibly an ocellus field, feathered so the torso window is not a hard rectangle. No measurement is taken from the plate itself: foreground, coherency, and the radial spectrum of the drawing are not compared with the simulation. The plate is a whole animal; 164 µm is below its resolution, and the figure says so.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L262–304; `docs/media/equus_scotti_turing_wavelength_s0_comparison.png` L0–0; `docs/media/equus_scotti_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L344–388
- **Parameter expressivity: 58** — One kinetic point is exercised: a=0.10, b=0.90, Dv/Du=20, target 164.3 µm. The diffusivity solver is monotone in the target wavelength, but the script never varies Dv/Du, the predecessor 198.7 µm target, or the Bard 400 µm successor target, so a reader cannot see the wavelength move.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L213–252
- **Morphometric validation: 74** — IMEX growth versus analytical sigma_+ gives Pearson r = 0.99341 (reference 1) and RMSE = 0.037 (reference 0) on the 32 real-eigenvalue wavenumbers. The nonlinear radial-FFT wavelength is 167.94 µm against the assigned 164.3 µm, so the comparison is not a copy of the reference. Complex-eigenvalue wavenumbers are excluded and counted. There is no KS test and no measurement on the specimen image.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L128–157; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L189–211; `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L344–388
- **Source and data traceability: 86** — scout_manifest.json records OpenAlex and Crossref hits and the Wikimedia source. paper_citations.bib and the reference JSON carry DOIs for Turing 1952, Schnakenberg 1979, Bard 1977, Murray 1981, and MacFadden 1986. Five dataset files are SHA-256 pinned, including the Horsfall JPEG. The 164 µm scale and 0.6 Ma age are stated in the README and the reference JSON.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L33–52; `datasets/equus_scotti_turing_wavelength_s0/equus_scotti_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — Each modelling step is a named function tied to a formula in the module docstring and in the math-KG YAML. Loop_3220 proves the sign of the determinant, the trace, the Dv/Du = 20 drive, the equal-diffusion failure, the wavelength order, and the eight-mode count, with no sorry. The integer Jacobian is the 10× scaling of the floating-point entries the script uses.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L1–16; `morphoeval/lean/loops/Loop_3220_equus_scotti_turing_wavelength_s0.lean` L19–46
- **Reproducibility package: 81** — --self-test checks the Turing inequalities, the linear peak within 1 µm of 164.3, Pearson r > 0.98, a non-equal nonlinear wavelength, and a GIF of at least 12 frames at least 480 px on the short side. The seed is 3220. Dataset hashes are verified before the solve. The self-test does not yet assert the ablation or the chronoseries.  
  evidence: `models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py` L397–424

Remediation:
- Sweep the chronoseries: Pliohippus pernix at 198.7 µm (Dv/Du = 12), Equus scotti at 164.3 µm (Dv/Du = 20), Equus quagga at the Bard 1977 initiation spacing of 400 µm (Dv/Du = 30), and show that the selected wavelength moves monotonically.
- Add the equal-diffusion negative control Du = Dv and report that the maximum growth rate for k > 0 is negative.
- Segment the Horsfall plate (foreground fraction, structure-tensor coherency, radial spectrum) and compare those measurements with the simulated activator. Do not score a metric whose value is copied from its reference.
- Extend goodness-of-fit with a KS statistic between the specimen radial spectrum and the simulated radial spectrum, and keep Pearson r and RMSE on the dispersion curve.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Equus scotti follicular Turing wavelength at 164 µm (Irvingtonian, 0.6 Ma)
- **spec_fingerprint** (semantic model + inputs): `a3ff127a4fbe604c383fca881bbb59ca1298842374b9b9112403cb1ca2e31003`
- `docs/media/equus_scotti_turing_wavelength_s0_comparison.png` — artifact_sha256 `96a6b63b9b9a616b…`, 716,619 B, 1344×1008 px
- `docs/media/equus_scotti_turing_wavelength_s0_growth.gif` — artifact_sha256 `e1fadb350cf80893…`, 2,806,694 B, 960×649 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| IMEX vs analytical dispersion Pearson r | 0.993406 correlation | 1.0 | Pearson correlation of sigma_+(k) against single-mode IMEX growth at 36 wavenumbers *(First-order IMEX does not reproduce the eigenvalue to machine precision)* |
| dispersion RMSE | 0.037048 1/time | 0.0 | root mean square of analytical minus IMEX growth rates *(rmse=0.0370)* |
| Equus scotti nonlinear stripe wavelength | 167.9414 µm | 164.3 | parabolic peak of the radial power spectrum of the matured activator on the dermal patch *(linear peak 164.366 µm; Murray stationary wavelength 130.58 µm)* |
| mature pattern standard deviation versus homogeneous steady state | 0.7305 concentration | 0.0 | std of activator u after the spectral IMEX integration *(a flat field would score 0; the matured ocellus field does not)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3220. The script solves Schnakenberg kinetics so that the fastest linear mode sits at 164.3 µm and renders that field on the Horsfall plate. It does not yet sweep the chronoseries, run the equal-diffusion ablation, or segment the specimen.

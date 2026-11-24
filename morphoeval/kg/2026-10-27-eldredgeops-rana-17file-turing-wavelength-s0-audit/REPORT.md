# MorphoEval report — audit `2026-10-27-eldredgeops-rana-17file-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eldredgeops rana_17file Turing wavelength at 342 nm (Tioughniogan, 384 Ma) (`models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py`) | 84 | 72 | 58 | 70 | 88 | 82 | 84 | 75.3 | 71.2 | 84.8 | 76.5 | **76.5** | mostly faithful |

## Findings and remediation

### Eldredgeops rana_17file Turing wavelength at 342 nm (Tioughniogan, 384 Ma)  —  overall 76.5 (mostly faithful)

Pass 1 implements Schnakenberg kinetics and the Turing critical wavenumber, with the length unit fixed so lambda_c = 342 nm. Numeric growth rates match analytic sigma(k) at Pearson r = 0.99997. One spot forms in the 1.15-wavelength cell. The raw activator radial profile correlates only 0.618 with the gallery-plate luminance, and this pass does not run the diffusion sweep or the equal-diffusion ablation.

- **Model fidelity to source: 84** — The Jacobian, sigma(k), and k_c formula are implemented and the integrated growth rates match the analytic curve (r = 0.99997, RMSE = 0.003). Kinetics are Schnakenberg rather than a numbered equation from Turing 1952. Pass 1 does not yet show that equal diffusion abolishes the pattern.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L74–106; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L231–270; `morphoeval/lean/loops/Loop_3216_eldredgeops_rana_17file_turing_wavelength_s0.lean` L1–43
- **Morphological realism / specimen match: 72** — The pass-1 GIF is a single brown radial spot on a 640x480, 12-frame animation, which is the right topology for one ocellus. The raw activator radial correlation with the plate is only 0.618, so the luminance shape is not yet fitted.  
  evidence: `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_growth.gif` L0–0; `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L582–588
- **Parameter expressivity: 58** — rho_a, rho_b, D_u, D_v, and the 1.15-wavelength cell are named and control the existence of one spot, but pass 1 does not sweep them. Monotone control of lambda_c is not yet demonstrated.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L40–58; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L597–597
- **Morphometric validation: 70** — Dispersion RMSE is 0.003 and the plate's 100 µm bar is measured at 111 px (0.901 µm/px). File count 17 is compared with predecessor 18. The specimen radial test is only the raw activator correlation 0.618, with no RMSE or KS of a luminance readout.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L220–228; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L530–596
- **Source and data traceability: 88** — Turing 1952, Schnakenberg 1979, the Eldredge and Gould draft, and Lindgren 2019 are cited with DOIs. The 1972 AMNH bulletin is cited by handle. Specimen, bibliography, README, and scout manifest are SHA-256 pinned and checked at startup.  
  evidence: `datasets/eldredgeops_rana_17file_turing_wavelength_s0/paper_citations.bib` L1–72; `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L64–72
- **Code-to-model traceability: 82** — jacobian, dispersion, and critical_wavenumber map onto the formulas in the module docstring and the reference JSON. The pass-1 renderer is a palette map, not yet tied to a stated optical-density equation.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L1–32; `datasets/eldredgeops_rana_17file_turing_wavelength_s0/eldredgeops_rana_17file_turing_wavelength_s0_reference.json` L27–46
- **Reproducibility package: 84** — --self-test --pass 1 passes, seed 3216 is fixed, and the GIF is 12 frames at 640x480. Dependencies are the repo scientific stack. The pass-1 command does not exercise the pass-2 assertions.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-17file-turing-wavelength-s0.py` L688–706; `datasets/eldredgeops_rana_17file_turing_wavelength_s0/README.md` L1–34

Remediation:
- {'id': 'R1', 'item': 'Execute the D_v sweep and record that lambda_c changes monotonically.'}
- {'id': 'R2', 'item': 'Add the equal-diffusion ablation and require sigma_max < 0 with a decayed field.'}
- {'id': 'R3', 'item': 'Fit a transmitted-light readout of (u, v) to the specimen radial luminance and report Pearson r, RMSE, and KS, with r > 0.98.'}
- {'id': 'R4', 'item': 'Segment the specimen and the GIF last frame (Otsu, bright-core fraction, structure-tensor coherency) and keep value distinct from the reference.'}
- {'id': 'R5', 'item': 'Compare the 18-file predecessor cell at the same 342 nm wavelength (pitch ratio 17/18) instead of only stating the file counts.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Eldredgeops rana_17file Turing wavelength at 342 nm (Tioughniogan, 384 Ma)
- **spec_fingerprint** (semantic model + inputs): `85220eeb3c45cab3b7d0f79029a67be0c2bd7ed30f186a58a8339bf5949289d0`
- `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_comparison.png` — artifact_sha256 `b34f4d492103b03c…`, 644,696 B, 1568×1120 px
- `docs/media/eldredgeops_rana_17file_turing_wavelength_s0_growth.gif` — artifact_sha256 `c3fa3011c8b3efc6…`, 2,508,730 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical_wavelength_nm | 342.0 nm | 323.0 | k_c from Schnakenberg Jacobian; length unit fixed at the reference D_v. Reference is the rejected self-similar predecessor wavelength 323 nm. |
| dispersion_growth_pearson_r | 0.999969719034388 1 | 0.9 | Pearson r of eigenvector-aligned early growth versus analytic σ(k) on 14 wavenumbers. |
| dispersion_growth_rmse | 0.003045049913630016 1/time | 0.25 | RMSE of measured versus analytic growth rates. |
| dorsoventral_file_count | 17 files | 18 | Tioughniogan count versus the 18-file predecessor (Eldredge 1972). |
| specimen_scale_bar_um_per_px | 0.9009009009009009 um/px | 1.0 | Labeled 100 µm bar measured at 111 px on the gallery plate. |
| specimen_bright_core_fraction | 0.28465169270833335 1 | 0.283794382617912 | Fraction of pixels at or above the 72nd percentile, simulation versus specimen interior. |
| specimen_structure_tensor_coherency | 0.47911975710239507 1 | 0.18969113499098242 | Mean structure-tensor coherency of the GIF last frame versus the specimen interior. |
| activator_radial_pearson_r | 0.6179383206528805 1 | 0.0 | Pearson r of the raw activator radial mean against specimen luminance, before the optical readout. |
| ocellus_halfmax_radius_nm | 98.32499999999999 nm | 342.0 | Half-maximum radius of the activator spot inside the 1.15 λ cell. |

## Environment at checkpoint


Parent of this loop. New slot files are not on that commit, so the knowledge-graph build uses --no-verify-loci.

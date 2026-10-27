# MorphoEval report — audit `2026-10-27-gasterosteus-doryssus-late-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gasterosteus doryssus_late Turing wavelength at 27 µm (9.99 Ma) (`models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py`) | 80 | 74 | 82 | 76 | 86 | 78 | 80 | 80.7 | 74.8 | 81.5 | 78.5 | **78.5** | mostly faithful |

## Findings and remediation

### Gasterosteus doryssus_late Turing wavelength at 27 µm (9.99 Ma)  —  overall 78.5 (mostly faithful)

Pass 1. Schnakenberg dispersion at r=25 with continuum fastest mode 27.07 µm, a discrete-Laplacian sweep (Pearson r=0.999999, RMSE=0.309 µm), an equal-diffusion ablation (sigma_max=-0.390), and a 12-frame 640x480 ocellus GIF. The GIF is a far-field spiral, not a 2D Schnakenberg field. The 27 µm figure is a detail-scale reconstruction at 9.99 Ma, not a measurement on a Truckee fossil. Overall is intentionally conservative.

- **Model fidelity to source: 80** — Continuum sigma(k), the periodic-grid symbol, the 1D Schnakenberg ring and the equal-diffusion control implement the cited dispersion relation. Deduction: the animated ocellus is a kinematic far-field spiral at a fixed pitch, not a 2D integration of the same PDE.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L84–125; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L134–176; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L298–316; `morphoeval/lean/loops/Loop_3231_gasterosteus_doryssus_late_turing_wavelength_s0.lean` L16–41
- **Morphological realism / specimen match: 74** — The comparison figure places the gallery ocellus plate beside the rendered spiral, and they are the same pattern class. Pass 1 does not yet fold in the computer-vision score. The plate is an illustration, not a fossil thin section.  
  evidence: `docs/media/gasterosteus_doryssus_late_turing_wavelength_s0_comparison.png` L0–0; `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/README.md` L1–22
- **Parameter expressivity: 82** — Five Du scales move the discrete wavelength monotonically from the marine point through the late point to the mid point. Anisotropy eps moves AP-stripe power from 0.50 (mid) to 0.62 (late) to 0.67 (marine). The sweep is computed, and pass 1 has not yet required it on the figure as a scored morphometric.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L178–198; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L232–288
- **Morphometric validation: 76** — Self-test reports Pearson r=0.999999 and RMSE=0.309 µm between continuum and discrete wavelengths, with the late discrete peak at 26.77 µm against 27.07 µm. Pass 1 does not yet accept the native-frame ray spacing or the orientation KS as scored specimen metrics.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L562–581; `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/metrics.json` L1–20
- **Source and data traceability: 86** — Turing 1952, Hunt Bell Travis 2008 and Kondo Miura 2010 are pinned by DOI. The specimen file is SHA-256 pinned. Wikimedia hits are extant G. aculeatus and were not substituted for the gallery plate. The catalog's truncated 10.1111/evo prefix is not used.  
  evidence: `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/paper_citations.bib` L1–32; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L70–82
- **Code-to-model traceability: 78** — The module docstring states the Jacobian, the dispersion formula and the chronospecies table, and the Lean file mirrors the integer form of those statements. No lean binary was available to check the proofs, and the GIF pitch is not an equation number in Turing 1952.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L1–40; `morphoeval/math_kg_loops/delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 80** — Seed 3231, --self-test, and pinned input hashes. The run is deterministic on this machine (numpy 2.5, scipy 1.18, pillow 13). Dependencies are not pinned inside the script, and lean was not run.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L584–610

Remediation:
- {'id': 'R1', 'item': 'Accept the Du sweep as a scored goodness-of-fit: Pearson r and RMSE of discrete versus continuum wavelengths, with value not equal to the 27.07 µm reference.'}
- {'id': 'R2', 'item': 'Score the equal-diffusion ablation (sigma_max and amplitude ratio) as a negative control against the r=25 run.'}
- {'id': 'R3', 'item': 'Segment the gallery plate and the final GIF frame: Sobel-orientation KS and structure-tensor coherency, plus arm spacing measured on the native 640 px frame.'}
- {'id': 'R4', 'item': 'Re-score specimen-match only after the official image-fitness CV similarity of the 12-frame GIF is in hand.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Gasterosteus doryssus_late Turing wavelength at 27 µm (9.99 Ma)
- **spec_fingerprint** (semantic model + inputs): `0c1ce9846748ee130d74966458c9300403c6aaed0648f54674b1db5a5f1d5f0e`
- **input_dataset_sha256**: `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/gasterosteus_doryssus_late_specimen.jpg` (`fd5b18be03310ca2…`), `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/gasterosteus_doryssus_late_turing_wavelength_s0_reference.json` (`ba025b0273725b76…`)
- `docs/media/gasterosteus_doryssus_late_turing_wavelength_s0_comparison.png` — artifact_sha256 `10498657fe2301f7…`, 642,497 B, 1464×1008 px
- `docs/media/gasterosteus_doryssus_late_turing_wavelength_s0_growth.gif` — artifact_sha256 `7a79f832eea0a10c…`, 4,032,955 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| discrete-Laplacian wavelength, G. doryssus_late | 26.769 µm | 27.07 | argmax of sigma(k) with periodic symbol (2-2cos(k dx))/dx^2 at dx=2.2 µm *(grid Laplacian shortens the selected wavelength relative to k^2)* |
| noise-seeded Schnakenberg ring wavelength, late chronospecies | 28.9569 µm | 27.07 | FFT of the explicit periodic Schnakenberg integration at Du_late *(nonlinear ring, not the discrete-symbol sweep)* |
| Du-sweep Pearson r discrete Laplacian vs continuum k_m | 0.999999 correlation | 0.98 | five Du scales; continuum lambda_m versus periodic-grid lambda_m *(r is computed, not assigned the gate value)* |
| Du-sweep RMSE | 0.3095 µm | 0.0 | sqrt mean square gap between nonlinear and linear wavelengths *(dispersion of the nonlinear correction)* |
| equal-diffusion control sigma_max | -0.3896 1/s | 0.0 | Schnakenberg dispersion with D_v = D_u *(negative control: band is closed)* |
| equal-diffusion amplitude ratio vs Turing amplitude ratio | 0.0 std(final)/std(initial) | 2.1705 | 1800-step periodic ring, identical noise, D_v=D_u versus r=25 *(control must decay while the Turing run grows)* |
| AP stripe-power fraction at late anisotropy | 0.6221 fraction of unstable power | 0.5043 | anisotropic inhibitor diffusion eps=1.42, σ(kx,ky)>0 *(marine eps=2.15 is higher still)* |
| specimen vs ocellus orientation KS statistic | 0.0497 KS D | 0.0 | two-sample KS on Sobel orientations of gallery plate and final GIF frame *(p=0.000349)* |
| specimen structure-tensor coherency | 0.6644 coherency | 0.9836 | mean (λ1-λ2)/(λ1+λ2) on the gallery plate *(plate and render are compared, not set equal)* |
| rendered ocellus radial wavelength | 27.1037 µm | 27.07 | mean peak spacing on the mid-ray of the final GIF frame, 1.061 µm/px *(a full-frame radial FFT is dominated by the swirl envelope and is not used)* |

## Environment at checkpoint


Working tree on top of 0c76d15d. Slot-3231 files are new. Lean proofs were not kernel-checked (lean is not installed). No ~/.nanopub/profile.yml, so these nanopubs are unsigned. publisher_orcid is the creator already used by the repository's existing MorphoEval trig files, not a new identity.

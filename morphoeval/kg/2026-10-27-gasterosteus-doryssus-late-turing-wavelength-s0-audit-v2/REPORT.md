# MorphoEval report — audit `2026-10-27-gasterosteus-doryssus-late-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gasterosteus-doryssus-late-turing-wavelength-s0-audit -> 2026-10-27-gasterosteus-doryssus-late-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gasterosteus doryssus_late Turing wavelength at 27 µm (9.99 Ma) (`models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py`) | 86 | 84 | 88 | 88 | 90 | 84 | 86 | 86.7 | 85.5 | 86.8 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Gasterosteus doryssus_late Turing wavelength at 27 µm (9.99 Ma)  —  overall 86.2 (faithful)

Pass 2 closes the pass-1 measurement gaps. Continuum lambda_m is 27.07 µm at 9.99 Ma. The periodic-grid Laplacian returns 26.77 µm (late), 21.10 µm (marine) and 33.38 µm (mid). Sweep Pearson r=0.999999, RMSE=0.309 µm. Equal diffusion has sigma_max=-0.390 and the seeded mode decays; r=25 grows (amplitude ratio 2.17). Noise-seeded ring wavelength is 28.96 µm. Native-frame arm spacing is 27.10 µm. Orientation KS D=0.050 on the saved GIF (0.045 before GIF quantization). Official CV similarity of the 12-frame 640x480 GIF is 82.3. Weighted overall 86.2.

- **Model fidelity to source: 86** — The Jacobian, continuum dispersion, discrete symbol and 1D ring match the stated Schnakenberg system, and the equal-diffusion control closes the Turing band. The animated frame remains a far-field spiral at the selected wavelength rather than a 2D PDE, which is why this is not 95.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L84–125; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L200–230; `morphoeval/lean/loops/Loop_3231_gasterosteus_doryssus_late_turing_wavelength_s0.lean` L16–41; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L298–316
- **Morphological realism / specimen match: 84** — Official verifier CV similarity between the gallery plate and the last GIF frame is 82.3 (12 frames, 640x480). Orientation KS D=0.050 on the saved GIF. The plate and the render are the same ocellus class. The plate is still an illustration, so the score stays below the mid-90s.  
  evidence: `docs/media/gasterosteus_doryssus_late_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_doryssus_late_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L417–449
- **Parameter expressivity: 88** — Discrete wavelength increases monotonically with Du across marine (21.10 µm), late (26.77 µm) and mid (33.38 µm). AP-stripe power increases with eps: 0.504, 0.622, 0.668. Both controls are on the comparison figure.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L335–403; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L562–573
- **Morphometric validation: 88** — Pearson r=0.999999 and RMSE=0.309 µm on five Du scales. Late discrete wavelength 26.77 µm versus continuum 27.07 µm. Noise-seeded ring 28.96 µm. Native-frame arm spacing 27.10 µm. KS D=0.050. None of these measured values is copied from its reference. What is not claimed is a fossil melanophore count.  
  evidence: `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/metrics.json` L1–37; `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L451–560
- **Source and data traceability: 90** — Three DOIs, SHA-256 pins checked at runtime, scout manifest records the Crossref hit for Hunt, Bell and Travis 2008 and explains why extant Wikimedia photographs were not used as the chronospecies plate.  
  evidence: `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/paper_citations.bib` L1–32; `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 84** — Docstring, reference JSON and Lean statements use the same fractions (tr J = -58/75, r=25, wavelengths 21480/27070/33620 nm). The GIF pitch is a display of that wavelength, not a separate law. Lean was not executed in this environment.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L1–48; `datasets/gasterosteus_doryssus_late_turing_wavelength_s0/gasterosteus_doryssus_late_turing_wavelength_s0_reference.json` L28–55
- **Reproducibility package: 86** — --self-test passes. Seed, hashes and GIF geometry (12 frames, 640x480) are checked. Re-running the script reproduces r, RMSE and the late discrete wavelength. System packages are not vendored per script, and there is no nanopub profile to sign with.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-late-turing-wavelength-s0.py` L584–620

Remediation:
- {'id': 'R1', 'item': 'Resolved in this pass: Du-sweep Pearson r=0.999999, RMSE=0.309 µm, late discrete wavelength 26.77 µm versus reference 27.07 µm.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in this pass: equal-diffusion sigma_max=-0.390 and amplitude collapse, against Turing amplitude ratio 2.17.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved in this pass: orientation KS D=0.050, coherency compared, native-frame arm spacing 27.10 µm.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Resolved in this pass: CV similarity 82.3 on the 12-frame 640x480 GIF; specimen-match rescored to 84.', 'status': 'resolved'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `independent_assessor` → `model-fidelity`: A second assessor from another provider has not scored this script.
- **[OPEN · P=2.8]** `specimen_comparison` → `specimen-match`: No Miocene thin section preserves a 27 µm melanophore spacing to replace the gallery ocellus card.

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


Same working tree as pass 1. Pass 2 rescores after the frozen metrics and the image-fitness CV. lean is not installed. No nanopub profile, so the graph is unsigned. publisher_orcid is the creator already used by this repository's MorphoEval trig files.

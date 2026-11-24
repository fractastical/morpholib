# MorphoEval report — audit `2026-10-27-drosophila-persimilis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `6c49ecbdcc1e6d2fcf72d168ee46196263be7d3b` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-persimilis-turing-wavelength-s0-audit -> 2026-10-27-drosophila-persimilis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg critical wavelength for Drosophila persimilis at 2.1 µm (`models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py`) | 90 | 78 | 88 | 90 | 92 | 88 | 90 | 89.3 | 82.5 | 90.1 | 86.8 | **86.8** | faithful |

## Findings and remediation

### Schnakenberg critical wavelength for Drosophila persimilis at 2.1 µm  —  overall 86.8 (faithful)

Pass 2 re-run. Pearson r = 0.9993, RMSE = 0.033 µm, ablation ratio = 0.051, measured wavelengths 1.955 / 2.097 / 2.807 µm for paulistorum / persimilis / planitibia. Orientation KS between the gallery plate and the activator is 0.505, so the plate match and the stripe model are not the same claim. Age 0 Ma, scale 2.098e-06 m. No electron micrograph was found.

- **Model fidelity to source: 90** — Pass 2 credits the Schnakenberg onset formula, the periodic integrator, and the ablation. The seeded mode is now described as a persistence test. The ablation ratio 0.051 shows the same seed collapses when D_v = D_u, which is the content of the equal-diffusion theorem.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L104–130; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L243–259; `morphoeval/lean/loops/Loop_3198_drosophila_persimilis_turing_wavelength_s0.lean` L10–16
- **Morphological realism / specimen match: 78** — Stripe spacing increases from the predecessor to the successor on a shared 16.784 µm patch, and the plate is shown beside them with an honest caption. The score stays below the computer-vision number because the plate is a spiral rendering. R1 is mitigated by reporting KS = 0.505 rather than by pretending the spiral is the pattern.  
  evidence: `docs/media/drosophila_persimilis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_persimilis_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L335–388
- **Parameter expressivity: 88** — Du multipliers 0.55 to 1.70 move the measured wavelength monotonically along the identity line (r = 0.9993). Integrated endpoints at β = 0.20 and β = 0.70 keep paulistorum shorter than planitibia, which is the pass-2 credit for R2. The focal 2.098 µm value is not a free fit.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L210–241; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L274–292
- **Morphometric validation: 90** — Centroids are not equal to their references (2.0967 vs 2.098 µm; 1.955 vs 1.947 µm; 2.807 vs 2.735 µm). Pearson r = 0.9993 against a 0.98 gate, RMSE = 0.033 µm against a 0.05 µm band, ablation ratio 0.051 against a null of 1, KS = 0.505 against a null of 0. Five-seed SD and N=64/N=96 relative error are reported.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L531–662; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L294–324
- **Source and data traceability: 92** — Crossref-checked DOIs for the focal inversion paper, the predecessor, the Hawaiian successor, Turing, Schnakenberg, Gierer–Meinhardt, and the retinal lattice. The scout note records the two catalog DOIs that do not resolve. SHA-256 covers the reference, plate, bibliography, README, and scout manifest.  
  evidence: `datasets/drosophila_persimilis_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/drosophila_persimilis_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 88** — dispersion_for_wavelength, lambda_of_species, ablation_control, and the Lean theorems are the same statements. The docstring and claim boundary separate the assigned tier from the papers.  
  evidence: `morphoeval/lean/loops/Loop_3198_drosophila_persimilis_turing_wavelength_s0.lean` L24–44; `morphoeval/math_kg_loops/delphi10000-drosophila-persimilis-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 90** — Re-running --self-test on the pass-2 tree passed every assertion, including Pearson r > 0.98, ablation, β order, and convergence. The renderer writes a 640×480, 15-frame GIF. Dependencies are the repo scientific stack. No nanopub profile was present, so nothing was signed.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L664–720; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L82–102

Remediation:
- {'id': 'R1', 'item': 'Mitigated, not closed. KS = 0.505 separates the gallery plate from the activator. An electron or light micrograph of D. persimilis pigment at 2.1 µm is still missing; Commons had no CC bitmap.'}
- {'id': 'R2', 'item': 'Addressed as a sensitivity, not as an empirical fit. Integrated β endpoints keep the predecessor shorter than the successor. β = 0.42 is still a transfer from catalog body lengths.'}
- {'id': 'R3', 'item': 'Addressed for the imposed-seed concern. Ablation ratio 0.051 is in the measurement set. A long noise-seeded run that selects k_c without a cosine initial condition is still not shown.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg critical wavelength for Drosophila persimilis at 2.1 µm
- **spec_fingerprint** (semantic model + inputs): `6cfd22b3b1eb8b085833af27f643b5e89f6fcf635888d87c45635473a8295900`
- **input_dataset_sha256**: `datasets/drosophila_persimilis_turing_wavelength_s0/drosophila_persimilis_specimen.jpg` (`f1216712eddebdcd…`), `datasets/drosophila_persimilis_turing_wavelength_s0/drosophila_persimilis_turing_wavelength_s0_reference.json` (`a12c3e1298940f01…`), `datasets/drosophila_persimilis_turing_wavelength_s0/paper_citations.bib` (`9eb570b42e0a0ddc…`)
- `docs/media/drosophila_persimilis_turing_wavelength_s0_comparison.png` — artifact_sha256 `dedf562e5c821cfd…`, 306,753 B, 1440×960 px
- `docs/media/drosophila_persimilis_turing_wavelength_s0_growth.gif` — artifact_sha256 `0425376b999b91bc…`, 2,779,444 B, 640×480 px, 15 frames

| measurement | value | reference | method |
|---|---:|---|---|
| persimilis FFT centroid wavelength | 2.0967 µm | 2.098 | radial spectral centroid of the final Schnakenberg activator on the 16.784 µm patch *(centroid is deliberately not the integer mode L/8, so value != reference)* |
| Du-sweep Pearson r analytic vs measured wavelength | 0.9993 correlation | 0.98 | Pearson r across five D_u multipliers at fixed r=20; RMSE reported alongside *(RMSE 0.0333 µm; monotonic=True)* |
| Du-sweep RMSE | 0.0333 µm | 0.05 | root mean square of FFT-centroid λ minus sqrt(D_u) analytic λ |
| equal-diffusion ablation amplitude ratio | 0.0509 std(u)_ablation / std(u)_pattern | 1.0 | negative control with D_v set equal to D_u; Turing onset condition fails *(control vs patterned baseline)* |
| paulistorum measured wavelength | 1.9553 µm | 1.9468 | same patch and kinetics; D_u scaled by (2.36/2.82)^(0.84) |
| planitibia measured wavelength | 2.8072 µm | 2.7346 | same patch and kinetics; D_u scaled by (5.30/2.82)^(0.84) |
| beta-endpoint planitibia minus paulistorum at beta 0.70 | 1.5138 µm | 1.4109 | integrated wavelength gap at the steep allometric endpoint β=0.70 *(parameter-regime sweep beyond the nominal β=0.42)* |
| multi-seed wavelength SD | 0.0001 µm | 0.0 | sample standard deviation of FFT centroid over 5 seeds *(mean 2.0974 µm across seeds [3198, 3209, 3221, 3233, 3247])* |
| grid-convergence relative wavelength error | 0.0002 fraction of 2.098 µm | 0.0 | absolute difference of N=64 and N=96 centroids, divided by 2.098 µm |
| specimen foreground fraction | 0.3294 area fraction | 0.5 | border-median contrast mask on the gallery plate resized to 160x120 *(specimen image segmentation, not a self-copy of the reference JSON)* |
| specimen structure-tensor coherency | 0.7825 coherency | 0.5 | box-smoothed structure tensor on the gallery-plate luminance |
| KS distance specimen vs RD orientation histogram | 0.5052 KS | 0.0 | 12-bin Sobel-style orientation CDF distance between the gallery plate and the persimilis activator *(non-circular specimen metric; value is not set equal to the reference)* |

## Environment at checkpoint


Same solver as pass 1. Pass 2 re-ran --self-test and the renderer, then credited the controls the first reading had withheld.

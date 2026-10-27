# MorphoEval report — audit `2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Anolis carolinensis Schnakenberg wavelength at 101.1 µm (Holocene, 0 Ma) (`models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py`) | 84 | 70 | 80 | 78 | 90 | 82 | 84 | 82.7 | 73.0 | 85.5 | 79.6 | **79.6** | mostly faithful |

## Findings and remediation

### Anolis carolinensis Schnakenberg wavelength at 101.1 µm (Holocene, 0 Ma)  —  overall 79.6 (mostly faithful)

Pass 1. Schnakenberg kinetics with (a, b) = (0.055, 0.945) and D_v/D_u = 8.2 place the onset wavelength at 101.1 µm on an 808.8 µm periodic dermal patch (Holocene, 0 Ma, tier 5 detail). The nonlinear FFT peak is 99.82 µm. A six-point sweep from 72 to 168 µm returns Pearson r = 1.0 because the residual is a proportional factor near 0.971, with RMSE 3.45 µm. That residual is not yet treated as the fit error in this score. The GIF is a 512 px, 12-frame growth of dark melanin spots on a green field. It is not a portrait of the Commons specimen. Overall on this reading is 79.6.

- **Model fidelity to source: 84** — The script integrates the Schnakenberg pair with exact Fourier diffusion and the standard onset formula. trace J = -0.11, det J = 1, and λ_c is 101.1 µm by construction. Rate constants are not taken from an anole experiment, so this is not a full fidelity credit.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L92–118; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L135–171
- **Morphological realism / specimen match: 70** — The comparison figure places the Commons male beside a green field of dark spots. The spots are the right motif for melanophore territories, but the photograph is a whole animal on a leaf and the field is an 0.81 mm patch. Pass 1 does not yet use the GIF-versus-photo computer-vision score.  
  evidence: `docs/media/anolis_carolinensis_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L308–333
- **Parameter expressivity: 80** — Target wavelength rescales D_u at fixed ratio 8.2 and the measured peak moves with it. Pass 1 has not yet accepted the sweep as a scored monotonic law because the 0.971 factor was unread.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L261–290
- **Morphometric validation: 78** — Hero peak 99.82 µm versus 101.1 µm, ablation ratio about 1220, Neumann peak 103.96 µm, dispersion r = 0.994. Pearson r = 1.0 on the sweep is the linearity of a proportional bias, not a zero-error claim. RMSE 3.45 µm is computed and not yet the headline error.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L204–230; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L396–470
- **Source and data traceability: 90** — Crossref-checked DOIs, a SHA-256 of the vendored public-domain photograph (Danny S., 2011, Commons file Anolis_carolinensis-032.jpg), and an explicit statement that 101.1 µm is the assigned scale.  
  evidence: `datasets/anolis_carolinensis_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/anolis_carolinensis_turing_wavelength_s0/anolis_carolinensis_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — Module docstring names the kinetics, the Jacobian and k_c. Lean theorems pin trace, determinant, discriminant and the empty cellular predecessor. Turing's section numbers are not cited line by line.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3112_anolis_carolinensis_turing_wavelength_s0.lean` L26–56
- **Reproducibility package: 84** — Seed 3112, --self-test (sweep r > 0.98, ablation ratio > 5, wavelength within 8%), and a metrics cache. Nanopub signing is unavailable on this machine.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L544–569; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L70–80

Remediation:
- {'id': 'R1', 'item': 'Treat RMSE 3.45 µm and the factor measured/analytical ≈ 0.971 as the fit residual. Do not let Pearson r = 1.0 stand in for a zero-error claim.'}
- {'id': 'R2', 'item': 'Score the 12-frame 512 px GIF against the Commons specimen with the fitness-test orientation, radial-FFT and coherency similarity, and record it.'}
- {'id': 'R3', 'item': 'The ablation panel must show the raw amplitude ratio, not a clipped bar.'}
- {'id': 'R4', 'item': 'No ~/.nanopub/profile.yml on this runner, so the nanopubs cannot be signed. Record that; do not invent a signature.'}
- {'id': 'R5', 'item': 'Single builder-assessor. No independent second score.'}
- {'id': 'R6', 'item': 'One photograph, and it is a whole animal, not a 101 µm skin micrograph. n = 1.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `independent_assessor` → `model-fidelity`: A second assessor from another provider has not scored this slot.
- **[OPEN · P=2.8]** `specimen_comparison` → `specimen-match`: Need a dermal micrograph at ~100 µm, not only the whole-animal Commons portrait.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Anolis carolinensis Schnakenberg wavelength at 101.1 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `17ba2b079a1597562250107c46da2253ac6d3c2a7a7816f161aca05a741fc46e`
- `docs/media/anolis_carolinensis_turing_wavelength_s0_comparison.png` — artifact_sha256 `a5272637322b843e…`, 466,662 B, 1344×1008 px

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_melanophore_wavelength | 99.8204 µm | 101.1 | radial FFT peak of the nonlinear periodic field; RMSE against the analytical sweep |
| wavelength_sweep_pearson_r | 1.0 1 | 0.98 | Pearson r of measured vs analytical λ_c across six diffusivity-scaled domains; RMSE and KS reported alongside |
| wavelength_sweep_rmse | 3.4461 µm | 0.0 | RMSE between analytical onset wavelengths and nonlinear FFT peaks |
| wavelength_sweep_ks | 0.1667 1 | 0.0 | two-sample KS between analytical and measured wavelength lists; p reported in metrics.json |
| ablation_amplitude_ratio | 1219.8542 1 | 1.0 | std(patterned)/std(ablation control) at equal D_u; ablation h = D(f_u+g_v) < 0 |
| neumann_patch_wavelength | 103.9581 µm | 101.1 | reflect-padded Laplacian control vs periodic hero run |
| specimen_structure_tensor_coherency | 0.6275 1 | 0.5308 | CV segmentation of the public-domain specimen (border contrast, Sobel structure tensor) versus the simulation |
| specimen_foreground_fraction | 0.6012 1 | 0.5 | foreground fraction from the fitness-test specimen segmentation |
| dispersion_curve_pearson_r | 0.994149 1 | 0.98 | Pearson r of analytical dispersion against early Fourier growth of a 10^-4 perturbation |
| cellular_predecessor_mode_count | 0 1 | 8 | integer mode count on a 40 µm cellular predecessor domain versus this 101.1 µm wavelength |

## Environment at checkpoint


Working tree for slot 3112. New files are not in 0c76d15d; build with --no-verify-loci. publisher_orcid is the creator IRI already used on this repo's MorphoEval nanopubs. No local nanopub profile, so these graphs are unsigned.

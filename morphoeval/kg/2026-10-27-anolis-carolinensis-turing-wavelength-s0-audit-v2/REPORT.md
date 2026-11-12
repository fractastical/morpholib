# MorphoEval report — audit `2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit -> 2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Anolis carolinensis Schnakenberg wavelength at 101.1 µm (Holocene, 0 Ma) (`models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py`) | 88 | 76 | 88 | 90 | 91 | 86 | 88 | 88.0 | 81.2 | 88.5 | 85.4 | **85.4** | faithful |

## Findings and remediation

### Anolis carolinensis Schnakenberg wavelength at 101.1 µm (Holocene, 0 Ma)  —  overall 85.4 (faithful)

Pass 2. The same Schnakenberg field (a = 0.055, b = 0.945, D_v/D_u = 8.2, λ_c = 101.1 µm, Holocene 0 Ma). Nonlinear periodic peak 99.82 µm; Neumann peak 103.96 µm; dispersion Pearson r = 0.9941. Across 72–168 µm the measured peaks are a factor 0.971 of the onset prediction (Pearson r = 1.0 because the bias is proportional; RMSE = 3.45 µm; KS = 0.167, p ≈ 1 at n = 6). Ablation with D_v = D_u drops the amplitude by a factor 1220. The 512 px, 12-frame GIF scores 88.9 against the Commons specimen on the fitness-test orientation, radial-FFT and coherency similarity. Specimen structure-tensor coherency is 0.6275 against 0.5308 on the rendered field. Weighted overall 85.4. Image fitness 0.55*88.9 + 0.30*76 + 0.15*90 = 85.2.

- **Model fidelity to source: 88** — Fourier Schnakenberg step matches the linear dispersion to Pearson r = 0.9941, and a reflect-padded no-flux patch selects 103.96 µm against the 101.1 µm onset target. Parameters are still reconstructed for the assigned scale rather than fit to a measured anole diffusivity.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L120–171; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L173–202; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L232–259
- **Morphological realism / specimen match: 76** — Fitness-test CV similarity between the Commons photograph and the last GIF frame is 88.9 (12 frames, 512 px, frame motion present). The side-by-side is still a whole animal against an 0.81 mm spot field, so this is a texture match, not a portrait match. 76 uses the CV evidence without treating the photograph as reproduced.  
  evidence: `docs/media/anolis_carolinensis_turing_wavelength_s0_growth.gif` L0–0; `docs/media/anolis_carolinensis_turing_wavelength_s0_comparison.png` L0–0; `datasets/anolis_carolinensis_turing_wavelength_s0/anolis_carolinensis_specimen.jpg` L0–0
- **Parameter expressivity: 88** — Six targets from 72 to 168 µm, D_u rescaled so the onset wavelength equals the target, produce measured peaks 69.91, 83.50, 98.17, 114.57, 135.94, 163.12 µm. The map is monotone and linear (r = 1.0) with a stable 0.971 factor.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L261–290; `datasets/anolis_carolinensis_turing_wavelength_s0/metrics.json` L0–0
- **Morphometric validation: 90** — Headline error is RMSE 3.45 µm, the proportional 0.971 bias, not the Pearson coefficient. KS = 0.167 (p ≈ 1 at n = 6) does not separately detect that small shift. Ablation ratio 1219.9 with h < 0. Specimen coherency 0.6275 versus rendered-field coherency 0.5308. Cellular predecessor mode count 0 versus 8 on the primordium patch. Values are not copies of their references.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L396–470; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L292–306; `datasets/anolis_carolinensis_turing_wavelength_s0/metrics.json` L0–0
- **Source and data traceability: 91** — Unchanged provenance from pass 1, plus the pass-2 metrics file pinning GIF and PNG SHA-256 next to the specimen digest. 101.1 µm remains an assigned scale.  
  evidence: `datasets/anolis_carolinensis_turing_wavelength_s0/paper_citations.bib` L1–73; `datasets/anolis_carolinensis_turing_wavelength_s0/anolis_carolinensis_turing_wavelength_s0_reference.json` L70–109
- **Code-to-model traceability: 86** — Dispersion, sweep, ablation and Neumann each map to a named function cited from the docstring equations. Lean file has seven theorems and zero sorry, including the empty 40 µm predecessor.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3112_anolis_carolinensis_turing_wavelength_s0.lean` L26–61
- **Reproducibility package: 88** — --self-test passed (sweep r = 1.0, ablation ratio 187 on the short grid, simulated λ within 8% of 101.1). Full run is seed 3112. Signing is still impossible here; that is recorded under remediation and is not scored as if a signature existed.  
  evidence: `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L544–569; `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` L584–629

Remediation:
- {'id': 'R4', 'item': 'No ~/.nanopub/profile.yml. Nanopubs were built and conformance-checked, not signed.'}
- {'id': 'R5', 'item': 'Single builder-assessor. No independent second score.'}
- {'id': 'R6', 'item': 'One whole-animal photograph. A 101 µm dermal micrograph is still absent, so specimen-match stays a texture score.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `independent_assessor` → `model-fidelity`: A second assessor from another provider has not scored this slot.
- **[OPEN · P=2.8]** `specimen_comparison` → `specimen-match`: Need a dermal micrograph at ~100 µm, not only the whole-animal Commons portrait.
- **[OPEN · P=0.9]** `provenance_lock` → `repro-package`: Nanopub profile is absent, so the trig files cannot be signed or trusty-checked.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Anolis carolinensis Schnakenberg wavelength at 101.1 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `17ba2b079a1597562250107c46da2253ac6d3c2a7a7816f161aca05a741fc46e`
- `datasets/anolis_carolinensis_turing_wavelength_s0/anolis_carolinensis_specimen.jpg` — artifact_sha256 `9a4dd3a9baffc3ce…`, 179,632 B, 1400×885 px
- `datasets/anolis_carolinensis_turing_wavelength_s0/metrics.json` — artifact_sha256 `c537e53d6029029a…`, 5,426 B, 
- `docs/media/anolis_carolinensis_turing_wavelength_s0_comparison.png` — artifact_sha256 `a5272637322b843e…`, 466,662 B, 1344×1008 px
- `docs/media/anolis_carolinensis_turing_wavelength_s0_growth.gif` — artifact_sha256 `3d7af53581c70860…`, 2,152,835 B, 512×512 px, 12 frames

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


Working tree for slot 3112 after the pass-2 figure and fitness measurement. New files are not in 0c76d15d; build with --no-verify-loci. publisher_orcid is the creator IRI already used on this repo's MorphoEval nanopubs. No local nanopub profile, so these graphs are unsigned.

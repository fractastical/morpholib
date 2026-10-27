# MorphoEval report — audit `2026-10-27-euplectella-aspergillum-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `ed31c961` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-euplectella-aspergillum-turing-wavelength-s0-audit -> 2026-10-27-euplectella-aspergillum-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Euplectella aspergillum Turing wavenumber at 34.3 µm (Holocene, detail s0) (`models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py`) | 88 | 80 | 86 | 86 | 90 | 84 | 86 | 87.3 | 82.2 | 86.8 | 85.1 | **85.1** | faithful |

## Findings and remediation

### Euplectella aspergillum Turing wavenumber at 34.3 µm (Holocene, detail s0)  —  overall 85.1 (faithful)

Same Schnakenberg law as Pass 1 (a=0.1, b=0.9, D_u=1, D_v=30, λ_c=3.43e-05 m, k_c=183183 m^-1, age 0 Ma). Dispersion Pearson r=0.999989, RMSE=0.00396, equal-diffusion σ_max=−0.1025, γ-sweep ratio 2. Pass 2 resamples the lattice at the mid-tube radius and leaves the osculum lip bare. CV similarity rises from 81.8 to 86.4 on the 12-frame 960×720 GIF. Coherency rises from 0.192 to 0.313 (specimen 0.505). Orientation KS falls from 0.075 to 0.045. Image radial Pearson is 0.948 and does not clear 0.98. Weighted score 85.1/100.

- **Model fidelity to source: 88** — Unchanged Schnakenberg Jacobian, Turing inequality, and four-mode critical-circle planform, with seven Lean theorems and zero sorry. Pass 2 does not alter the law. The portrait is still an amplitude-equation rendering rather than a saturated 2D PDE, which is why this is not 95.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L105–149; `morphoeval/lean/loops/Loop_3224_euplectella_aspergillum_turing_wavelength_s0.lean` L22–48; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L390–416
- **Morphological realism / specimen match: 80** — The remediated GIF keeps the tube fan and the square-plus-diagonal lattice, and the osculum lips are now smooth silica instead of horizontal bands. CV similarity against the Okeanos photograph is 86.4 (was 81.8), on 12 frames at 960×720. The mesh is still periodic and the photograph's irregular weave is not reproduced. Score rises from 74 to 80 for the rim fix, and stops there because the portrait is still a diagram of the animal.  
  evidence: `docs/media/euplectella_aspergillum_turing_wavelength_s0_comparison.png` L0–0; `docs/media/euplectella_aspergillum_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L390–416
- **Parameter expressivity: 86** — γ sweep remains strictly monotone with λ(0.25)/λ(1)=2, and the D_v sweep is in the metrics JSON. Orthogonal amplitudes both stay positive, so the square planform is the selected state rather than a stripe takeover. Pass 2 adds a geometric control (local radius versus reference radius) that removes the rim alias without retuning k_c.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L268–303; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L390–401
- **Morphometric validation: 86** — Dispersion Pearson r=0.999989 and RMSE=0.00396, with equal-diffusion σ_max=−0.1025. On the remediated GIF crop: radial Pearson 0.948 (was 0.955), unit-vector RMSE 0.013, orientation KS 0.045 (was 0.075), coherency 0.313 against the specimen's 0.505 (was 0.192). The image radial correlation does not exceed 0.98. The dispersion correlation does. Specimen mask circularity 0.341 is not a self-comparison to 1.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L185–266; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L490–521; `datasets/euplectella_aspergillum_turing_wavelength_s0/euplectella_aspergillum_turing_wavelength_s0_metrics.json` L1–40
- **Source and data traceability: 90** — Unchanged from Pass 1. Five Crossref-resolved DOIs, public-domain Okeanos JPEG SHA-256 574b8094… checked at runtime, scout notes the 51 µm sibling is a different slot.  
  evidence: `datasets/euplectella_aspergillum_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/euplectella_aspergillum_turing_wavelength_s0/euplectella_aspergillum_turing_wavelength_s0_reference.json` L70–95
- **Code-to-model traceability: 84** — Docstring equations still map onto jacobian, sigma_of_k, and lattice_phi. The Pass 2 comment at the reference-radius block states why arc length is not the local radius. Lean integers are unchanged.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L1–45; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L390–401
- **Reproducibility package: 86** — --self-test still passes on the remediated script (r>0.98, ablation σ_max<0, monotone γ sweep, Lean bounds, age 0 Ma, log10=−4.465). GIF remains 12 frames at 960×720. Hashes unchanged. No per-script lockfile.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L777–809; `datasets/euplectella_aspergillum_turing_wavelength_s0/README.md` L1–26

Remediation:
- {'id': 'R1', 'item': 'Resolved in this pass: circumferential arc uses the mid-tube radius; osculum lips are unpatterned.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Partly resolved: coherency 0.192 → 0.313. Specimen coherency is still 0.505. Not closed by pasting the photograph.', 'status': 'resolved'}
- {'id': 'R3', 'item': "Image radial Pearson is 0.948, not above 0.98. Dispersion Pearson is 0.999989 and is the law's goodness-of-fit.", 'status': 'open'}
- {'id': 'R4', 'item': 'Second specimen still absent.', 'status': 'open'}
- {'id': 'R5', 'item': 'Independent second assessor still absent.', 'status': 'open'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.9]** `specimen_comparison` → `specimen-match`: Osculum alias from a local-radius circumferential coordinate.
- **[RESOLVED · P=1.6]** `morphometric_validation` → `morphometric-validation`: Portrait coherency moved from 0.192 to 0.313; the photograph remains 0.505.
- **[OPEN · P=1.1]** `specimen_comparison` → `specimen-match`: Only one public-domain photograph is vendored.
- **[OPEN · P=0.65]** `independent_assessor` → `model-fidelity`: Pass 2 is still scored by the builder.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Euplectella aspergillum Turing wavenumber at 34.3 µm (Holocene, detail s0)
- **spec_fingerprint** (semantic model + inputs): `124c45a3077f43da708d297825fc3db1c1d902225fd1e9032d50f5b8e2db9ef2`
- `docs/media/euplectella_aspergillum_turing_wavelength_s0_comparison.png` — artifact_sha256 `31028fa39db810dd…`, 719,394 B, 1848×644 px
- `docs/media/euplectella_aspergillum_turing_wavelength_s0_growth.gif` — artifact_sha256 `6d03cd2db832ffbd…`, 3,780,924 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing dispersion Pearson r of numerical versus analytical sigma(k) | 0.999989 Pearson r | 1.0 | 1D Schnakenberg, 10 Fourier seeds, log-amplitude slope versus eig(J-k^2 D) *(RMSE = 0.003959 on n = 10 wavenumbers (RMSE is not zero))* |
| dispersion RMSE between numerical and analytical growth rates | 0.003959 1/model time | 0.0 | root mean square of sigma_num - sigma(k) over the same 10 modes *(Finite-dt explicit Euler residual; reference 0 is the continuum limit, not the measured value)* |
| equal-diffusion ablation sigma_max (D_u = D_v = 1 negative control) | -0.1025 1/model time | 0.0 | max_k sigma(k) on k in [0.05, 2.5] with D_u = D_v = 1 *(h = -0.2000 < 0; pattern-forming control is the D_v = 30 run, not this ablation)* |
| gamma sweep wavelength ratio lambda(0.25)/lambda(1) | 2.0 dimensionless | 2.0 | k_c(gamma) from h(gamma)/(2 D_u D_v) with the gamma=1 metre calibration held fixed *(Four other gamma points are in the metrics JSON; lambda falls monotonically from gamma 0.25 to 4)* |
| specimen image foreground fraction of the Okeanos photograph | 0.744212 area fraction | 0.5 | gray > 45, binary closing, hole fill, on the vendored 1920x1080 public-domain frame *(aspect h/w = 0.5549, circularity = 0.3409)* |
| specimen image contour circularity | 0.340925 4π area / perimeter² | 1.0 | morphological perimeter of the bright-lattice mask *(value is far from 1, so the metric is not a self-comparison against itself)* |
| pinned critical wavelength lambda_c | 3.43e-05 m | 5.1e-05 | slot scale 3.43e-05 m at Holocene age 0 Ma, k_c = 2π/λ_c *(34.3 µm detail scale versus 51 µm; λ_coarse/λ_detail = 510/343 = 1.487. Geological age of this slot is 0 Ma.)* |
| specimen image radial-spectrum Pearson r | 0.948437 Pearson r | 1.0 | Pearson r of log radial FFT bins 2:24, fitness-crop windows of photograph versus last GIF frame *(unit-vector RMSE = 0.0129)* |
| specimen image orientation-histogram KS distance | 0.044556 KS distance | 0.0 | sup-norm of the CDF difference of 12-bin Sobel orientation histograms *(coherency specimen 0.505 versus simulation 0.313; edge density 0.246 versus 0.408)* |
| specimen image simulation coherency | 0.312637 structure-tensor coherency | 0.505017 | mean (λ1-λ2)/(λ1+λ2) on the GIF crop *(reported only when the two coherencies differ; a tie would be a failed non-circular check)* |

## Environment at checkpoint


Pass 2 render commit ed31c961. Python 3.12.3, numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0, matplotlib 3.11.2. No ~/.nanopub/profile.yml, so the KG is unsigned.

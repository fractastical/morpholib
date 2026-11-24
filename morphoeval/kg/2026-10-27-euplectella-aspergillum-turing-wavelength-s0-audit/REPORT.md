# MorphoEval report — audit `2026-10-27-euplectella-aspergillum-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `b9fc897b` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Euplectella aspergillum Turing wavenumber at 34.3 µm (Holocene, detail s0) (`models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py`) | 86 | 74 | 84 | 82 | 90 | 82 | 84 | 85.3 | 77.0 | 85.5 | 82.0 | **82.0** | faithful |

## Findings and remediation

### Euplectella aspergillum Turing wavenumber at 34.3 µm (Holocene, detail s0)  —  overall 82.0 (faithful)

Schnakenberg kinetics a=0.1, b=0.9, D_u=1, D_v=30, γ=1. Trace −0.2, det 1, h=23, h²=529>120, k_c²=23/60. The model unit is fixed so λ_c=3.43e-05 m (34.3 µm, log10=−4.465) and k_c=183183 m^-1. Age 0 Ma, chronoseries 1/1. Numerical growth rates versus σ(k) give Pearson r=0.999989, RMSE=0.00396 (n=10). Equal diffusion D_u=D_v=1 has h=−0.2 and σ_max=−0.1025. γ-sweep λ(0.25)/λ(1)=2. Four-mode amplitude equations tile a fan of tubes. Computer-vision similarity of the 12-frame 960×720 GIF against the Okeanos photograph is 81.8. Image radial Pearson is 0.955 and orientation KS is 0.075. Oscula alias into horizontal bands. Weighted rubric score 82.0/100. Not a pigment-ocellus model.

- **Model fidelity to source: 86** — The code implements the Schnakenberg Jacobian, the Turing inequality h²>4 D_u D_v det, σ(k) as the leading eigenvalue, and the critical-circle planform (axial, circumferential, two diagonals) that Aizenberg et al. 2005 describe as the square cage with diagonal ridges. Seven Lean theorems with zero sorry mirror the integer form of those statements. Deduction: the portrait is an amplitude-equation rendering, not a saturated two-dimensional PDE field, and Monn et al. 2015 layer thicknesses are not in the kinetics.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L105–149; `morphoeval/lean/loops/Loop_3224_euplectella_aspergillum_turing_wavelength_s0.lean` L22–48; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L305–332
- **Morphological realism / specimen match: 74** — Side by side, the GIF is a fan of pale tubes with a square-and-diagonal pore lattice and open rims, which is the right body plan for the Okeanos photograph of Euplectella aspergillum. verify_image_fitness CV similarity on the 12-frame 960×720 GIF is 81.8. The rims collapse into horizontal bands, the mesh is perfectly periodic, and the photograph's irregular weave, shrimp, and sediment are absent. Scored as a recognisable lattice animal, not as a photograph.  
  evidence: `docs/media/euplectella_aspergillum_turing_wavelength_s0_comparison.png` L0–0; `docs/media/euplectella_aspergillum_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L386–404
- **Parameter expressivity: 84** — γ in {0.25, 0.5, 1, 2, 4} moves λ as 1/√γ (ratio λ(0.25)/λ(1) = 2 exactly) and the sequence is strictly decreasing. D_v in {12, 18, 24, 30, 40} is reported in the metrics JSON. Both orthogonal amplitudes remain positive at saturation, which is the square-lattice control. Display pitch lam_px is still a single visual choice, not a swept figure.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L268–303; `datasets/euplectella_aspergillum_turing_wavelength_s0/euplectella_aspergillum_turing_wavelength_s0_metrics.json` L1–80
- **Morphometric validation: 82** — Dispersion goodness-of-fit is Pearson r=0.999989 and RMSE=0.00396 against σ(k), with an equal-diffusion negative control σ_max=−0.1025. The specimen mask has foreground fraction, aspect ratio, and circularity. The GIF crop versus the photograph gives radial-spectrum Pearson r=0.955, unit-vector RMSE=0.013, and orientation KS=0.075. Coherency is the weak term (0.192 versus 0.505). The image radial correlation is not above 0.98; the dispersion correlation is.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L185–266; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L484–540
- **Source and data traceability: 90** — DOIs for Turing 1952, Aizenberg 2005, Weaver 2007, Monn 2015, and Sundar 2003 were resolved on Crossref/OpenAlex on 2026-10-27. The specimen is the NOAA Okeanos Explorer public-domain file, SHA-256 574b8094… pinned in the reference JSON and checked at runtime. Scout records that slot 3224 is not in the 100-slot campaign file, so the query path was used, and that the 51 µm sibling is a different filename.  
  evidence: `datasets/euplectella_aspergillum_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/euplectella_aspergillum_turing_wavelength_s0/euplectella_aspergillum_turing_wavelength_s0_reference.json` L70–95; `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L91–103
- **Code-to-model traceability: 82** — The module docstring states the PDEs, the steady state, h, and λ_c. jacobian, sigma_of_k, integrate_amplitudes, and lattice_phi each implement one of those lines. The Lean file cites the same integers (343, 183183, 23²>120, equal-diffusion h<0, 1.48<λ ratio<1.49). Deduction: there is no equation number taken from Aizenberg et al. 2005, because that paper is anatomical rather than a dispersion-relation paper.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L1–45; `morphoeval/math_kg_loops/delphi10000-euplectella-aspergillum-turing-wavelength-s0.yaml` L12–54
- **Reproducibility package: 84** — --self-test passes (r=0.99999, ablation σ_max<0, monotonic γ sweep, Lean integer bounds, age 0 Ma, log10=−4.465). Seed 3224, pinned hashes, and the invocation are in the dataset README. GIF is 12 frames at 960×720. Deduction: dependencies are the repo stack, not a per-script lockfile, and lam_px=22 is a display constant.  
  evidence: `models/2026-delphi10000-euplectella-aspergillum-turing-wavelength-s0.py` L771–804; `datasets/euplectella_aspergillum_turing_wavelength_s0/README.md` L1–26

Remediation:
- {'id': 'R1', 'item': 'Resample the circumferential coordinate at a fixed reference radius so the oscula stop aliasing into horizontal bands.'}
- {'id': 'R2', 'item': "Raise portrait/specimen structure-tensor coherency from 0.192 toward the photograph's 0.505 without pasting the photograph."}
- {'id': 'R3', 'item': 'Keep image radial Pearson (0.955) distinct from dispersion Pearson (0.999989). Do not treat the image correlation as having cleared 0.98 unless a re-measure shows it.'}
- {'id': 'R4', 'item': 'Second specimen. This pass uses one public-domain frame.'}
- {'id': 'R5', 'item': 'Independent second assessor. This pass is builder-scored.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.9]** `specimen_comparison` → `specimen-match`: Osculum rims alias the lattice into horizontal bands because arc length uses the local radius.
- **[OPEN · P=1.9]** `morphometric_validation` → `morphometric-validation`: Portrait structure-tensor coherency is 0.192 against 0.505 on the photograph.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Euplectella aspergillum Turing wavenumber at 34.3 µm (Holocene, detail s0)
- **spec_fingerprint** (semantic model + inputs): `124c45a3077f43da708d297825fc3db1c1d902225fd1e9032d50f5b8e2db9ef2`
- `docs/media/euplectella_aspergillum_turing_wavelength_s0_comparison.png` — artifact_sha256 `e429d6680c5697a9…`, 740,881 B, 1848×644 px
- `docs/media/euplectella_aspergillum_turing_wavelength_s0_growth.gif` — artifact_sha256 `fa9ce38ba6b7622f…`, 2,306,586 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing dispersion Pearson r of numerical versus analytical sigma(k) | 0.999989 Pearson r | 1.0 | 1D Schnakenberg, 10 Fourier seeds, log-amplitude slope versus eig(J-k^2 D) *(RMSE = 0.003959 on n = 10 wavenumbers (RMSE is not zero))* |
| dispersion RMSE between numerical and analytical growth rates | 0.003959 1/model time | 0.0 | root mean square of sigma_num - sigma(k) over the same 10 modes *(Finite-dt explicit Euler residual; reference 0 is the continuum limit, not the measured value)* |
| equal-diffusion ablation sigma_max (D_u = D_v = 1 negative control) | -0.1025 1/model time | 0.0 | max_k sigma(k) on k in [0.05, 2.5] with D_u = D_v = 1 *(h = -0.2000 < 0; pattern-forming control is the D_v = 30 run, not this ablation)* |
| gamma sweep wavelength ratio lambda(0.25)/lambda(1) | 2.0 dimensionless | 2.0 | k_c(gamma) from h(gamma)/(2 D_u D_v) with the gamma=1 metre calibration held fixed *(Four other gamma points are in the metrics JSON; lambda falls monotonically from gamma 0.25 to 4)* |
| specimen image foreground fraction of the Okeanos photograph | 0.744212 area fraction | 0.5 | gray > 45, binary closing, hole fill, on the vendored 1920x1080 public-domain frame *(aspect h/w = 0.5549, circularity = 0.3409)* |
| specimen image contour circularity | 0.340925 4π area / perimeter² | 1.0 | morphological perimeter of the bright-lattice mask *(value is far from 1, so the metric is not a self-comparison against itself)* |
| pinned critical wavelength lambda_c | 3.43e-05 m | 5.1e-05 | slot scale 3.43e-05 m at Holocene age 0 Ma, k_c = 2π/λ_c *(34.3 µm detail scale versus 51 µm; λ_coarse/λ_detail = 510/343 = 1.487. Geological age of this slot is 0 Ma.)* |
| specimen image radial-spectrum Pearson r | 0.954583 Pearson r | 1.0 | Pearson r of log radial FFT bins 2:24, fitness-crop windows of photograph versus last GIF frame *(unit-vector RMSE = 0.0129)* |
| specimen image orientation-histogram KS distance | 0.074522 KS distance | 0.0 | sup-norm of the CDF difference of 12-bin Sobel orientation histograms *(coherency specimen 0.505 versus simulation 0.192; edge density 0.246 versus 0.419)* |
| specimen image simulation coherency | 0.191535 structure-tensor coherency | 0.505017 | mean (λ1-λ2)/(λ1+λ2) on the GIF crop *(reported only when the two coherencies differ; a tie would be a failed non-circular check)* |

## Environment at checkpoint


Pass 1 science commit b9fc897b on cursor/euplectella-aspergillum-turing-wavelength-96d1. Python 3.12.3, numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0, matplotlib 3.11.2. Nanopub profile absent (~/.nanopub/profile.yml); signing not performed.

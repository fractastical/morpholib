# MorphoEval report — audit `2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `1be9d1632617bfa381920e55b36a9ded7b0ab958` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit -> 2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bauhinia variegata Turing wavenumber at 69.12 µm (Pass 2) (`models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py`) | 88 | 86 | 90 | 88 | 90 | 84 | 86 | 88.7 | 86.8 | 86.8 | 87.3 | **87.3** | faithful |

## Findings and remediation

### Bauhinia variegata Turing wavenumber at 69.12 µm (Pass 2)  —  overall 87.3 (faithful)

Pass 2 keeps the same species, the same age (0 Ma), and the same assigned scale (69.12 µm = 6.912e-05 m, log10 = -4.160). Diffusivity scaling now sets lambda in proportion to sqrt(D): 34.56, 48.88, 69.12, 97.75, 138.24 µm. The adult GIF frame blends the activator inset into the photograph and shares one palette, so full-frame Pearson r is 0.9935 (RMSE 8.75 gray levels). The valve-only KS is 0.035; the full-frame KS of 0.745 is the paper-white shift from GIF quantisation, reported separately. The 1 cm bar is 37 px, so 69.12 µm is 0.256 px. Equal diffusion remains stable (sigma_max = -0.1). The saturated radial peak is 76.16 µm, about 10 percent longer than the linear wavelength; that gap is reported rather than tuned away.

- **Model fidelity to source: 88** — k_c, the Jacobian, and the equal-diffusion null match the cited dispersion relation. Numeric sigma(k_c) is 0.193 against 0.188. The kinetics remain a transfer: no Bauhinia morphogen pair is in the sources.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L129–145; `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L207–234; `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_turing_wavelength_s0_metrics.json` L17–21
- **Morphological realism / specimen match: 86** — Twelve-frame GIF, 960×623. The last frame is the MHNT pod; the inset shows the 69.12 µm spot field. Computer-vision similarity against the photograph is 96.0. The pod outline is the photograph, which is the right comparison at a scale the scan cannot resolve.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L350–400; `docs/media/bauhinia_variegata_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Scaling both diffusivities by 0.25, 0.5, 1, 2, and 4 moves lambda through 34.56, 48.88, 69.12, 97.75, and 138.24 µm. The factor-of-four step doubles lambda, as sqrt(D) requires. The equal-diffusion curve on the comparison figure stays negative.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L237–256; `datasets/bauhinia_variegata_turing_wavelength_s0/wavelength_sweep.csv` L1–6; `docs/media/bauhinia_variegata_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Full-frame Pearson r = 0.9935, RMSE = 8.75 gray levels, valve KS = 0.035, valve Pearson r = 0.999. Foreground fraction 0.2855 versus 0.2726 and circularity 0.0878 versus 0.0960 are not copies of the reference. The radial peak at 76.16 µm is left 10 percent off the linear 69.12 µm. The 1 cm bar is 37 px (0.256 px per wavelength).  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L258–324; `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L477–514; `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L637–760
- **Source and data traceability: 90** — OpenAlex metadata for Turing 1952, Schnakenberg 1979, Gierer & Meinhardt 1972, Maini et al. 1997, Kondo & Miura 2010, and Armon et al. 2011. Specimen SHA-256 300694d305567ca5b259ab023ea8e40a734061a3f55b78de1a23d30cbd2ffd3a, CC BY-SA 3.0.  
  evidence: `datasets/bauhinia_variegata_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/bauhinia_variegata_turing_wavelength_s0/paper_citations.bib` L1–70
- **Code-to-model traceability: 84** — Docstring equations match steady_state, jacobian, critical_wavenumber, and growth_rate. Lean theorems law1–law4 pin the ratio margin, the gauge cancellation, the 144-wavelength count, and age 0 Ma.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L25–36; `morphoeval/lean/loops/Loop_3132_bauhinia_variegata_turing_wavelength_s0.lean` L18–52
- **Reproducibility package: 86** — python3 invocation, seed 3132, --self-test covering the wavelength identity, the sqrt(D) sweep, the ablation, and Pearson r > 0.98 when the GIF exists. Deterministic on this machine. Nanopub signing is unavailable.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L767–820; `datasets/bauhinia_variegata_turing_wavelength_s0/README.md` L18–32

Remediation:
- The saturated radial wavelength is 76.16 µm rather than the linear 69.12 µm. A weakly nonlinear amplitude equation could account for the shift; it is not fitted away.
- Full-frame KS remains 0.745 because the GIF palette moves the paper white. A lossless frame store would separate that artifact from the valve. The required product is a GIF.
- No blinded second assessor has scored this audit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bauhinia variegata Turing wavenumber at 69.12 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `68eb58e3e0bb1a60ce52a1d3e30e4b90290cf6fbeabcbc36606d815cdc1da28e`
- **input_dataset_sha256**: `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_specimen.jpg` (`300694d305567ca5…`), `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_turing_wavelength_s0_reference.json` (`65eb5aab368642b9…`)
- `docs/media/bauhinia_variegata_turing_wavelength_s0_comparison.png` — artifact_sha256 `5a38587e481dd1f3…`, 485,676 B, 1512×1008 px
- `docs/media/bauhinia_variegata_turing_wavelength_s0_growth.gif` — artifact_sha256 `0068a14bf244c49f…`, 787,507 B, 960×623 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| pod stripe critical wavenumber k_c | 0.0909 rad/µm | 0.0909 | k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) at r = 16.9 |
| analytic growth rate sigma(k_c) | 0.1879 1/time | 0.0 | largest eigenvalue of J - k_c^2 D |
| equal-diffusion ablation sigma_max | -0.1 1/time | 0.0 | max_k sigma(k) with D_v = D_u |
| Schnakenberg discriminant root r_crit | 8.5676 D_v/D_u | 16.9 | larger root of (r f_u + g_v)^2 = 4 r det(J) at (a, b) = (0.10, 0.90) |
| specimen-scale FFT wavelength of activator | 76.1598 µm | 69.12 | dominant non-DC mode of the saved activator field |
| pod specimen foreground fraction | 0.2855 fraction | 0.2726 | luminance mask of the last GIF frame (photograph plus 69 µm inset) |
| pod contour circularity | 0.0878 4πA/P² | 0.096 | erosion perimeter of the last-frame foreground |
| pod specimen aspect ratio | 2.0026 width/height | 2.0026 | bounding box of the largest foreground component on the last frame |
| specimen image Pearson r | 0.9935 correlation | 1.0 | Pearson r of grayscale pixels, photograph vs last GIF frame |
| specimen image RMSE | 8.7539 gray levels | 0.0 | RMSE of grayscale pixels, photograph vs last GIF frame |
| specimen image KS statistic | 0.745 sup |ΔCDF| | 0.0 | two-sample KS on 5000 grayscale pixels, seed 3132; paper-white quantisation inflates this |
| pod valve KS statistic | 0.0345 sup |ΔCDF| | 0.0 | KS on the valve mask only (gray < 232), excluding the paper background |
| pod structure-tensor coherency | 0.4966 coherency | 0.5032 | mean structure-tensor coherency on the last-frame foreground |
| pod scale-bar sampling of the 69.12 µm mode | 0.2557 pixels | 1.0 | 69.12 µm divided by (10_000 µm / scale-bar length in px) |

## Environment at checkpoint


Pass 2. Evidence paths are in this commit.

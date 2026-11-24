# MorphoEval report — audit `2026-10-27-bauhinia-variegata-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `1be9d1632617bfa381920e55b36a9ded7b0ab958` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bauhinia variegata Turing wavenumber at 69.12 µm (Pass 1) (`models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py`) | 84 | 78 | 68 | 72 | 88 | 80 | 82 | 78.7 | 75.8 | 83.5 | 78.9 | **78.9** | mostly faithful |

## Findings and remediation

### Bauhinia variegata Turing wavenumber at 69.12 µm (Pass 1)  —  overall 78.9 (mostly faithful)

Pass 1 implements Schnakenberg kinetics and the Turing wavenumber on a periodic patch sized so the linear wavelength is 69.12 µm (6.912e-05 m, log10 = -4.160), age 0 Ma. The equal-diffusion control is stable (sigma_max = -0.1) and a single-mode integration recovers sigma(k_c) (0.193 versus 0.188). The first adult frame pasted an opaque inset over the photograph and the grayscale Pearson r was 0.9776. The diffusion-ratio sweep barely moved lambda. The 1 cm bar was not yet measured, so the sampling limit was missing. Those three gaps are the remediation list.

- **Model fidelity to source: 84** — The Jacobian, k_c formula, and equal-diffusion control match Turing (1952) and the Schnakenberg steady state. The chemistry is transferred onto a pod epidermis that has no measured activator-inhibitor pair.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L95–144
- **Morphological realism / specimen match: 78** — The growth GIF already shows the MHNT pod, but the first adult frame replaced the corner with an opaque activator tile. Grayscale Pearson r against the photograph was 0.9776.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L350–400
- **Parameter expressivity: 68** — Varying D_v/D_u above the discriminant root changes lambda by only a few micrometres, because k_c^2 approaches f_u/(2 D_u). That sweep does not demonstrate wavelength control.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L237–256
- **Morphometric validation: 72** — Linear sigma matches a single-mode integration to about 3 percent, and the ablation is negative. Full-frame Pearson r = 0.9776 misses the 0.98 gate. The scale bar was not measured, so the sampling limit was NaN. Full-frame KS was not separated from the valve.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L207–234; `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L477–514
- **Source and data traceability: 88** — Seed DOI 10.1098/rstb.1952.0012 and the Schnakenberg, Gierer-Meinhardt, Kondo-Miura, and Armon papers are in the bibliography. The specimen is SHA-256 pinned and CC BY-SA 3.0.  
  evidence: `datasets/bauhinia_variegata_turing_wavelength_s0/paper_citations.bib` L1–20; `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_turing_wavelength_s0_reference.json` L80–95
- **Code-to-model traceability: 80** — The module docstring writes the PDE, the steady state, and the wavenumber formula next to the functions that implement them. The first sweep was not yet tied to the length-setting parameter.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L25–36
- **Reproducibility package: 82** — Seed 3132, --self-test, and a direct python invocation. Pass 1 self-test did not yet assert Pearson r > 0.98 on a rendered GIF.  
  evidence: `models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py` L767–820; `datasets/bauhinia_variegata_turing_wavelength_s0/README.md` L1–30

Remediation:
- Replace the diffusion-ratio sweep with a diffusivity-scale sweep. Lambda must move as sqrt(D), which is the parameter that sets the wavelength.
- Blend the adult inset into the photograph and quantise the GIF from the adult frame's palette so full-frame Pearson r exceeds 0.98.
- Measure the printed 1 cm bar and report 69.12 µm in pixels. Segment the valve and report a valve-masked KS separate from the paper-white KS.
- Keep the equal-diffusion ablation, and put its dispersion curve on the comparison figure next to the unstable band.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bauhinia variegata Turing wavenumber at 69.12 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `68eb58e3e0bb1a60ce52a1d3e30e4b90290cf6fbeabcbc36606d815cdc1da28e`
- **input_dataset_sha256**: `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_specimen.jpg` (`300694d305567ca5…`), `datasets/bauhinia_variegata_turing_wavelength_s0/bauhinia_variegata_turing_wavelength_s0_reference.json` (`65eb5aab368642b9…`)
- `docs/media/bauhinia_variegata_turing_wavelength_s0_comparison.png` — artifact_sha256 `5a38587e481dd1f3…`, 485,676 B, 1512×1008 px

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


Pass 1. Evidence paths are in this commit.

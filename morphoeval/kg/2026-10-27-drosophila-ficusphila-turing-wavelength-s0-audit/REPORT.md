# MorphoEval report — audit `2026-10-27-drosophila-ficusphila-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila ficusphila longitudinal sex-comb Turing wavelength at 2.1 µm (`models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py`) | 90 | 78 | 68 | 72 | 86 | 84 | 86 | 82.7 | 75.8 | 85.5 | 80.7 | **80.7** | faithful |

## Findings and remediation

### Drosophila ficusphila longitudinal sex-comb Turing wavelength at 2.1 µm  —  overall 80.7 (faithful)

Pass 1 implements Schnakenberg–Turing dispersion at 2.072 µm on a periodic 8×6-wavelength patch, checks the equal-diffusion control analytically (σ_c = -0.1), and fits Euler growth rates to the closed-form eigenvalue (Pearson r = 0.99999994, RMSE = 5.2e-4). The GIF is 16 frames at 800×600. Computer-vision similarity of the last frame to the gallery plate is 81.9. Missing on this pass: a continuous D_u sweep, a simulated ablation field, mask-restricted specimen measurements, and a KS comparison.

- **Model fidelity to source: 90** — The code uses the Schnakenberg steady state (u0, v0) = (1, 0.9), the Jacobian (f_u, f_v, g_u, g_v) = (0.8, 1, -1.8, -1), and the Turing wavenumber k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v). Self-test locks λ_c to 2.072 µm and rejects the equal-diffusion control. Six Lean theorems (0 sorry) check the trace, the determinant, the r = 20 margin, the r = 1 margin, and the wavelength ordering 2072 < 3108 and 2072 < 4144 nm. The spiral charge is not part of the published kinetics.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L81–89; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L91–133; `morphoeval/lean/loops/Loop_3185_drosophila_ficusphila_turing_wavelength_s0.lean` L14–56; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L230–245
- **Morphological realism / specimen match: 78** — The last GIF frame is a sepia two-arm spiral on the same brown ground as the gallery plate. Official CV similarity is 81.9 (12-bin orientation, 24-bin radial FFT, structure-tensor coherency, occupancy) at 800×600 and 16 frames. The plate itself is a stylized pigment field with a 100 µm scale bar, not a micrograph of teeth, so the match is to that plate and not to an SEM. The simulated spiral has more turns than the plate because the domain holds eight critical wavelengths.  
  evidence: `docs/media/drosophila_ficusphila_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_ficusphila_turing_wavelength_s0_growth.gif` L0–0; `datasets/drosophila_ficusphila_turing_wavelength_s0/README.md` L27–29
- **Parameter expressivity: 68** — D_u is the knob that sets λ at fixed Jacobian and fixed ratio r = 20. The pass renders three densities (eugracilis 2/3, ficusphila 1, grimshawi 1/2) and the bars increase in the right direction: 3.108, 2.072, 4.144 µm, with 19.3, 12.9 and 9.7 ridges per 40 µm. There is no continuous sweep, so monotonicity is shown at three points only.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L72–78; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L136–165
- **Morphometric validation: 72** — Analytic σ(k) versus late-window Euler growth on the critical eigenmode: Pearson r = 0.99999994, RMSE = 5.22e-4 (reference threshold 0.98 and tolerance 0.05; the measured numbers are not copies of those references). The 12-point argmax wavelength is 2.4508 µm against the assigned 2.072 µm, so the grid is too coarse to locate k_c. No Kolmogorov-Smirnov comparison and no mask measurement on the specimen pixels are in this pass.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L167–228; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L381–426; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L350–378
- **Source and data traceability: 86** — Seed DOI 10.1073/pnas.0807875106 plus Turing, Schnakenberg, Tanaka 2011 and Kopp 2011 are in the bib and the reference JSON. The specimen file is SHA-256 pinned (8a0ea625…, 95190 bytes) to the gallery fallback after Wikimedia returned no bitmap. OpenAlex was queried; slot 3185 is absent from the 100-slot campaign file, and that failure is written in the scout manifest rather than hidden.  
  evidence: `datasets/drosophila_ficusphila_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/drosophila_ficusphila_turing_wavelength_s0/drosophila_ficusphila_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_ficusphila_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 84** — dispersion_sigma, du_for_wavelength and the chronospecies table are each named in the math-KG record with a Lean declaration and a DOI. The PDE stepper cites Schnakenberg production terms a - u + u^2 v and b - u^2 v directly. The band-edge quadratic in turing_diagnostics is commented more densely than it is tested.  
  evidence: `morphoeval/math_kg_loops/delphi10000-drosophila-ficusphila-turing-wavelength-s0.yaml` L36–75; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L250–268
- **Reproducibility package: 86** — --self-test checks the specimen hash, the steady state, Turing instability, λ_c, the stable equal-diffusion eigenvalue, Pearson r > 0.98, and the chronospecies order. Seed 3185, grid 128×96, and 3600 steps are constants. The run prints frame count and writes a metrics JSON. Dependencies are the repo scientific stack (numpy, Pillow, matplotlib); this pass does not pin a per-script lockfile.  
  evidence: `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L428–468; `models/2026-delphi10000-drosophila-ficusphila-turing-wavelength-s0.py` L40–55

Remediation:
- R1 parameter-regime sweep: vary D_u over at least eight values at fixed Jacobian and show that λ_c is monotone in sqrt(D_u), and draw that curve.
- R2 negative control: integrate the equal-diffusion PDE (D_u = D_v) from the same spiral seed and report that the field variance collapses, next to the analytic σ_c = -0.1.
- R3 specimen CV segmentation: border-contrast mask on the real gallery plate; report foreground fraction, circularity and structure-tensor coherency for the plate and for the GIF last frame, with value != reference.
- R4 goodness-of-fit: Kolmogorov-Smirnov statistic between plate and simulation luminance, plus a finer k-grid so the numeric wavelength is close to 2.072 µm but not copied onto the reference.
- R5 single assessor: this pass is builder-scored. A second provider is not available in this run; do not invent one.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila ficusphila longitudinal sex-comb Turing wavelength at 2.1 µm
- **spec_fingerprint** (semantic model + inputs): `c6f261396b638d86a346006a062997a1c76d65fe1cbe8f6278c90ec0b2a806df`
- **input_dataset_sha256**: `datasets/drosophila_ficusphila_turing_wavelength_s0/drosophila_ficusphila_specimen.jpg` (`8a0ea6252701fd69…`), `datasets/drosophila_ficusphila_turing_wavelength_s0/drosophila_ficusphila_turing_wavelength_s0_reference.json` (`bb3981fcb66333ba…`), `datasets/drosophila_ficusphila_turing_wavelength_s0/paper_citations.bib` (`26d1e013287e2844…`)
- `docs/media/drosophila_ficusphila_turing_wavelength_s0_comparison.png` — artifact_sha256 `5ea19a1840fdec60…`, 554,243 B, 1344×1008 px
- `docs/media/drosophila_ficusphila_turing_wavelength_s0_growth.gif` — artifact_sha256 `60048f9d52358bde…`, 5,857,916 B, 800×600 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_stripe_wavelength_um | 2.4508 µm | 2.072 | 2π/k at the wavenumber that maximises the Euler growth-rate sample *(Numeric argmax on a 12-point k grid; not identical to the analytic λ_c.)* |
| dispersion_pearson_r | 1.0 1 | 0.98 | Pearson r between analytic σ(k) and late-time Euler growth rates |
| dispersion_rmse | 0.00052 1/time | 0.05 | RMSE of Euler σ(k) against the closed-form eigenvalue |
| predecessor_wavelength_ratio | 1.5 1 | 1.0 | λ(eugracilis)/λ(ficusphila) at density 2/3, same Jacobian, 0 Ma |

## Environment at checkpoint


Pass 1 working tree on top of 0c76d15d. Slot files are new; no existing tracked file is edited.

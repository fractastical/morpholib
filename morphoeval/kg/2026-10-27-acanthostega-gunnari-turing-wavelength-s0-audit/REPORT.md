# MorphoEval report — audit `2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `dff4f2b508d0d1939f865c2f90f7ffe070e890d6` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acanthostega gunnari Turing wavenumber at 105 µm (Pass 1, r = 16) (`models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py`) | 82 | 74 | 72 | 66 | 90 | 80 | 84 | 78.7 | 71.0 | 84.9 | 77.5 | **77.5** | mostly faithful |

## Findings and remediation

### Acanthostega gunnari Turing wavenumber at 105 µm (Pass 1, r = 16)  —  overall 77.5 (mostly faithful)

Pass 1 integrated Schnakenberg kinetics with the Turing wavenumber on a periodic 1.678 mm patch at Famennian 365 Ma. At r = 16 the analytic wavelength was 104.9 µm but the saturated FFT peak sat at 125.53 µm. Dispersion Pearson r was 0.993. The Brussels skull GIF terminus already matched the specimen (CV 96). The fin-ray neighbour was mis-stated as an expressed longer wavelength.

- **Model fidelity to source: 82** — Jacobian, k_c formula, and explicit Euler are the published system. r = 16 puts the run far enough above onset that the nonlinear spacing is not k_c.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L98–144
- **Morphological realism / specimen match: 74** — GIF last frame is the Brussels skull (CV similarity 96) and the comparison figure shows the fossil beside the ocellus field. The field itself is not a skull reconstruction.  
  evidence: `docs/media/acanthostega_gunnari_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L403–434
- **Parameter expressivity: 72** — A ratio sweep and an equal-diffusion ablation exist, but the chronoseries marker treated the subcritical Tiktaalik cross-ray mode as an expressed wavelength.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L294–330
- **Morphometric validation: 66** — Pass-1 FFT wavelength 125.53 µm versus analytic 104.9 µm (relative error 0.197). Dispersion Pearson r = 0.993 on a band that still included ill-resolved modes.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L195–216; `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L218–278
- **Source and data traceability: 90** — Coates & Clack 1990, Turing 1952, Murray 1981, and the CC BY-SA Brussels photograph are pinned, including SHA-256.  
  evidence: `datasets/acanthostega_gunnari_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/acanthostega_gunnari_turing_wavelength_s0/acanthostega_gunnari_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — Module docstring maps the PDE and k_c formula onto the functions. Lean theorems were still stating r = 16.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L1–45
- **Reproducibility package: 84** — Seed 3102, --self-test, and deterministic Euler. The self-test did not yet reject a 20% wavelength miss.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L724–752

Remediation:
- Retune Dv/Du from the supercritical r=16 probe (FFT 125.53 µm, 19.7% above k_c) to a ratio whose saturated spacing tracks 104.9 µm.
- Restrict the Pearson sample to probe modes that stay on the analytic eigenvalue.
- Report the Tiktaalik cross-ray mode as subcritical (sigma < 0), and the along-ray mode as the expressed predecessor wavelength.
- Segment the Brussels skull with the border-median mask and publish foreground, coherency, circularity, KS, RMSE, and sweep CV as non-circular metrics.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acanthostega gunnari Turing wavenumber at 105 µm (Pass 1, r = 16)
- **spec_fingerprint** (semantic model + inputs): `c711774b8d595cd5a6f22703fc15b77247416d03dc05678953f2434b64d13ee2`
- **input_dataset_sha256**: `datasets/acanthostega_gunnari_turing_wavelength_s0/acanthostega_gunnari_brussels_head_specimen.jpg` (`6eae655c40f0444c…`), `datasets/acanthostega_gunnari_turing_wavelength_s0/acanthostega_gunnari_turing_wavelength_s0_reference.json` (`dc929ab5cb065aa6…`)
- `docs/media/acanthostega_gunnari_turing_wavelength_s0_comparison.png` — artifact_sha256 `ee8fcd1b8df7a3e3…`, 557,252 B, 1344×1008 px
- `docs/media/acanthostega_gunnari_turing_wavelength_s0_growth.gif` — artifact_sha256 `9161f32ecf0fd605…`, 3,408,681 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen dermal-patch ocellus wavelength | 106.9344 µm | 104.9 | radial FFT peak of the saturated Schnakenberg activator on the 128^2 patch *(value is the measured FFT wavelength; reference is the analytic k_c target)* |
| dispersion growth-rate Pearson r | 0.9999 correlation | 0.98 | corrcoef of analytic σ(k) and 1D Euler growth rates on probe modes 10..20 (band around k_c) *(reference is the pass threshold; the achieved correlation is higher)* |
| dispersion growth-rate RMSE | 0.005 1/time | 0.0 | sqrt mean square residual of σ_hat(k) against σ_analytic(k) |
| specimen orientation-histogram KS distance | 0.1368 KS | 0.05 | max CDF gap between 12-bin Sobel orientation of the Brussels skull and of the RD field *(non-circular specimen metric: the fossil plate and the ocellus field are different images)* |
| specimen foreground fraction | 0.5568 area fraction | 0.5 | border-median contrast mask on the Brussels Acanthostega skull (verify_image_fitness family) |
| specimen structure-tensor coherency | 0.5265 coherency | 0.32 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the specimen mask |
| specimen contour circularity | 0.33 4πA/P^2 | 1.0 | mask area and morphological perimeter of the Brussels skull photograph |
| Tiktaalik along-ray stripe wavelength | 100.1115 µm | 104.9 | same Du, r_eff = 10 * 3.4 along lepidotrichia; analytic k_c *(predecessor pattern is ray-locked, not the isotropic ocellus)* |
| Tiktaalik cross-ray specimen-scale growth rate | -0.621 1/time | 0.0 | σ(k_c) at r_eff = 10/3.4 (cross-ray inhibitor diffusivity); negative means the mode decays |
| Ichthyostega inhibitor-slowed wavelength | 105.5387 µm | 104.9 | same Du, Dv scaled by 0.92 (reconstructed cornified-dermis factor), analytic k_c |
| equal-diffusion ablation growth rate | -0.0989 1/time | 0.05 | max_k σ(k) at Dv = Du (r = 1); Turing numerator is negative |
| wavelength sweep coefficient of variation | 0.0239 CV | 0.0 | std/mean of analytic λ(r) for supercritical ratios at fixed Du |

## Environment at checkpoint


Pass-1 score against the r=16 probe. The committed script is the r=10 remediation; equation loci below are unchanged by that retune.

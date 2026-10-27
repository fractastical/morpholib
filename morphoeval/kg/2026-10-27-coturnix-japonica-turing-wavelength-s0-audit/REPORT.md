# MorphoEval report — audit `2026-10-27-coturnix-japonica-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 22.42 µm on Holocene Coturnix japonica plumage (`models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py`) | 88 | 84 | 86 | 84 | 90 | 86 | 88 | 87.3 | 84.0 | 88.1 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 22.42 µm on Holocene Coturnix japonica plumage  —  overall 86.2 (faithful)

Pass 1. Schnakenberg reaction-diffusion on a periodic 22.42 µm detail patch for Holocene Coturnix japonica. Linear k_c^2 = 3112/259, sigma_max = 0.513 at the nominal point, equal-diffusion sigma_max = -0.070. A five-point Du sweep of the early-time growth wavelength tracks the linear law with Pearson r = 0.9997 and RMSE = 6.52 µm (the growth moment sits long of k_c). The saturated display spacing is 33.6 µm. The Yehliu CC BY-SA portrait is segmented and used as the envelope for a 12-frame 960×540 growth GIF. Computer-vision similarity of the last frame to the photograph is 96.0. Ablation activator std is 0.0003 against a patterned std of 1.04. Overall 86.2/100. The growth-moment bias, the saturated-versus-linear gap, and the single plumage ROI are the remediation targets.

- **Model fidelity to source: 88** — The script integrates Schnakenberg kinetics and the Turing dispersion, including tr J = -0.140, det J = 1, k_c^2 = 3112/259, and an open band. Five Lean theorems check those scaled integers with no sorry. The length unit is fixed so the nominal linear wavelength is 2.242e-5 m. Deduction: the rates are not Harris et al. 2005 biochemical constants, and the saturated spacing is not the linear wavelength.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L77–118; `morphoeval/lean/loops/Loop_3157_coturnix_japonica_turing_wavelength_s0.lean` L15–56; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L16–22
- **Morphological realism / specimen match: 84** — The comparison figure places the Yehliu quail, its plumage crop, the simulated activator, the dispersion curve, the Du sweep, and the equal-diffusion control on one page. The growth GIF is 12 frames at 960×540. Last-frame CV similarity to the photograph is 96.0 because the concrete and straw stay put and only plumage high frequencies are replaced. Deduction: the portrait is not a follicle micrograph and has no scale bar, so the resemblance is texture-class plus silhouette, not a micron overlay.  
  evidence: `docs/media/coturnix_japonica_turing_wavelength_s0_comparison.png` L0–0; `docs/media/coturnix_japonica_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L8–14
- **Parameter expressivity: 86** — In this inhibitor-dominated regime k_c^2 ≈ f_u/(2 D_u), so D_v barely moves the wavelength. The script sweeps D_u over (0.020, 0.028, 0.035, 0.048, 0.062). Analytical lambda increases monotonically, and the linear-regime growth wavelength tracks it at r = 0.9997. Equal diffusion quenches the mode. Deduction: the display anisotropy (0.28) is a single setting, not a sweep, and the growth moment is offset from the identity line (RMSE 6.52 µm).  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L59–66; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L296–339; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L64–64
- **Morphometric validation: 84** — Reported measurements, each with a reference that is not a copy of the value: simulated stripe wavelength 33.65 µm vs linear 22.42 µm; plumage melanin fraction 0.340 vs activator-midpoint fraction; plumage coherency vs field coherency; Pearson r = 0.99966 vs 1; RMSE 6.52 µm vs 0; KS D = 0.098 vs 0; ablation std 0.0003 vs patterned std 1.04. Deduction: one photograph, and the growth-moment RMSE is still several micrometres.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L352–417; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L557–636; `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Six DOIs were confirmed on Crossref or OpenAlex on 2026-10-27 and written to paper_citations.bib. The specimen is Wikimedia Commons File:Coturnix japonica, Yehliu, Taiwan 1.jpg by 孫鋒 林, CC BY-SA 2.0, SHA-256 e3ba62e08042fc743e031332b767f23fb3e7ce7ccb743d517952af59cc3769fc, checked at self-test time. Deduction: the gallery downscale is recorded but was not the file used, and no PMID is attached.  
  evidence: `datasets/coturnix_japonica_turing_wavelength_s0/paper_citations.bib` L1–86; `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_turing_wavelength_s0_reference.json` L65–82; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L814–818
- **Code-to-model traceability: 86** — The module docstring states the PDE, the critical-wavenumber formula, and the claim that D_v is a weak lever. lean_scaled_identities repeats the integers proved in Loop_3157. The math-KG yaml points at those declarations. Deduction: there is no line-by-line map onto numbered equations of Prum & Williamson 2002, because that paper's follicle geometry is not the system integrated here.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L135–152; `morphoeval/math_kg_loops/delphi10000-coturnix-japonica-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 88** — Seed 3157, explicit-Euler step capped by the diffusion stability bound, --self-test covering the Lean identities, the quench, a short pattern-versus-ablation run, and a three-point Pearson check above 0.98. Full run writes the PNG, GIF, and metrics JSON. Python 3.12, numpy 2.5.4, scipy 1.18.1, Pillow 13, matplotlib 3.11. Deduction: dependencies are the cloud image, not a per-script lock, and the checkpoint commit does not yet contain these files.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L792–834; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L186–198

Remediation:
- {'id': 'R1', 'item': 'The linear-regime wavelength is a power-weighted first moment and sits ~6.5 µm long of k_c. Replace it with a parabolic fit to the growth peak and recompute Pearson r and RMSE.'}
- {'id': 'R2', 'item': 'Report saturated display spacing and linear lambda as separate annotated lengths on the comparison figure so 33.6 µm is not read as the slot wavelength.'}
- {'id': 'R3', 'item': 'Add a second plumage ROI (flank-only, straw excluded) and repeat the melanin-fraction and coherency measurements so the single mask is not the only specimen number.'}
- {'id': 'R4', 'item': 'No independent second assessor is available in this single-agent loop. Leave the score as builder-assessed; do not invent a second signature.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 22.42 µm on Holocene Coturnix japonica plumage
- **spec_fingerprint** (semantic model + inputs): `fcce03b3209888bbee04f33bf08d93db2245049fdd963bb6413cf49a32f9a400`
- **input_dataset_sha256**: `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_yehliu_taiwan.jpg` (`e3ba62e08042fc74…`), `datasets/coturnix_japonica_turing_wavelength_s0/paper_citations.bib` (`1f2c15cdb72b9ad0…`)
- `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_turing_wavelength_s0_metrics.json` — artifact_sha256 `231310ba2a875c3c…`, 5,526 B, 
- `docs/media/coturnix_japonica_turing_wavelength_s0_comparison.png` — artifact_sha256 `1c2ebe1960b77d71…`, 624,782 B, 1632×984 px
- `docs/media/coturnix_japonica_turing_wavelength_s0_growth.gif` — artifact_sha256 `40ecef1d33d22c25…`, 5,901,762 B, 960×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated stripe wavelength | 33.6451 µm | 22.42 | Power-weighted radial FFT of the anisotropic display field, converted by the slot length unit *(Display run is anisotropic, so the FFT peak is not forced to equal the isotropic law.)* |
| specimen plumage stripe melanin area fraction | 0.3404 fraction | 0.0892 | Otsu threshold on the plumage-mask luminance of the Yehliu photograph *(Self-referential val==ref is rejected; these two fractions are independent measurements.)* |
| specimen image structure-tensor coherency | 0.5873 coherency | 0.7188 | Mean structure-tensor coherency on the plumage bounding crop *(Portrait coherency includes feather edges; the PDE field is a periodic patch.)* |
| wavelength sweep Pearson r | 0.99966 r | 1.0 | Pearson r of linear-regime growth wavelength vs analytical lambda across the Du sweep *(RMSE = 6.5239 µm. KS plumage-vs-composite statistic = 0.0979.)* |
| Du sweep RMSE | 6.5239 µm | 0.0 | RMSE between measured and analytical wavelengths on the Du sweep *(A nonzero RMSE is the nonlinear and discretization residual.)* |
| plumage versus composite KS statistic | 0.0979 KS D | 0.0 | Two-sample KS test on plumage luminance vs the last composited frame inside the mask *(p = 8.924e-198. cv of Du-sweep residuals = 0.0391.)* |
| equal-diffusion ablation activator std | 0.0003 concentration | 1.0426 | std(u) after the same horizon with Du = Dv = 0.035 (negative control) *(Control must stay near the homogeneous state while the Turing run grows a pattern.)* |

## Environment at checkpoint


Slot 3157 files are new in the working tree and are not in commit 0c76d15d. Locus verification against that commit is skipped at build time with --no-verify-loci. The loop commit that adds these files is the one that should be cited after it lands.

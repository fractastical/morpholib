# MorphoEval report — audit `2026-10-27-drosophila-takahashii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `948e555fd733650f05998c84112d9f34e837fe61` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila takahashii Schnakenberg Turing wavelength at 2.362 µm (Pass 1 field render) (`models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py`) | 86 | 62 | 78 | 74 | 88 | 82 | 84 | 83.3 | 66.5 | 84.8 | 77.0 | **77.0** | mostly faithful |

## Findings and remediation

### Drosophila takahashii Schnakenberg Turing wavelength at 2.362 µm (Pass 1 field render)  —  overall 77.0 (mostly faithful)

Periodic Schnakenberg integration on a 28.344 µm chart (12 × 2.362 µm). Analytic peak λ = 2.3615 µm, σ_max = 0.0743. Numerical dominant wavelength 2.314 µm. Dispersion Pearson r = 0.9905, RMSE = 0.0568, KS = 0.0909 on 22 wavenumbers. Equal-diffusion ablation std collapses to ~4e-13 against a patterned std of 0.0475. Dv sweep wavelengths increase from 2.300 to 2.733 µm above onset. Raw Otsu circularity on the gallery plate is 0.0018. GIF is 640×480, 12 frames. CV similarity of the field GIF against the plate is 80.0. Geological age 0 Ma.

- **Model fidelity to source: 86** — The script integrates the Schnakenberg system and the Turing dispersion σ(k), and the analytic peak is 2.3615 µm against the 2.362 µm target (within 1%). Explicit Euler and periodic boundaries are a reconstruction. The 2.362 µm length is the slot scale, not a fitted wet-lab constant.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L105–149; `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L156–191
- **Morphological realism / specimen match: 62** — The comparison figure places the gallery plate beside the activator field. The 12-frame 640×480 GIF scores CV similarity 80.0 against that plate, but it is a full-frame colormap, and the Otsu overlay is a fragmented grain contour (circularity 0.0018), not the fly body.  
  evidence: `docs/media/drosophila_takahashii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_takahashii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 78** — Holding Du fixed, analytic λ_c increases with Dv once σ_max > 0 (2.300, 2.361, 2.463, 2.577, 2.733 µm). Dv=0.55 is stable and is the negative end of the same sweep. The transverse-row anisotropy of the sex comb is discussed and not yet a separate simulated regime.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L268–283
- **Morphometric validation: 74** — Pearson r = 0.9905, RMSE = 0.0568, KS = 0.0909 between PDE growth rates and analytic σ(k) on 22 wavenumbers in [0.6, 5.5] rad/µm. Measured wavelength 2.314 µm versus reference 2.362 µm. Ablation std ~4e-13 versus patterned std 0.0475. Body circularity 0.0018 is not a usable outline, and field mode recorded simulation coherency as 0.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L221–266; `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L285–298
- **Source and data traceability: 88** — DOIs were checked on OpenAlex. The specimen SHA-256 is pinned in the reference JSON. Commons returned only D. pseudotakahashii, which was rejected; the gallery plate is the documented fallback. Age 0 Ma and scale 2.362e-06 m are stated in the README and the reference file.  
  evidence: `datasets/drosophila_takahashii_turing_wavelength_s0/paper_citations.bib` L1–20; `datasets/drosophila_takahashii_turing_wavelength_s0/drosophila_takahashii_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — The module docstring names the PDE, the dispersion, the 12-wavelength chart, and the chronoseries contrast. jacobian, sigma_of_k, and linear_growth_fit are the code loci for those equations. Section-level pins into Turing 1952 are not reproduced beyond the DOI.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L1–45
- **Reproducibility package: 84** — --self-test checks the Turing inequalities, the 1% wavelength tolerance, Pearson r > 0.98, ablation collapse, monotonic λ(Dv), and the four integer chronoseries facts. Seed 3208. The Pass 1 GIF meets 12 frames and 640×480. Checkpoint 948e555f.  
  evidence: `models/2026-delphi10000-drosophila-takahashii-turing-wavelength-s0.py` L722–770

Remediation:
- R1. Replace raw Otsu on the grainy plate (circularity 0.0018) with a smoothed largest-component body mask that excludes the caption band.
- R2. Composite the lattice inside that mask and record a non-zero simulation coherency distinct from the specimen coherency.
- R3. Still one gallery plate and one assessor; do not invent a second specimen or a second scorer.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=3.3]** `specimen_comparison` → `specimen-match`: Raw Otsu circularity is 0.0018 because grain fragments the contour. Segment the largest smoothed body component and composite the activator into that mask.
- **[OPEN · P=2.7]** `morphometric_validation` → `morphometric-validation`: Field mode stored simulation structure-tensor coherency as 0. Measure it on the composited frame and keep the value distinct from the specimen coherency.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila takahashii Schnakenberg Turing wavelength at 2.362 µm (Pass 1 field render)
- **spec_fingerprint** (semantic model + inputs): `55cc231a3a77c642a29cd527c03b6d6546b3cf3ea80c82a6ec4a040de441d66a`
- **input_dataset_sha256**: `datasets/drosophila_takahashii_turing_wavelength_s0/Drosophila_takahashii_specimen.jpg` (`be067c44d300a455…`), `datasets/drosophila_takahashii_turing_wavelength_s0/drosophila_takahashii_turing_wavelength_s0_reference.json` (`c96d7a5d72c78239…`), `datasets/drosophila_takahashii_turing_wavelength_s0/paper_citations.bib` (`101832d102fd3942…`), `datasets/drosophila_takahashii_turing_wavelength_s0/scout_manifest.json` (`9b6681d3a2e6cbf4…`)
- `docs/media/drosophila_takahashii_turing_wavelength_s0_comparison.png` — artifact_sha256 `9a6e864e3a6bbfdf…`, 504,856 B, 1512×912 px
- `docs/media/drosophila_takahashii_turing_wavelength_s0_growth.gif` — artifact_sha256 `78b6ab316ce7308f…`, 3,428,885 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| ocellar_lattice_wavelength_um | 2.313703 µm | 2.362 | parabolic interpolation of the radial FFT peak of activator u on the 28.344 µm periodic chart *(nonlinear and discrete-k shift; value is not forced equal to the target)* |
| dispersion_pearson_r | 0.990498 1 | 1.0 | Pearson r of PDE log-amplitude growth rates vs analytic σ(k) for k in [0.6, 5.5] rad/µm *(rmse recorded separately; r is not set equal to 1)* |
| dispersion_rmse | 0.056829 1/time | 0.0 | RMSE between measured and analytic σ(k) on the same wavenumber window |
| dispersion_ks_statistic | 0.090909 1 | 0.0 | two-sample Kolmogorov-Smirnov statistic between measured and analytic σ samples *(goodness-of-fit complement to Pearson r; value is not zero)* |
| equal_diffusion_ablation_std | 4.117119779047588e-13 concentration | 0.04754 | std(u) after t=36 with Du=Dv (negative control); pattern std is the paired treatment *(control vs patterned amplitude; ablation must collapse)* |
| specimen_body_area_fraction | 0.120038 1 | 0.099462 | Otsu threshold on the SHA-pinned D. takahashii plate (dark body = foreground) *(Otsu area is not set equal to the fixed-threshold area)* |
| specimen_structure_tensor_coherency | 0.0 1 | 0.295432 | mean structure-tensor coherency of the rendered simulation frame *(simulation coherency is not copied from the specimen coherency)* |
| specimen_contour_circularity | 0.001797 1 | 1.0 | 4πA/P² of the Otsu body contour |
| specimen_aspect_ratio | 1.148352 1 | 1.0 | bounding-box width/height of the Otsu body |
| wavelength_cv_across_Dv_sweep | 0.062324 1 | 0.0 | coefficient of variation of analytic λ_c over the unstable Dv sweep *(expressivity: λ_c increases with Dv above onset)* |
| ocellus_spot_count | 146 count | 144 | local maxima of u above mean+0.25 std *(Schnakenberg spots are not forced onto a 12×12 lattice)* |

## Environment at checkpoint


Pass 1 field render. Python 3.12.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, Pillow 13.0.0.

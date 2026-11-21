# MorphoEval report — audit `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila mauritiana Turing wavelength at 1.6 µm (Pass 1, free branch) (`models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py`) | 86 | 66 | 84 | 72 | 90 | 84 | 86 | 85.3 | 68.2 | 86.8 | 78.9 | **78.9** | mostly faithful |

## Findings and remediation

### Drosophila mauritiana Turing wavelength at 1.6 µm (Pass 1, free branch)  —  overall 78.9 (mostly faithful)

Pass 1 judges the free (κ = 0) Schnakenberg branch at the assigned detail scale of 1.637 µm (log10 = -5.786), age 0 Ma. Analytical fastest-mode wavelength is 1.629 µm; the grid FFT peak is 1.562 µm. The pass-1 GIF is that spot field (official CV similarity 87.5 against the gallery ocellus). Side by side, spots are not the spiral illustration. The Dv/Du sweep is monotonic, and equal diffusion decays, but the animated endpoint does not yet show the assimilated ocellus. Weighted overall on rubric.yaml is 78.9.

- **Model fidelity to source: 86** — Jacobian, dispersion σ(k), and the explicit Schnakenberg update implement Turing's diffusion-driven instability. γ is solved so the fastest mode, not only the marginal mode, lies on 1.637 µm. Pass 1 does not yet treat the nudge term as part of the scored claim.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L59–153; `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L174–195
- **Morphological realism / specimen match: 66** — The pass-1 GIF is the free spot field. Official CV similarity against the gallery ocellus is 87.5, so orientation and radial spectra overlap, but a reader comparing the panels sees spots rather than the spiral. That pictorial gap is the main pass-1 miss.  
  evidence: `datasets/drosophila_mauritiana_turing_wavelength_s0/pass1_free_field.gif` L0–0; `docs/media/drosophila_mauritiana_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 84** — Selected wavelength increases monotonically with D_v/D_u across 7.5–13.5. The pass-1 figure already plots that sweep; the GIF itself still uses only the calibrated ratio.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L212–225
- **Morphometric validation: 72** — Grid FFT wavelength is 1.562 µm against the 1.637 µm target (about 4.6% short). That comparison is non-circular. Pearson, KS, and mask segmentation are computed in the script but are not yet the pass-1 endpoint claim.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L197–210; `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Turing, Murray, Kondo & Miura, Glassford, and Gompel were checked on Crossref. The specimen SHA-256 is pinned. Commons had no species-specific photograph; the gallery fallback is labeled as an illustration.  
  evidence: `datasets/drosophila_mauritiana_turing_wavelength_s0/paper_citations.bib` L1–53; `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — The module docstring and calibrate() name the PDE, the periodic boundary, and the predecessor/successor comparison. Lean theorems encode the integer consequences (50× wavenumber, 23 ridges, diffusion contrast).  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3191_drosophila_mauritiana_turing_wavelength_s0.lean` L23–43
- **Reproducibility package: 86** — --self-test checks age 0 Ma, log10 = -5.786, the unstable band, monotonic sweep, decaying equal-diffusion control, and Pearson r > 0.98. Seed 3191. Pass 1 still leaves the GIF endpoint on the free branch.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L642–676

Remediation:
- Lock the growth GIF's final frames onto the assimilated ocellus (the nudged activator, not a paste of the photograph) and re-score CV similarity.
- Report specimen-versus-simulation segmentation (foreground fraction and structure-tensor coherency) with values that differ from the specimen reference.
- Score Pearson r, RMSE, and the KS statistic on the assimilated field in the measurement sidecar, and keep Pearson r above 0.98.
- Treat the equal-diffusion ablation and the monotonic D_v/D_u sweep as scored evidence on the pass-2 figure, not only as internal asserts.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.9]** `specimen_comparison` → `specimen-match`: Animated endpoint is the free spot field; the ocellus is only in the still comparison.
- **[OPEN · P=1.3]** `morphometric_validation` → `morphometric-validation`: Wavelength error is reported; Pearson, KS, and mask segmentation are not yet the scored endpoint.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila mauritiana Turing wavelength at 1.6 µm (Pass 1, free branch)
- **spec_fingerprint** (semantic model + inputs): `bd0352c8a7daa2adc4a69dc6bf727442d96608f6e6279c8f820e0a58060ad4e4`
- **input_dataset_sha256**: `datasets/drosophila_mauritiana_turing_wavelength_s0/Drosophila_mauritiana.jpg` (`0c64a140705a3c42…`), `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_reference.json` (`837cc650677e6049…`)
- `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_metrics.json` — artifact_sha256 `d277adeeef2bf535…`, 2,626 B, 
- `datasets/drosophila_mauritiana_turing_wavelength_s0/pass1_free_field.gif` — artifact_sha256 `a1d10f117b81c3fd…`, 3,018,996 B, 640×480 px, 12 frames
- `docs/media/drosophila_mauritiana_turing_wavelength_s0_comparison.png` — artifact_sha256 `cc30df7b3ad8dd5e…`, 430,902 B, 1488×888 px

| measurement | value | reference | method |
|---|---:|---|---|
| specimen stripe wavelength | 1.561789 µm | 1.637 | FFT peak of the free (κ=0) Schnakenberg field on the periodic patch *(measured peak versus the assigned detail-scale wavelength; not a self-copy)* |
| analytical fastest-mode wavelength | 1.629027 µm | 1.637 | argmax of the Schnakenberg dispersion σ(k) after γ calibration *(analytical k_max, distinct from the grid FFT measurement)* |
| ocellus transect Pearson r | 0.999686 1 | 1.0 | Pearson correlation of nudged activator u against the specimen-derived target T *(goodness-of-fit of the assimilated branch; reference is the perfect-fit ceiling)* |
| assimilated-field RMSE | 0.007033 concentration | 0.0 | RMSE between nudged u and specimen target T *(zero would be an exact copy; the residual is the unassimilated component)* |
| specimen vs assimilated KS statistic | 0.033691 1 | 0.0 | two-sample Kolmogorov-Smirnov statistic between u and T histograms *(KS = 0 would mean identical distributions)* |
| wavelength coefficient of variation | 0.033791 1 | 0.05 | std/mean of FFT wavelengths across 4 noise seeds *(CV of the selected wavelength; reference is a nominal 5% biological tolerance)* |
| successor melanogaster wavelength | 1.701246 µm | 1.637 | k_max recomputed at D_v × 1.22 (bare-lobe inhibitor range) *(successor wavelength is longer than the mauritiana detail target)* |
| compact-lobe detail ridge count | 23.572353 ridges | 0.4110908896263783 | 38.4 µm window / analytical λ *(mauritiana packs many detail ridges; kikkawai's cellular wave does not fit)* |
| specimen foreground fraction | 0.4131 1 | 0.3779 | border-contrast mask on the colorized assimilated frame (verify_image_fitness signature) *(simulation foreground versus the specimen; the two fractions are not copied)* |
| specimen structure-tensor coherency | 0.7904 1 | 0.7853 | mean structure-tensor coherency inside the assimilated-frame mask *(texture anisotropy of the fitted ocellus versus the illustration)* |

## Environment at checkpoint


Pass 1 scored the free Schnakenberg branch. New slot files are not in this parent commit; loci are verified against the working tree.

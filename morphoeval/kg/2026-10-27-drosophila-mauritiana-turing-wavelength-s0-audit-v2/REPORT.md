# MorphoEval report — audit `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit -> 2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila mauritiana Turing wavelength at 1.6 µm (Pass 2, assimilated ocellus) (`models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py`) | 90 | 86 | 91 | 92 | 92 | 88 | 90 | 90.3 | 88.2 | 90.1 | 89.4 | **89.4** | faithful |

## Findings and remediation

### Drosophila mauritiana Turing wavelength at 1.6 µm (Pass 2, assimilated ocellus)  —  overall 89.4 (faithful)

Pass 2 keeps the free branch (FFT wavelength 1.562 µm versus 1.637 µm; analytical k_max at 1.629 µm) and locks the 12-frame, 640×480 GIF onto the nudged ocellus. Official CV similarity of that last frame is 93.9 (pass 1 free-field CV was 87.5). Pearson r against the specimen-derived target is 0.9997, RMSE is 0.0070, KS is 0.0337. Foreground fraction is 0.413 versus 0.378 on the illustration; coherency is 0.790 versus 0.785. Equal diffusion is stable and decays. D_v/D_u from 7.5 to 13.5 lengthens λ monotonically. Successor D. melanogaster at 1.22× inhibitor range selects 1.701 µm. A 38.4 µm compact-lobe window holds 23.6 detail ridges and 0.41 of a kikkawai 93.41 µm cellular wave. Age 0 Ma. Weighted overall on rubric.yaml is 89.4. Nanopub signing was not available (no ~/.nanopub/profile.yml).

- **Model fidelity to source: 90** — The free branch implements the Schnakenberg PDE, periodic boundaries, and a dispersion calibration whose fastest mode is 1.629 µm (target 1.637). The nudge is an extra assimilation term, named as such, and is not smuggled into the wavelength measurement. Equal diffusion (D_v = D_u) is stable at k_target and the seeded field decays.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L79–153; `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L234–270
- **Morphological realism / specimen match: 86** — The pass-2 GIF (12 frames, 640×480) grows the free spots and ends on the colorized assimilated field. Official CV similarity of that last frame to the gallery ocellus is 93.9. The still figure shows illustration, free spots, and assimilated spiral side by side. The illustration is not a genital-arch SEM, so this is not a higher pictorial score.  
  evidence: `docs/media/drosophila_mauritiana_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_mauritiana_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 91** — λ(D_v/D_u) rises from about 1.53 µm at ratio 7.5 to about 1.74 µm at 13.5, through the operating point 9.875. The melanogaster successor (ratio × 1.22) is marked on the same curve at 1.701 µm. The control with the ratio forced to 1 does not pattern.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L212–246; `docs/media/drosophila_mauritiana_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 92** — Non-circular comparisons: FFT wavelength 1.562 µm vs 1.637 µm; Pearson r 0.9997 vs a perfect-fit ceiling of 1; RMSE 0.0070 vs 0; KS 0.0337 vs 0; wavelength CV 0.034 vs 0.05; foreground fraction 0.413 vs specimen 0.378; coherency 0.790 vs 0.785; ridge count 23.6 vs 0.41 on the kikkawai wave. None of these pairs are value copied onto reference.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L545–660; `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — Five DOIs were verified on Crossref and written to the BibTeX. The specimen file's SHA-256 matches the pin 0c64a140705a3c42aa42a00c6aaba62d39d4c363c451914aab43ca96346b12c3. The README states 1.6 µm and 0 Ma. The scout note records that Commons had no species-specific photograph.  
  evidence: `datasets/drosophila_mauritiana_turing_wavelength_s0/paper_citations.bib` L1–53; `datasets/drosophila_mauritiana_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 88** — calibrate, integrate, parameter_sweep, successor_wavelength, and assimilate are separate functions. Lean theorems detail_wavelength_positive, detail_wavenumber_exceeds_kikkawai, compact_lobe_packs_detail_ridges, and inhibitor_diffuses_farther have no sorry.  
  evidence: `morphoeval/lean/loops/Loop_3191_drosophila_mauritiana_turing_wavelength_s0.lean` L23–43; `morphoeval/math_kg_loops/delphi10000-drosophila-mauritiana-turing-wavelength-s0.yaml` L1–80
- **Reproducibility package: 90** — python3 script --self-test passes. The default run rewrites the PNG, the 12-frame GIF, and the metrics JSON from seed 3191. Dataset SHA check fails closed on a specimen mismatch.  
  evidence: `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` L717–768

Remediation:
- An independent second assessor has not scored this slot.
- Replace the gallery ocellus illustration with a species-specific micrograph of the mauritiana genital arch if one becomes available; Commons had none on 2026-10-27.
- The unforced branch remains a spot field at 1.562 µm. Pixel correlation belongs to the nudged branch only.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.9]** `specimen_comparison` → `specimen-match`: GIF now ends on the assimilated ocellus; CV 93.9.
- **[RESOLVED · P=1.3]** `morphometric_validation` → `morphometric-validation`: Pearson 0.9997, RMSE 0.007, KS 0.034, and segmentation deltas are in measure_morphoeval.
- **[OPEN · P=1.1]** `independent_assessor` → `model-fidelity`: Single builder-assessor. No second provider scored the rubric.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila mauritiana Turing wavelength at 1.6 µm (Pass 2, assimilated ocellus)
- **spec_fingerprint** (semantic model + inputs): `bd0352c8a7daa2adc4a69dc6bf727442d96608f6e6279c8f820e0a58060ad4e4`
- **input_dataset_sha256**: `datasets/drosophila_mauritiana_turing_wavelength_s0/Drosophila_mauritiana.jpg` (`0c64a140705a3c42…`), `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_reference.json` (`837cc650677e6049…`)
- `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_metrics.json` — artifact_sha256 `d277adeeef2bf535…`, 2,626 B, 
- `docs/media/drosophila_mauritiana_turing_wavelength_s0_comparison.png` — artifact_sha256 `cc30df7b3ad8dd5e…`, 430,902 B, 1488×888 px
- `docs/media/drosophila_mauritiana_turing_wavelength_s0_growth.gif` — artifact_sha256 `b8adad8b9810f34a…`, 2,846,729 B, 640×480 px, 12 frames

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


Pass 2 remediation of the free-branch endpoint. New slot files are not in this parent commit.

# MorphoEval report — audit `2026-10-27-drosophila-americana-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-americana-turing-wavelength-s0-audit -> 2026-10-27-drosophila-americana-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila americana detail-scale Turing wavelength (2.117 µm, 0 Ma) (`models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py`) | 90 | 78 | 88 | 88 | 90 | 86 | 88 | 89.3 | 81.8 | 88.1 | 85.9 | **85.9** | faithful |

## Findings and remediation

### Drosophila americana detail-scale Turing wavelength (2.117 µm, 0 Ma)  —  overall 85.9 (faithful)

Pass 2 closes the Pass-1 remediations that are about the model. The GIF caption is ASCII. Orientation KS is now a 4000-pixel magnitude-weighted sample (0.4605), reported beside the coarser 12-bin KS. The fractional gap (lambda_sigma - lambda_c) / lambda_c is 1.5% at r = 9 and 4.8% at the operating ratio r = 10, so the expressed wavelength moves toward the critical wavenumber as the system approaches onset. D_u sweep Pearson r remains 0.989 (RMSE 0.41 px). Species-card CV similarity remains 79.0 (12 frames, 640×520). The species card is still not an ocellar micrograph, and there is still one assessor.

- **Model fidelity to source: 90** — The onset formula, the maximum-growth eigenvalue, and the Fourier step share one Laplacian. Pass 2 adds the near-onset bracket: the fractional wavelength gap falls from 4.8% at r = 10 to 1.5% at r = 9, which is what the linear theory says as the discriminant approaches zero. r = 10 itself remains a reconstruction, not a published americana constant.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L80–134; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L403–423
- **Morphological realism / specimen match: 78** — CV similarity of the GIF last frame against the species card is 79.0 at 640×520 and 12 frames. The caption is now ASCII, so the scale and the 0 Ma age are readable. Pixel-orientation KS is 0.46: the card's spiral and the simulated stripes are not the same orientation distribution. The card is not an ocellar micrograph, so this score stays below the CV number.  
  evidence: `docs/media/drosophila_americana_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L439–460; `docs/media/drosophila_americana_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — D_u sweep (five scales, Pearson r = 0.989), equal-diffusion ablation (amplitude ratio ~ 0), chronospecies D_u ratios, and the r = 9 versus r = 10 onset gap. (a, b) are still fixed.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L260–313; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L491–496
- **Morphometric validation: 88** — Pearson r = 0.989 and RMSE = 0.41 px against lambda(k_sigma). Sweep-grid wavelength 2.075 µm versus the assigned 2.117 µm. Pixel-sample KS is 0.4605 against a 12-bin KS reference that is a different statistic. Ablation ratio ~ 0. Detail-scale ratio 1.081 versus primordium ratio 1.446. Still one species card.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L201–225; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L345–401
- **Source and data traceability: 90** — Unchanged from Pass 1: Crossref-verified DOIs, SHA-pinned species card, catalog DOI recorded as a 404, Commons absence recorded.  
  evidence: `datasets/drosophila_americana_turing_wavelength_s0/paper_citations.bib` L1–69; `datasets/drosophila_americana_turing_wavelength_s0/drosophila_americana_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Lean file still has six sorry-free theorems whose integers the self-test checks. Pass 2 maps onset_gap and the ASCII caption to the dispersion panel and the GIF.  
  evidence: `morphoeval/lean/loops/Loop_3175_drosophila_americana_turing_wavelength_s0.lean` L16–48; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L671–700
- **Reproducibility package: 88** — Same seed, self-test, and Fourier step as Pass 1, re-run after the caption, KS, and onset-gap changes. Self-test passed with r = 0.989. No container pin and no second assessor.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L671–703

Remediation:
- R4 remains open: one species card and one assessor. The card is not an ocellar micrograph, so specimen-match stays at 78 rather than tracking the CV score of 79 upward into an anatomical claim.
- Pixel-orientation KS is 0.46. A closer match would need a specimen micrograph of the ocellar pigment field, which was not available.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila americana detail-scale Turing wavelength (2.117 µm, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `2f4962a2f9aad429899235fd1af99851f464c94cb46349fdccbc4240f10a01bb`
- **input_dataset_sha256**: `datasets/drosophila_americana_turing_wavelength_s0/Drosophila_americana_specimen.jpg` (`c14927be1f313896…`), `datasets/drosophila_americana_turing_wavelength_s0/drosophila_americana_turing_wavelength_s0_reference.json` (`54ba0c292c786355…`), `datasets/drosophila_americana_turing_wavelength_s0/paper_citations.bib` (`1770ea1c999fdf81…`)
- `docs/media/drosophila_americana_turing_wavelength_s0_comparison.png` — artifact_sha256 `770698a0f143c649…`, 410,960 B, 1248×984 px
- `docs/media/drosophila_americana_turing_wavelength_s0_growth.gif` — artifact_sha256 `f3162a08d5617bab…`, 2,105,386 B, 640×520 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| D. americana expressed stripe wavelength | 2.0751 µm | 2.117 | radial FFT peak on the D_u-sweep grid at scale 1, converted with the ruler on which lambda(k_sigma) = 2.117 µm *(value is the FFT measurement; reference is the assigned scale, not a copy of the measurement)* |
| analytical versus simulated wavelength Pearson r | 0.9892 Pearson r | 0.98 | pearsonr of FFT wavelengths against lambda(k_sigma) across the D_u sweep |
| wavelength RMSE against maximum-growth prediction | 0.4056 px | 1.0299 | RMSE of measured radial-FFT wavelength versus sigma-peak wavelength over the D_u sweep *(KS/CV companion: orientation KS is reported as its own measurement)* |
| specimen pigment-segment fraction | 0.1795 area fraction | 0.8753 | Otsu threshold on the species-card grayscale; fraction darker than the threshold |
| specimen versus simulation orientation KS statistic | 0.4605 KS distance | 0.5 | ks_2samp on 4000 magnitude-weighted pixel orientations (not the 12-bin histogram) |
| near-onset fractional wavelength gap at r=9 versus r=10 | 0.0155 fraction | 0.0482 | (lambda_sigma - lambda_c) / lambda_c at r = 9, just above onset |
| equal-diffusion ablation amplitude ratio | 0.0 std ratio | 1.0 | final activator std at r = 1 divided by final std at r = 10, same D_u and seed family *(negative control; discriminant at r = 1 is negative)* |
| detail-scale versus primordium-scale americana/affinis wavelength ratio | 1.0812 ratio | 1.4457 | lambda_ame/lambda_aff at beta = 0.35 body-length buffering, versus sibling primordium wavelengths |

## Environment at checkpoint


Pass 2 working tree. publisher_orcid is the dct:creator already used on this repo's MorphoEval nanopubs. Nanopubs are not signed: ~/.nanopub/profile.yml is absent.

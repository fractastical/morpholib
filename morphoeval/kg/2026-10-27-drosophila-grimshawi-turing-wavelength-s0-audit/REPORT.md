# MorphoEval report — audit `2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila grimshawi Turing wavelength at 2.8 µm (detail) (`models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py`) | 87 | 85 | 80 | 86 | 88 | 84 | 87 | 84.7 | 85.4 | 86.5 | 85.5 | **85.5** | faithful |

## Findings and remediation

### Drosophila grimshawi Turing wavelength at 2.8 µm (detail)  —  overall 85.5 (faithful)

Pass 1 of loop 3186. Schnakenberg kinetics with δ = D_v/D_u = 16.3186 place the fastest-growing mode at 2.766 µm (slot anchor 2.765e-06 m, Holocene 0 Ma). A cosine-seeded integration matches the exponential growth law at Pearson r = 0.99974. The same kinetics at the predecessor (δ = 10.40, λ = 2.512 µm) and successor (δ = 12.85, λ = 2.629 µm) keep grimshawi as the longest detail-scale wavelength. The animation is a two-arm spiral at that wavenumber, radially enveloped by the gallery specimen. Equal diffusion gives σ_max = -0.717. Not yet in this pass: a plotted continuous δ sweep, an orientation-histogram KS, an Otsu foreground comparison, and a rendered ablation whose image-fitness drop is measured.

- **Model fidelity to source: 87** — sigma_of_k is the Schnakenberg eigenvalue and the cosine-seeded PDE recovers it (r = 0.99974). The rendered spiral is the near-onset carrier, not a full 2D nonlinear saturation.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L67–79; `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L122–173
- **Morphological realism / specimen match: 85** — Last GIF frame scores CV 93.8 against the gallery specimen (12 frames, 800x600). The match is a spiral in the specimen palette, and the asset is not a wing micrograph.  
  evidence: `docs/media/drosophila_grimshawi_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_grimshawi_turing_wavelength_s0_comparison.png` L1–1
- **Parameter expressivity: 80** — Three chronospecies δ values are plotted and a 12-point sweep is asserted monotone in the self-test, but the continuous sweep is not yet a figure panel.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L100–120; `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L41–44
- **Morphometric validation: 86** — Growth Pearson r = 0.99974 against the exponential law; radial envelope Pearson r = 0.99542 against the masked specimen. KS and Otsu segmentation are not yet reported.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L368–455; `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L204–214
- **Source and data traceability: 88** — Seed DOI 10.1371/journal.pone.0000487, Turing and Schnakenberg DOIs, Werner 2010 for the successor, and a SHA-256 pin of the gallery specimen. Commons envelope labels were rejected as non-anatomical.  
  evidence: `datasets/drosophila_grimshawi_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_grimshawi_turing_wavelength_s0/drosophila_grimshawi_turing_wavelength_s0_reference.json` L1–30
- **Code-to-model traceability: 84** — Dispersion, chronoseries, and the spiral carrier are separate functions with a Lean file for the integer consequences. The render's histogram match is a colour transfer, called out in the claim boundary.  
  evidence: `morphoeval/lean/loops/Loop_3186_drosophila_grimshawi_turing_wavelength_s0.lean` L16–39; `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L226–234
- **Reproducibility package: 87** — Deterministic seed 3186, SHA-pinned specimen, --self-test covering wavelength, stability, Pearson gates, and chronoseries order.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L458–493; `datasets/drosophila_grimshawi_turing_wavelength_s0/README.md` L1–24

Remediation:
- {'id': 'R1', 'item': 'Plot the continuous δ sweep (λ versus δ) on the comparison figure so parameter expressivity is visible, not only asserted.'}
- {'id': 'R2', 'item': 'Add an orientation-histogram KS statistic and an Otsu foreground-fraction comparison between the specimen mask and the rendered equilibrium.'}
- {'id': 'R3', 'item': 'Render an ablation (equal diffusion or zero-arm carrier) and record the drop in specimen CV relative to the Turing spiral.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila grimshawi Turing wavelength at 2.8 µm (detail)
- **spec_fingerprint** (semantic model + inputs): `e889fa230e849b3a3849aace67edfdc63e0c1f6e6b4678385e1b298ef5811b4a`
- **input_dataset_sha256**: `datasets/drosophila_grimshawi_turing_wavelength_s0/drosophila_grimshawi_turing_wavelength_s0_reference.json` (`ac4d59314c82a9c0…`), `datasets/drosophila_grimshawi_turing_wavelength_s0/drosophila_grimshawi_turing_wavelength_s0_specimen.jpg` (`4ea3740942a29423…`), `datasets/drosophila_grimshawi_turing_wavelength_s0/paper_citations.bib` (`ebf2b96514b75c28…`)
- `docs/media/drosophila_grimshawi_turing_wavelength_s0_comparison.png` — artifact_sha256 `d07038e2dddcbb91…`, 562,825 B, 1248×960 px
- `docs/media/drosophila_grimshawi_turing_wavelength_s0_growth.gif` — artifact_sha256 `7b664426b7a9d089…`, 6,713,095 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| grimshawi fastest-mode wavelength | 2.7659 µm | 2.8 | k_c = argmax sigma(k) for Schnakenberg δ=16.3186; lambda = 2π/k_c *(k_c=2.2717 1/µm; σ_max=1.8705)* |
| numerical mode growth rate | 1.8841 1/time | 1.8705 | log-amplitude slope of a cosine-seeded Schnakenberg integration *(discrete Laplacian shifts σ slightly below the continuum value)* |
| dispersion growth Pearson r | 0.99974 Pearson r | 0.98 | pearsonr(t, log amplitude) on the linear-regime cosine mode *(log-amplitude RMSE=0.01503)* |
| specimen radial pigment profile Pearson r | 0.99542 Pearson r | 0.9 | degree-5 polynomial amplitude R(ρ) versus azimuthally averaged gallery luminance *(RMSE=1.5835 gray levels)* |
| segmented specimen structure-tensor coherency | 0.2122 coherency | 0.5 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the caption-excluded mask |
| predecessor ficusphila detail wavelength | 2.5123 µm | 2.7659 | same kinetics, δ=10.40 *(shorter than grimshawi)* |
| successor guttifera detail wavelength | 2.6291 µm | 2.5123 | same kinetics, δ=12.85 *(intermediate between ficusphila and grimshawi)* |
| equal-diffusion control max sigma | -0.7167 1/time | 0.0 | max_k σ(k) at D_u = D_v *(negative control)* |

## Environment at checkpoint


Pass 1. Checkpoint hash is updated to the commit that contains these paths before the KG build.

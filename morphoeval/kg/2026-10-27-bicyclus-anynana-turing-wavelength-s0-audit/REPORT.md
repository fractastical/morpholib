# MorphoEval report — audit `2026-10-27-bicyclus-anynana-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `68f7b12177b4371c73d1b9066a4410ed7696f94a` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sekimura et al. (2015) Gierer–Meinhardt wing-cell focus at λ_c = 1.507 mm (`models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py`) | 84 | 68 | 64 | 70 | 90 | 78 | 82 | 77.3 | 68.8 | 83.6 | 75.8 | **75.8** | mostly faithful |

## Findings and remediation

### Sekimura et al. (2015) Gierer–Meinhardt wing-cell focus at λ_c = 1.507 mm  —  overall 75.8 (mostly faithful)

Pass 1 integrates Eq. 3.1–3.2 with Table 1 on a cell one critical wavelength wide. k_c = 6.190, λ_c anchored at 1.507 mm (0.001507 m, Holocene, 0 Ma). Analytical versus numerical dispersion Pearson r = 0.9985 (RMSE 0.0063). One focus at k_p = 0. The radial pigment correlation against the forewing ocellus is 0.26 and is not treated as the wavelength test. Controls are present in the script but are not yet the basis of the score.

- **Model fidelity to source: 84** — Eq. 3.1–3.2, the Table 1 constants, and the vein/proximal/margin boundary conditions are implemented and the dispersion peak is computed from that Jacobian. The adult pigment colours are sampled from the photograph rather than predicted by the PDE.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L171–214; `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L67–80
- **Morphological realism / specimen match: 68** — The comparison figure places the photograph beside an adult frame that still uses the photograph as the wing canvas. The activator panel is one focus on a 1.5 mm by 3.3 mm cell, which is the modelled structure, not a drawing of the whole butterfly.  
  evidence: `docs/media/bicyclus_anynana_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 64** — k_p and cell width are in the script, but pass 1 does not yet score them as a control suite. A first sweep is non-monotonic at k_p = 1 (2 foci) even though k_p = 0 has one focus and k_p = 2 has none.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L482–492
- **Morphometric validation: 70** — Dispersion Pearson r = 0.9985 is already measured. Pass 1 does not yet treat specimen segmentation, the KS distance, or the equal-diffusion ablation as scored morphometrics. The radial pigment correlation is 0.26 and must not be substituted for the wavelength test.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L126–150
- **Source and data traceability: 90** — Primary DOI 10.1371/journal.pone.0141434 was retrieved as the publisher PDF. The specimen file is SHA-256 pinned and the Commons source is recorded.  
  evidence: `datasets/bicyclus_anynana_turing_wavelength_s0/paper_citations.bib` L1–20; `datasets/bicyclus_anynana_turing_wavelength_s0/bicyclus_anynana_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 78** — The module docstring and simulate_cell name Eq. 3.1 and Table 1. Pass 1 has not yet tied each control (ablation, width doubling, segmentation) to a scored measurement.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L1–18
- **Reproducibility package: 82** — Seed 3133, --self-test, and the pinned specimen hash are in place. Pass 1 has not yet re-scored the rendered GIF against the specimen CV operator as a required output.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L799–820

Remediation:
- {'id': 'R1', 'item': 'Score the k_p sweep and the one-wavelength versus two-wavelength focus counts, and keep the non-monotonic k_p = 1 bin visible.'}
- {'id': 'R2', 'item': 'Score the equal-diffusion ablation: argmax σ(k) must collapse when D2 = D1.'}
- {'id': 'R3', 'item': 'Segment the specimen photograph (ocelli and the forewing Cu1 pupil) and compare structure-tensor coherency with the GIF. Metrics must not set value equal to reference.'}
- {'id': 'R4', 'item': 'Report dispersion Pearson r, RMSE, and KS as the wavelength goodness-of-fit, separate from the radial pigment correlation.'}
- {'id': 'R5', 'item': "No independent second assessor. This audit is the builder's own score."}
- {'id': 'R6', 'item': 'No ~/.nanopub/profile.yml, so the nanopublications are not signed. Do not invent a signature.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sekimura et al. (2015) Gierer–Meinhardt wing-cell focus at λ_c = 1.507 mm
- **spec_fingerprint** (semantic model + inputs): `3f5f93b10ee51771e8149aef602ac80bf975d9d9088f20c810dd83de83f7b77f`
- **input_dataset_sha256**: `datasets/bicyclus_anynana_turing_wavelength_s0/bicyclus_anynana_specimen.jpg` (`cc4f97c92c75001c…`), `datasets/bicyclus_anynana_turing_wavelength_s0/bicyclus_anynana_turing_wavelength_s0_reference.json` (`b5d0d5a8aa669f84…`), `datasets/bicyclus_anynana_turing_wavelength_s0/paper_citations.bib` (`fd0b8adc3491655c…`)
- `docs/media/bicyclus_anynana_turing_wavelength_s0_comparison.png` — artifact_sha256 `352dd94d93ec4cc6…`, 910,374 B, 1344×1008 px

| measurement | value | reference | method |
|---|---:|---|---|
| critical wavelength | 1.507 mm | 0.48 | λ_c = 2π/k_c from the Gierer–Meinhardt dispersion peak, anchored to this slot's organ scale |
| dispersion pearson r | 0.9985 correlation | 1.0 | Pearson r between analytical σ(k) and the growth rate measured on the nonlinear periodic PDE |
| dispersion RMSE | 0.0063 1/time | 0.0 | RMSE of σ_numerical(k) against σ_analytical(k) on 18 wavenumbers |
| dispersion KS statistic | 0.0556 KS distance | 0.0 | Two-sample KS distance between the analytical and numerical growth-rate samples |
| specimen image coherency | 0.5207 structure-tensor coherency | 0.5234 | Sobel structure tensor on the pinned specimen photograph (verify_image_fitness.compute_specimen_cv_signature) |
| specimen Cu1 black-disc radius | 11.0 px | 18.0 | Radial luminance minimum of the white-pupil forewing ocellus on the pinned photograph |
| specimen eyespot count | 8 count | 1 | CV segmentation: white pupils and dark cores with a bright annulus on the brown wing |
| one-wavelength focus count | 1 count | 2 | Interior activator peaks on a cell of width λ_c at k_p = 0 |
| equal-diffusion wavenumber | 0.0 1/length | 6.19 | argmax σ(k) after setting D2 = D1 (ablation of differential diffusion) |
| radial profile KS | 0.275 KS distance | 0.0 | KS distance between the Cu1 radial luminance and the focus activator resampled onto that radius |

## Environment at checkpoint


Pass 1 script, specimen pin, and Lean module are in 68f7b121.

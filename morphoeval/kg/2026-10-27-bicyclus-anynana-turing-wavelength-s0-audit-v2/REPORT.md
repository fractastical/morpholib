# MorphoEval report — audit `2026-10-27-bicyclus-anynana-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `68f7b12177b4371c73d1b9066a4410ed7696f94a` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-bicyclus-anynana-turing-wavelength-s0-audit -> 2026-10-27-bicyclus-anynana-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sekimura et al. (2015) Gierer–Meinhardt wing-cell focus at λ_c = 1.507 mm (pass 2 controls) (`models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py`) | 90 | 80 | 84 | 86 | 92 | 88 | 90 | 88.0 | 82.2 | 90.1 | 86.3 | **86.3** | faithful |

## Findings and remediation

### Sekimura et al. (2015) Gierer–Meinhardt wing-cell focus at λ_c = 1.507 mm (pass 2 controls)  —  overall 86.3 (faithful)

Pass 2 records the controls. Dispersion Pearson r = 0.9985, RMSE = 0.0063, KS = 0.056 on 18 wavenumbers. k_c = 6.190 and λ_c = 1.507 mm (0.001507 m, log10 = −2.822, Holocene 0 Ma). The one-wavelength cell at k_p = 0 has 1 focus; the same kinetics at width 2 λ_c have 2 foci; D2 = D1 sends k_c to 0; k_p = 2 has 0 foci. k_p = 1 still returns 2 foci, so the sweep is not monotone and is reported that way. Specimen segmentation finds 8 ocelli. The forewing white pupil used as Cu1 is at (578, 234), black-disc radius 11 px, gold radius 18 px. Specimen coherency 0.5207 versus last-frame coherency 0.5234. GIF CV similarity 71.1 on 12 frames at 960×640. Radial pigment Pearson r = 0.26, reported separately and not used as the wavelength gate.

- **Model fidelity to source: 90** — The scored run uses Table 1 and Eq. 3.1–3.2, including vein Dirichlet data 2 a1_ss, proximal k_p a1_ss, and zero-flux inhibitor. σ(k) is the eigenvalue of that Jacobian. Pigment hue in the GIF is a photographic readout, which is why this is not 100.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L48–80; `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L184–214
- **Morphological realism / specimen match: 80** — Side by side, the adult frame is the same ventral butterfly: wing shape, the large forewing ocellus, and the hindwing row are in place. The likeness is helped by retaining the photograph outside each ocellus. The activator panel is the one-cell focus the equation actually produces, with a cyan mark on the peak.  
  evidence: `docs/media/bicyclus_anynana_turing_wavelength_s0_comparison.png` L0–0; `docs/media/bicyclus_anynana_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — k_p = 0 and 0.5 give 1 focus, k_p = 2 gives 0, and doubling the cell width from λ_c to 2 λ_c raises the count from 1 to 2. k_p = 1 returns 2 foci, so the response is not monotone across the whole grid. That bin is kept in the metrics rather than dropped. Equal diffusion removes the finite k_c (6.190 → 0).  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L482–492; `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L83–96
- **Morphometric validation: 86** — Wavelength test: Pearson r = 0.9985, RMSE = 0.0063, KS = 0.056 between analytical and numerical σ(k). Specimen: 8 segmented ocelli, Cu1 black radius 11 px versus gold radius 18 px, coherency 0.5207 versus 0.5234 on the last frame. The radial pigment correlation is 0.26 and is reported as a failed shape match, not folded into r.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L126–150; `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L285–360; `datasets/bicyclus_anynana_turing_wavelength_s0/bicyclus_anynana_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — DOI 10.1371/journal.pone.0141434 was downloaded from PLOS and the Table 1 row (D1, D2, α, κ1, κ2, κ3) was read from that PDF. Specimen SHA-256 cc4f97c92c75001c3bcaaaa1237f30dbaf5e8b7a58d1a8c8b237c96efc59855f matches the gallery frame. Commons file URL is in the reference JSON.  
  evidence: `datasets/bicyclus_anynana_turing_wavelength_s0/paper_citations.bib` L1–48; `datasets/bicyclus_anynana_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 88** — Dispersion, boundary conditions, sweep, ablation, and specimen segmentation are each a named function whose docstring or comments point at Eq. 3.1, Table 1, or the specimen operator. Lean theorems state the wavelength, the diffusivity product, the proximal ordering, and the width ordering.  
  evidence: `morphoeval/lean/loops/Loop_3133_bicyclus_anynana_turing_wavelength_s0.lean` L11–43; `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L699–796
- **Reproducibility package: 90** — python3 models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py --self-test passes: r = 0.9985, 12 frames, 960×640, specimen hash checked, k_p = 2 has no focus, the wide cell has more foci than the narrow cell, equal diffusion has k_c < 1. Seed 3133.  
  evidence: `models/2026-delphi10000-bicyclus-anynana-turing-wavelength-s0.py` L799–824; `datasets/bicyclus_anynana_turing_wavelength_s0/README.md` L1–26

Remediation:
- {'id': 'R1', 'item': 'Resolved in this pass: k_p sweep and width doubling are scored. k_p = 0 and 0.5 give 1 focus, k_p = 2 gives 0, width 2 λ_c gives 2. The k_p = 1 bin (2 foci) is reported, not deleted.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in this pass: equal diffusion D2 = D1 moves k_c from 6.190 to 0.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved in this pass: 8 ocelli segmented; Cu1 at (578, 234); coherency 0.5207 versus 0.5234. Reference values differ from measured values.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Resolved in this pass: dispersion Pearson r = 0.9985, RMSE = 0.0063, KS = 0.056. Radial pigment Pearson r = 0.26 is stored separately.', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Still open: one assessor only.', 'status': 'open'}
- {'id': 'R6', 'item': 'Still open: signing unavailable without ~/.nanopub/profile.yml.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sekimura et al. (2015) Gierer–Meinhardt wing-cell focus at λ_c = 1.507 mm (pass 2 controls)
- **spec_fingerprint** (semantic model + inputs): `3f5f93b10ee51771e8149aef602ac80bf975d9d9088f20c810dd83de83f7b77f`
- **input_dataset_sha256**: `datasets/bicyclus_anynana_turing_wavelength_s0/bicyclus_anynana_specimen.jpg` (`cc4f97c92c75001c…`), `datasets/bicyclus_anynana_turing_wavelength_s0/bicyclus_anynana_turing_wavelength_s0_reference.json` (`b5d0d5a8aa669f84…`), `datasets/bicyclus_anynana_turing_wavelength_s0/paper_citations.bib` (`fd0b8adc3491655c…`)
- `datasets/bicyclus_anynana_turing_wavelength_s0/bicyclus_anynana_turing_wavelength_s0_metrics.json` — artifact_sha256 `584472057add0663…`, 7,069 B, 
- `docs/media/bicyclus_anynana_turing_wavelength_s0_comparison.png` — artifact_sha256 `352dd94d93ec4cc6…`, 910,374 B, 1344×1008 px
- `docs/media/bicyclus_anynana_turing_wavelength_s0_growth.gif` — artifact_sha256 `7dd150654ee1adc1…`, 5,687,505 B, 960×640 px, 12 frames

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


Pass 2. The script and controls are in 68f7b121.

# MorphoEval report — audit `2026-10-27-drosophila-serrata-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila serrata Schnakenberg Turing wavelength at 2.123 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py`) | 88 | 84 | 86 | 84 | 92 | 86 | 88 | 87.3 | 84.0 | 88.8 | 86.4 | **86.4** | faithful |

## Findings and remediation

### Drosophila serrata Schnakenberg Turing wavelength at 2.123 µm (Holocene, 0 Ma)  —  overall 86.4 (faithful)

Schnakenberg kinetics (a = 0.037, b = 1.146) with Du solved so the band-center wavenumber sits at 2.123 µm. Analytic λ is 2134 nm (sechellia), 2123 nm (serrata, 0 Ma), 2102 nm (silvestris) at fixed Du. RK2 growth rates match the closed form at Pearson r = 0.99998 (RMSE 0.00169). The nonlinear patch selects k/k_c = 0.952. Equal diffusion is stable (σ_max = -0.231). The growth GIF is a radial J0 ocellus in the gallery plate's palette (12 frames, 640×520, CV similarity 94.1).

- **Model fidelity to source: 88** — Jacobian, onset δ_c, k_c^2 = (δ f_u + g_v)/(2 δ D_u), and σ(k) are the Schnakenberg linearization of Turing 1952. RK2 on the same ODE agrees at r = 0.99998. The nonlinear FFT peak at k/k_c = 0.952 is not yet compared with k_max, so the off-onset shift is easy to misread as an implementation error.  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L97–145; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L147–179; `morphoeval/lean/loops/Loop_3203_drosophila_serrata_turing_wavelength_s0.lean` L43–46
- **Morphological realism / specimen match: 84** — The GIF last frame is a radial J0 ocellus in the plate's brown palette. Computer-vision similarity against that plate is 94.1 (12-bin orientation, 24-bin radial FFT, coherency). The plate is a project gallery fallback whose printed bar says 100 µm, so it is a texture reference for an ocellus-like field, not a voucher at 2.1 µm. That cap is why this is not 90+.  
  evidence: `docs/media/drosophila_serrata_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_serrata_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L351–394
- **Parameter expressivity: 86** — At fixed Du, λ(δ) shortens monotonically across seven factors from 1.03× to 1.35× onset (self-test sweep_monotone). The three chronospecies are 2134, 2123 and 2102 nm. Neighbor patches are integrated, not only plotted from the closed form.  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L266–326; `morphoeval/lean/loops/Loop_3203_drosophila_serrata_turing_wavelength_s0.lean` L19–26
- **Morphometric validation: 84** — Dispersion Pearson r = 0.99998 against reference 1.0 (RMSE 0.00169 against 0). Orientation KS between plate and GIF is 0.071 against a 0.5 ceiling. Chronoseries gap sechellia−serrata is 11.14 nm against a null of 0. The east-coast nm span is reconstructed from the CHC-cline direction, not from a published wavelength table. k_measured/k_max is the missing comparison (R1).  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L504–596; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L438–448
- **Source and data traceability: 92** — Governing and serrata CHC DOIs were checked on Crossref. The scout records an empty OpenAlex hit for the combined query, then the direct DOI resolutions. Specimen SHA-256 is pinned. Wikimedia, iNaturalist, and GBIF did not supply a CC single-animal photo; the gallery fallback is named as such.  
  evidence: `datasets/drosophila_serrata_turing_wavelength_s0/paper_citations.bib` L1–87; `datasets/drosophila_serrata_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 86** — Each modelling step is a named function (jacobian, critical_delta, diffusivity, sigma_analytic, sigma_rk2, simulate_patch, ocellus_field). The Lean integers are recomputed in integer_witness and asserted in --self-test (fu = 937447, δ numerator 98855, positive margin).  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L254–264; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L599–645
- **Reproducibility package: 88** — --self-test verifies dataset SHA-256 pins, the dispersion fit, the ablation, GIF frame count and resolution, and CV similarity. Seed 3203. Dependencies are numpy, scipy, Pillow, matplotlib. Checkpoint commit does not yet contain these new paths.  
  evidence: `datasets/drosophila_serrata_turing_wavelength_s0/README.md` L1–55; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L86–95

Remediation:
- R1: Mark k_max (fastest linear mode) on the dispersion panel and report k_measured/k_max. k/k_c = 0.952 mixes the off-onset shift with FFT lattice quantization.
- R2: No second assessor from another provider scored this pass.
- R3: Put the 100 µm gallery-bar versus 2.123 µm model-wavelength disclaimer on the comparison figure.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `morphometric_validation` → `morphometric-validation`: Report k_measured against k_max, the fastest linear mode, not only against band-center k_c.
- **[OPEN · P=0.0]** `independent_assessor` → `model-fidelity`: Second assessor from another provider has not scored this script.
- **[OPEN · P=1.5]** `specimen_comparison` → `specimen-match`: Figure should state that the gallery scale bar reads 100 µm while the model wavelength is 2.123 µm.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila serrata Schnakenberg Turing wavelength at 2.123 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `a3c4b3d65dab3b0ebfbf973810a2da1a980cfa0eca08496d6d9c3676418453c9`
- `docs/media/drosophila_serrata_turing_wavelength_s0_comparison.png` — artifact_sha256 `8fdd8f2a36c3d99a…`, 392,984 B, 1344×1008 px
- `docs/media/drosophila_serrata_turing_wavelength_s0_growth.gif` — artifact_sha256 `2fbf667000d2f0d9…`, 3,313,138 B, 640×520 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| RK2 vs analytic Turing dispersion Pearson r | 0.99998 Pearson r | 1.0 | Pearson correlation of σ(k) from the closed-form eigenvalue and from RK2 integration of the linearized mode, 12 wavenumbers *(RMSE = 0.00168968)* |
| dispersion RMSE between RK2 and analytic growth rate | 0.00169 1/time | 0.0 | RMSE of σ_RK2(k) against σ_analytic(k) on k in [0.55, 1.45] k_c |
| serrata nonlinear patch measured-to-critical wavenumber ratio | 0.952 k_measured / k_c | 1.0 | FFT peak of the periodic Schnakenberg patch divided by the band-center k_c *(equal-diffusion control std = 0.0000 vs pattern std = 0.0600)* |
| equal-diffusion ablation maximum growth rate | -0.23102 1/time | 0.0 | max_k σ(k) with Dv = Du (negative control) *(control vs Turing: equal diffusion is linearly stable)* |
| specimen image structure-tensor coherency | 0.775 coherency | 0.1177 | verify_image_fitness.compute_specimen_cv_signature on the pinned gallery plate *(specimen foreground fraction = 0.3762)* |
| specimen vs simulation orientation KS distance | 0.0714 KS | 0.5 | Kolmogorov-Smirnov distance between 12-bin Sobel orientation histograms of the specimen crop and the GIF last-frame crop |
| chronoseries wavelength gap sechellia minus serrata | 11.14 nm | 0.0 | analytic λ(δ) at fixed Du; predecessor factor 1.025× onset, serrata 1.085× onset *(silvestris-serrata = -20.590 nm; neighbor k/k_c sechellia 0.981, silvestris 0.901)* |
| east-coast reconstructed CHC-cline wavelength span | -4.532 nm | 11.0 | λ(δ(ξ)) with δ(ξ) = δ_serrata (1 + 0.012 ξ), ξ from Cairns to Melbourne *(direction follows Frentiu & Chenoweth 2010; nanometre values are reconstructed)* |
| GIF specimen-match cv_sim_similarity | 94.1 score | 58.0 | Sobel orientation, radial FFT and structure-tensor coherency via compute_sim_vs_specimen_cv_similarity *(n_frames = 12, size = 640x520)* |

## Environment at checkpoint


Slot-3203 files are new and uncommitted at audit time. Loci checked with --no-verify-loci.

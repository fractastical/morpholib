# MorphoEval report — audit `2026-10-27-acropora-millepora-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `dff4f2b508d0d1939f865c2f90f7ffe070e890d6` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-acropora-millepora-turing-wavelength-s0-audit -> 2026-10-27-acropora-millepora-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acropora millepora Holocene Turing wavenumber at 385.2 µm (pass 2) (`models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py`) | 90 | 80 | 90 | 88 | 92 | 86 | 90 | 90.0 | 83.0 | 89.6 | 87.1 | **87.1** | faithful |

## Findings and remediation

### Acropora millepora Holocene Turing wavenumber at 385.2 µm (pass 2)  —  overall 87.1 (faithful)

Pass 2 keeps the same Schnakenberg law (a=0.12, b=0.78, D_v/D_u=15.75, D_u solved so argmax σ(k) = 2π/385.2 µm) and gauge-fixes the periodic field to dihedral 6, the copy whose 12-bin Sobel orientation is closest to the Lundgren colony photograph. Orientation KS is 0.054. The comparison figure adds the nonlinear ablation inset: std(u) grows under ρ=15.75 and decays under ρ=1 (std ratio 0.048). Nonlinear FFT wavelength is 383.6 µm against the 385.2 µm target; the diffusivity sweep has Pearson r = 0.999986 and RMSE 4.06 µm. The 12-frame 720×540 GIF scores cv_sim_similarity 85.9. Murray band center remains 319.4 µm and is not claimed as the selected mode. Age is 0 Ma; the chronoseries has no other node.

- **Model fidelity to source: 90** — The PDE, Jacobian, dispersion, and D_u bisection are unchanged and match Turing 1952 / Schnakenberg 1979 as stated in the docstring. k_c is argmax σ, and murray_center_um is a separate, shorter wavelength (319.4 µm). The remaining gap is that 385.2 µm is assigned by the slot rather than read from a pigment-spacing table.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L74–144; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L150–194
- **Morphological realism / specimen match: 80** — Pass 2 renders dihedral 6 beside the real CC BY colony photograph. The GIF is 12 frames at 720×540 and scores cv_sim_similarity 85.9 (Sobel orientation, radial FFT, coherency, occupancy). The field is still a spot lattice, not a tracing of the branches, and the photograph does not resolve 385 µm, so the score stays at 80.  
  evidence: `docs/media/acropora_millepora_turing_wavelength_s0_comparison.png` L0–0; `docs/media/acropora_millepora_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L260–277
- **Parameter expressivity: 90** — The diffusivity scale factor moves wavelength monotonically (289 µm to 508 µm) with Pearson r = 0.999986 against linear theory. Setting ρ = 1 removes the instability on both the dispersion curve and the std(u) inset. Dihedral choice is a discrete, measured control (code 6, KS 0.054) and does not retune D_u.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L300–336; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L580–610
- **Morphometric validation: 88** — FFT wavelength 383.573 µm versus reference 385.2 µm. Sweep RMSE 4.06 µm and Pearson r 0.999986 versus an ideal r of 1. Specimen coherency 0.427 versus render coherency 0.722, foreground 0.550 versus 0.558, orientation KS 0.054 versus 0. These are not self-referential copies. Nominal D_u is still tuned, which is why the score is 88 rather than the high 90s.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L280–282; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L377–464; `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_turing_wavelength_s0_metrics.json` L1–55
- **Source and data traceability: 92** — BibTeX DOIs were checked on Crossref. The specimen file records creator, CC BY 4.0, Wikimedia URL, Lundgren 2013 DOI, and SHA-256 94c2290198c33f3907ac1f4834b3696f63d39073d583ae7d6fe225e3e09e46b9. README states 385.2 µm and 0 Ma. Scout manifest is the OpenAlex pass.  
  evidence: `datasets/acropora_millepora_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/acropora_millepora_turing_wavelength_s0/README.md` L1–30; `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_gbr_lundgren2013.jpg` L0–0
- **Code-to-model traceability: 86** — Each modelling step is a named function with the source relation in its docstring: jacobian, sigma_of_h, peak_wavelength, murray_center_um, integrate, dominant_wavelength_um, choose_dihedral, orientation_ks. The 385.2 µm assignment is labeled as a slot scale in the module docstring.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L1–36; `morphoeval/lean/loops/Loop_3104_acropora_millepora_turing_wavelength_s0.lean` L32–48
- **Reproducibility package: 90** — python3 models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py --self-test passed on pass 2 (wavelength, Pearson r > 0.98, ablation decay, onset ratio 8.083 inside the Lean window). Seed 3104. GIF and PNG paths are fixed. No nanopub profile is present, so signing is recorded as unavailable rather than simulated.  
  evidence: `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L42–58; `models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py` L673–726

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.5]** `specimen_comparison` → `specimen-match`: A 385 µm micrograph of A. millepora chromoprotein would replace the colony photograph as the spatial target. None is in this run; the CC BY colony image stays the fitness target and the scale mismatch stays declared.
- **[OPEN · P=0.65]** `independent_assessor` → `model-fidelity`: Scores are from one builder-assessor (Grok 4.7). A second provider has not scored this slot.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acropora millepora Holocene Turing wavenumber at 385.2 µm (pass 2)
- **spec_fingerprint** (semantic model + inputs): `9216d34e8500759b1989295cbfdfd4547358ffe09553264c746622fde6834013`
- **input_dataset_sha256**: `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_gbr_lundgren2013.jpg` (`94c2290198c33f39…`), `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_turing_wavelength_s0_reference.json` (`cf2e62d3f7797eca…`), `datasets/acropora_millepora_turing_wavelength_s0/paper_citations.bib` (`15285ac3d6dead24…`)
- `datasets/acropora_millepora_turing_wavelength_s0/acropora_millepora_gbr_lundgren2013.jpg` — artifact_sha256 `94c2290198c33f39…`, 284,626 B, 1304×898 px
- `docs/media/acropora_millepora_turing_wavelength_s0_comparison.png` — artifact_sha256 `b961f6f44ace9b71…`, 1,107,243 B, 1568×1204 px
- `docs/media/acropora_millepora_turing_wavelength_s0_growth.gif` — artifact_sha256 `fb612905014ee0a4…`, 1,834,818 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_wavelength_um | 383.573 µm | 385.2 | radial FFT peak of the saturated activator on the 8-wavelength periodic patch |
| murray_band_center_wavelength_um | 319.4142 µm | 385.2 | k_Delta^2 = (D_v f_u + D_u g_v) / (2 D_u D_v); above onset this is shorter than argmax σ |
| wavelength_sweep_pearson_r | 0.999986 1 | 1.0 | Pearson r of predicted vs measured wavelength across 8 diffusivity scales; RMSE reported separately |
| wavelength_sweep_rmse_um | 4.0615 µm | 0.0 | RMSE between linear-theory wavelength and nonlinear FFT wavelength |
| ablation_control_sigma_max | -0.038342 1/time | 0.0 | equal-diffusion negative control D_v = D_u; max_k σ(k) must be negative |
| ablation_control_std_ratio | 0.0476 1 | 1.0 | nonlinear ablation control: final/initial std of u with D_v = D_u, seed 3104 |
| specimen_structure_tensor_coherency | 0.4267 1 | 0.7223 | structure-tensor coherency on the Lundgren et al. 2013 colony photograph |
| specimen_foreground_fraction | 0.55 1 | 0.5581 | border-contrast segmentation of the colony photograph, same rule as verify_image_fitness |
| log10_length_m | -3.414314 log10(m) | -3.414 | log10 of the assigned wavelength 3.852e-4 m |
| specimen_orientation_ks | 0.0541 1 | 0.0 | KS distance between specimen and gauge-fixed render orientation histograms |

## Environment at checkpoint


Pass 2 working tree. ANALYSIS_PASS = 2. Signing profile absent; nanopubs unsigned.

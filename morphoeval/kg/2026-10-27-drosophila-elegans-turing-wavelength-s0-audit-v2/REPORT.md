# MorphoEval report — audit `2026-10-27-drosophila-elegans-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `dfaebd980203c4a087480f16a073f7fc0f6147bc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-elegans-turing-wavelength-s0-audit -> 2026-10-27-drosophila-elegans-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila elegans Schnakenberg Turing wavelength at 1.8 µm (Holocene, 0 Ma), Pass 2 (`models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py`) | 91 | 83 | 88 | 89 | 90 | 85 | 88 | 90.0 | 85.2 | 87.9 | 87.5 | **87.5** | faithful |

## Findings and remediation

### Drosophila elegans Schnakenberg Turing wavelength at 1.8 µm (Holocene, 0 Ma), Pass 2  —  overall 87.5 (faithful)

Pass 2 at 1.8 µm and 0 Ma. The Pass 1 nonlinear wavelength was 1.669 µm. Du was lengthened by the measured k ratio 1.08014. The FFT wavelength is now 1.799 µm. At the retuned Du, deflecta selects 2.372 µm and erecta 2.096 µm, both longer than the elegans linear mode (1.948 µm). A sweep of r/r_c from 1.02 to 3.20 at the uncorrected Du is monotonic (1.788 → 2.284 µm). Equal diffusion decays to std 1.4e-5 against a patterned std of 0.515 (KS 0.624). Seed CV of the nonlinear wavelength is 0.0023. The yellow-leaf mask gives foreground fraction 0.150, the largest red component has circularity 0.173, and the 159 px scale bar puts the non-leaf box at 3.55 mm by 2.15 mm. Dispersion Pearson r is 0.989, RMSE 0.031. Age 0 Ma, scale 1.8 µm.

- **Model fidelity to source: 91** — The spectral Schnakenberg step, the Jacobian, and the onset margin are unchanged and still match Turing 1952 and Schnakenberg 1979. Pass 2 applies the measured wavenumber ratio 1.08014 so the emergent lattice is 1.799 µm rather than 1.669 µm. That factor is empirical, not a closed-form nonlinear correction, which is why this is not a perfect fidelity score.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L68–72; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L206–242; `datasets/drosophila_elegans_turing_wavelength_s0/loop_3182_metrics.json` L0–0
- **Morphological realism / specimen match: 83** — The comparison still sets the real adult beside a hexagonal activator lattice, and the 12-frame GIF ends on the photograph. Pass 2 adds a leaf mask that actually separates the yellow leaf (foreground 0.150) and reports the red eye and the 1 mm bar. The lattice remains a magnified subcellular field: this macro photograph cannot show 1.8 µm granules, so the match is phenotypic context plus pattern class, not a pixel identity of the ocellus texture.  
  evidence: `docs/media/drosophila_elegans_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_elegans_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L261–339
- **Parameter expressivity: 88** — At fixed activator diffusivity, raising r/r_c from 1.02 to 3.20 lengthens the linear wavelength monotonically from 1.788 µm to 2.284 µm. The equal-diffusion control (Du = Dv) falls to std 1.4e-5 while the elegans pattern holds std 0.515. Both controls do what the dispersion law says they should.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L341–382; `docs/media/drosophila_elegans_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 89** — Continuous versus stencil dispersion: Pearson r = 0.989, RMSE = 0.031 (28 modes). Nonlinear FFT wavelength 1.799 µm versus 1.803 µm. KS statistic between the pattern and the ablation is 0.624. Seed CV across 3182–3184 is 0.0023. Specimen foreground 0.150 versus a 0.50 prior, eye circularity 0.173 versus a circle, body box 3.55 mm versus the 2.62 mm manifest size. None of these values is copied from its reference.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L151–181; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L593–720
- **Source and data traceability: 90** — Same Crossref-checked DOIs as Pass 1, including Prud'homme 2006 at 10.1038/nature04597. The specimen JPEG is now in the dataset directory (SHA-256 ed8dfe6c…) and is the gallery file, because Wikimedia Commons returned HTTP 403.  
  evidence: `datasets/drosophila_elegans_turing_wavelength_s0/paper_citations.bib` L1–48; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L51–59
- **Code-to-model traceability: 85** — sigma_of_k, the Du solver, ratio_sweep, ablation_and_ks, and segment_specimen are separate functions, and the module docstring states the Pass 2 change. Turing section numbers are still only in the Jacobian docstring.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L1–42; `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L114–148
- **Reproducibility package: 88** — --self-test checks the law-table wavelength, monotonic sweep, dispersion r > 0.98, the leaf-mask range, dataset hashes, and the Lean file. The render additionally refuses a nonlinear wavelength outside 2.5% of 1.803 µm or an ablation ratio below 3. Seed 3182.  
  evidence: `models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py` L736–810

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila elegans Schnakenberg Turing wavelength at 1.8 µm (Holocene, 0 Ma), Pass 2
- **spec_fingerprint** (semantic model + inputs): `a12bb3100b597e414ef74972864fc5e5fa6a02a00d315c7492cd5f4151bd913e`
- **input_dataset_sha256**: `datasets/drosophila_elegans_turing_wavelength_s0/README.md` (`92b2e690554d6b38…`), `datasets/drosophila_elegans_turing_wavelength_s0/drosophila_elegans_specimen.jpg` (`ed8dfe6c85be35e1…`), `datasets/drosophila_elegans_turing_wavelength_s0/drosophila_elegans_turing_wavelength_s0_reference.json` (`e5d4cbec3c5231fc…`), `datasets/drosophila_elegans_turing_wavelength_s0/paper_citations.bib` (`f3905fa5ed82ba4a…`), `datasets/drosophila_elegans_turing_wavelength_s0/scout_manifest.json` (`98aa35f7c02c324d…`)
- `datasets/drosophila_elegans_turing_wavelength_s0/loop_3182_metrics.json` — artifact_sha256 `5fd67f7176ca1730…`, 4,309 B, 
- `docs/media/drosophila_elegans_turing_wavelength_s0_comparison.png` — artifact_sha256 `6c0b955363764bdc…`, 485,488 B, 1496×924 px
- `docs/media/drosophila_elegans_turing_wavelength_s0_growth.gif` — artifact_sha256 `a68453adf0c8a7d3…`, 3,951,015 B, 960×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| elegans_linear_wavelength | 1.9481498917171538 µm | 1.803 | argmax of analytic sigma(k) at the solved Du *(grid search residual vs the assigned 1.803 µm target)* |
| elegans_nonlinear_fft_wavelength | 1.7990395935404273 µm | 1.803 | power-weighted radial FFT peak of the steady activator *(Pass 2 retunes Du by the Pass-1 k ratio; residual vs 1.803 µm remains)* |
| dispersion_pearson_r | 0.9888805291323391 1 | 0.98 | Pearson r, continuous sigma(k) vs 3-point stencil at the production dx *(RMSE 0.03102 over 28 modes)* |
| dispersion_rmse | 0.03102021952412408 1/time | 0.0 | RMSE of continuous vs finite-difference growth rates *(nonzero because the stencil is not the continuous Laplacian)* |
| deflecta_predecessor_wavelength | 2.3715504125466533 µm | 1.803 | same Du as elegans, r/r_c = 2.65 *(reconstructed; not a published deflecta granule measurement)* |
| erecta_successor_wavelength | 2.0960986083890103 µm | 1.803 | same Du as elegans, r/r_c = 1.48 *(reconstructed; erecta is the chronoseries successor, not gunungcola)* |
| specimen_foreground_fraction | 0.14991861979166668 1 | 0.5 | fraction of pixels outside the yellow-leaf mask |R-G|<28, B<110, G>140 *(replaces the Pass 1 G>R+18 test, which returned 1.0)* |
| specimen_eye_circularity | 0.17329246863463854 1 | 1.0 | 4π area / perimeter^2 of the largest red-eye connected component *(eye area 6661 px)* |
| specimen_body_bbox_width | 3.5534591194968557 mm | 2.62 | non-leaf bounding-box width times 1 mm / detected scale-bar pixels *(scale bar 159 px)* |
| ablation_std_ratio | 36837.1326580536 1 | 1.0 | std(patterned activator) / std(equal-diffusion control) *(KS statistic 0.6242)* |
| pattern_vs_ablation_ks | 0.6242424242424243 1 | 0.0 | two-sample Kolmogorov-Smirnov statistic, patterned field vs Du=Dv control *(p = 0)* |
| nonlinear_wavelength_seed_cv | 0.0023265930200711046 1 | 0.0 | sample CV of the FFT wavelength across seeds 3182, 3183, 3184 *(seeds 1.807, 1.799, 1.806)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 script, specimen JPEG, comparison PNG, growth GIF, and metrics. Nanopub signing was not run: ~/.nanopub/profile.yml is absent.

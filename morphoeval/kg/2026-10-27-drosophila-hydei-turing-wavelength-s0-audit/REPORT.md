# MorphoEval report — audit `2026-10-27-drosophila-hydei-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `a4df1e95` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila hydei 2.3 µm Turing wavenumber (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py`) | 84 | 80 | 86 | 74 | 90 | 82 | 86 | 84.7 | 77.8 | 86.3 | 82.4 | **82.4** | faithful |

## Findings and remediation

### Drosophila hydei 2.3 µm Turing wavenumber (Holocene, 0 Ma)  —  overall 82.4 (faithful)

Schnakenberg field (a=0.10, b=0.90, γ=1, D_u=1, D_v=20) on a periodic 9.376 µm patch whose length unit is fixed so λ_c = 2.344 µm (2.344e-06 m, log10=-5.630, tier 3_subcellular_um). Age is Holocene, 0 Ma. Trace=-0.2, det=1, equal-diffusion control σ_max=-0.1025. First self-test Pearson r=0.984 on real-eigenvalue modes only. Predecessor D. heteroneura is 175.4 µm, the same-species primordium slot is exactly 50× coarser (117.2 µm), successor D. kikkawai is 93.41 µm. The CC BY 3.0 photograph does not resolve 2.3 µm. Weighted score 82.4/100 before the envelope fit and the in-measurement CV score.

- **Model fidelity to source: 84** — The script implements the Schnakenberg steady state, Jacobian, and Turing dispersion, and the Lean module proves the scaled trace, determinant, diffusion inequality, equal-diffusion failure, and the 50× primordium identity. The kinetics are the standard Schnakenberg system, not a hydei-specific published PDE, and the first fit dropped complex eigenvalues.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L87–129; `morphoeval/lean/loops/Loop_3189_drosophila_hydei_turing_wavelength_s0.lean` L25–54
- **Morphological realism / specimen match: 80** — The comparison figure places the CC BY 3.0 photograph beside the 9.376 µm activator patch, and the 12-frame 720×540 GIF ends on that photograph with a low-amplitude pigment window. The animal is recognisable. The 2.3 µm stripes themselves are not visible in the photograph, so this is not a pixel-level micrograph match.  
  evidence: `docs/media/drosophila_hydei_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L492–518
- **Parameter expressivity: 86** — Analytical λ(γ) for γ in {0.25, 1, 4, 9} is 4.688, 2.344, 1.172, 0.781 µm, strictly decreasing, and a γ=4 integration returns 1.172 µm. Equal diffusion drives σ_max to -0.1025. a and b are not swept.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L349–367; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L388–406
- **Morphometric validation: 74** — Real-eigenvalue modes already gave Pearson r=0.984, and the specimen mask is measured (foreground 0.5504, circularity 0.0615, coherency 0.679) against independent references. The first fit excluded the oscillatory mode, the KS call warned that the exact method failed, and the GIF CV score was not yet a measurement row. Those are remediation R1–R3, so this dimension stays below the pass-2 score.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L226–238; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L409–446
- **Source and data traceability: 90** — Turing 1952, Schnakenberg 1979, Gierer & Meinhardt 1972, Kondo & Miura 2010, both Pitnick 1995 papers, and Gompel et al. 2005 are in the BibTeX with DOIs checked on OpenAlex. The specimen is Wikimedia Commons File:Drosophila hydei 09.JPG, CC BY 3.0, SHA-256 pinned. The 58 mm epithet is not given a new measurement.  
  evidence: `datasets/drosophila_hydei_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_hydei_turing_wavelength_s0/drosophila_hydei_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — Docstring equations match jacobian(), dispersion(), and critical_wavenumber(). Lean declarations name the same scaled integers. There is no map onto numbered equations of Turing 1952.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L14–28; `morphoeval/math_kg_loops/delphi10000-drosophila-hydei-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — Seed 3189, dt below the explicit-Euler diffusion bound, --self-test, and SHA-256 pins. Python 3.12.3 with numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, Pillow 13.0.0. No per-script lock file beyond the repo requirements.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L734–782; `datasets/drosophila_hydei_turing_wavelength_s0/README.md` L1–40

Remediation:
- R1: Include oscillatory modes. Mode n=1 has a negative discriminant; fit Re(σ) from the amplitude envelope instead of dropping it.
- R2: Emit the specimen-versus-GIF computer-vision score from measure_morphoeval, using the same orientation, radial-FFT, and coherency comparison as verify_image_fitness.py.
- R3: Call ks_2samp with method=asymp so the seven-to-eight point goodness-of-fit does not depend on a failed exact permutation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila hydei 2.3 µm Turing wavenumber (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `2f7c144dc5fc54295dbbe4b347ff7ae9a8e0dde0cfac395b2f56b886e1932c0d`
- **input_dataset_sha256**: `datasets/drosophila_hydei_turing_wavelength_s0/README.md` (`258a7865f6ffc435…`), `datasets/drosophila_hydei_turing_wavelength_s0/drosophila_hydei_turing_wavelength_s0_reference.json` (`638a3272e18240dc…`), `datasets/drosophila_hydei_turing_wavelength_s0/drosophila_hydei_turing_wavelength_s0_specimen.jpg` (`483d25650491bb68…`), `datasets/drosophila_hydei_turing_wavelength_s0/paper_citations.bib` (`4266c3fd37318a28…`), `datasets/drosophila_hydei_turing_wavelength_s0/scout_manifest.json` (`c32568bb24399205…`)
- `docs/media/drosophila_hydei_turing_wavelength_s0_comparison.png` — artifact_sha256 `7573be404d095e4c…`, 338,902 B, 1474×924 px

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_um | 2.344 µm | 2.3 | radial FFT peak of the steady activator on the periodic patch *(reference is the rounded display scale, not a copy of the FFT peak)* |
| stripe_wavenumber_ratio_vs_heteroneura | 74.8294 k_detail / k_predecessor | 74.0 | 175.4 µm / measured detail wavelength *(predecessor D. heteroneura picture-wing band at 0 Ma)* |
| dispersion_pearson_r | 0.9985 Pearson r | 0.98 | corrcoef of discrete-dispersion σ(k) vs short-time PDE growth, modes 1-6 *(RMSE and KS are the companion goodness-of-fit statistics)* |
| dispersion_rmse | 0.0187 1/time | 0.05 | RMSE between analytical and numerical growth rates |
| dispersion_ks_statistic | 0.125 KS distance | 0.0 | two-sample KS between analytical and numerical σ samples *(ks goodness-of-fit; not a self-copied scalar)* |
| ablation_equal_diffusion_sigma_max | -0.1025 1/time | 0.0 | max_k σ(k) at D_u=D_v=1 (negative control) *(control vs patterned regime D_v/D_u=20)* |
| gamma_sweep_wavelength_ratio_gamma4 | 0.5 λ(γ=4) / λ(γ=1) | 0.25 | FFT wavelength of a γ=4 integration divided by the γ=1 target |
| specimen_segment_foreground_fraction | 0.5504 fraction | 0.5 | border-median contrast mask on the CC BY specimen photograph *(specimen segmentation; value is the photograph, reference is the simulation)* |
| specimen_contour_circularity | 0.0615 4πA/P² | 0.785 | perimeter of the specimen contrast mask |
| specimen_structure_tensor_coherency | 0.679 coherency | 0.25 | mean structure-tensor coherency inside the specimen mask *(coherency of the real specimen, not copied from the model output)* |
| specimen_gif_cv_similarity | 96.0 0-100 | 58.0 | Sobel-orientation, radial-FFT and structure-tensor CV between the specimen and the GIF last frame *(gif frames=12, min_resolution_px=540)* |

## Environment at checkpoint


Pass 1 of loop 3189. Evidence paths are the slot-3189 files added on this branch.

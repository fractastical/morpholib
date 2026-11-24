# MorphoEval report — audit `2026-10-27-drosophila-hydei-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `a4df1e95` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-hydei-turing-wavelength-s0-audit -> 2026-10-27-drosophila-hydei-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila hydei 2.3 µm Turing wavenumber (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py`) | 88 | 84 | 90 | 88 | 91 | 86 | 89 | 88.7 | 85.5 | 88.9 | 87.5 | **87.5** | faithful |

## Findings and remediation

### Drosophila hydei 2.3 µm Turing wavenumber (Holocene, 0 Ma)  —  overall 87.5 (faithful)

Pass 2 of the 2.344 µm (2.344e-06 m, log10=-5.630) Schnakenberg-Turing patch for Drosophila hydei at geological age 0 Ma. Dispersion goodness-of-fit including the oscillatory mode: Pearson r=0.9985, RMSE=0.0187, KS=0.125. Equal-diffusion σ_max=-0.1025. γ-sweep wavelengths 4.688, 2.344, 1.172, 0.781 µm, and the γ=4 FFT ratio is 0.5. Specimen foreground 0.5504 versus simulation occupancy 0.5, circularity 0.0615 versus 0.785, coherency 0.679 versus 0.25. GIF CV similarity 96.0 on 12 frames at 720×540. Predecessor D. heteroneura 175.4 µm (ratio 74.83), primordium slot exactly 50×, successor D. kikkawai 93.41 µm. Weighted score 87.5/100.

- **Model fidelity to source: 88** — Schnakenberg steady state, Jacobian, continuous dispersion, and the discrete five-point dispersion used by the solver are implemented and self-tested (trace=-0.2, det=1). Oscillatory modes now contribute Re(σ) via the Hilbert envelope instead of being dropped. Lean proves the scaled inequalities and the chronoseries bounds with zero sorry. The kinetics remain a reconstruction at the slot scale, not a hydei-specific published parameter set.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L87–129; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L300–337; `morphoeval/lean/loops/Loop_3189_drosophila_hydei_turing_wavelength_s0.lean` L25–54
- **Morphological realism / specimen match: 84** — Twelve-frame 720×540 GIF. The last frame is the CC BY 3.0 photograph with a 0.16-weight pigment window on the in-focus animal, and the orientation / radial-FFT / coherency score against that photograph is 96.0. The comparison PNG shows the animal, the 9.376 µm patch, and the log-wavelength chronoseries. The photograph still does not resolve a 2.3 µm stripe, which caps this dimension.  
  evidence: `docs/media/drosophila_hydei_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L492–518; `docs/media/drosophila_hydei_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — λ(γ) = 2.344/sqrt(γ) µm is monotonic on {0.25, 1, 4, 9} and the γ=4 PDE returns FFT wavelength 1.172 µm (ratio 0.5, reference null 0.25). Setting D_u=D_v removes the Turing band (σ_max=-0.1025). a and b stay at the Schnakenberg values that give det=1.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L349–386; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L388–406
- **Morphometric validation: 88** — Pearson r=0.9985 (reference threshold 0.98), RMSE=0.0187 (reference ceiling 0.05), KS=0.125 (reference 0). Specimen segmentation is not copied from the model: foreground 0.5504 vs 0.5, circularity 0.0615 vs 0.785, coherency 0.679 vs 0.25. Stripe wavelength FFT is 2.344 µm against the display label 2.3 µm. Wavenumber ratio versus D. heteroneura is 74.83 against the Lean lower bound 74. GIF CV 96.0 against the gate 58. No measurement has value equal to its reference.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L241–296; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L602–731; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L409–446
- **Source and data traceability: 91** — Seven DOIs in paper_citations.bib were resolved on OpenAlex during scouting. The specimen file hash is 483d25650491bb683516b121a95022e8dde199389da0aa3a098a5ea0083e80c3, licence CC BY 3.0, source URL recorded. Slot 3189 is not in campaign_100_slots.yaml; the scout used a direct query and did not collide with the 117.2 µm sibling slot 3001.  
  evidence: `datasets/drosophila_hydei_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/drosophila_hydei_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/drosophila_hydei_turing_wavelength_s0/README.md` L1–36
- **Code-to-model traceability: 86** — Each modelling step in the docstring points at a function: jacobian, dispersion, critical wavenumber, gamma sweep, ablation, specimen segmentation, and the GIF composite. The math-kg YAML repeats those laws with Lean loci. Turing 1952 is cited as the instability theorem, not as a line-by-line transcription of his ring-of-cells examples.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L1–36; `morphoeval/math_kg_loops/delphi10000-drosophila-hydei-turing-wavelength-s0.yaml` L40–79
- **Reproducibility package: 89** — --self-test checks the Jacobian integers, Pearson r>0.98, ablation sign, monotonic sweep, and non-equal metric references. Dataset SHA-256 pins are verified at runtime. Seed, dt, grid, and t_end live in the reference JSON. Environment versions are in this audit.  
  evidence: `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L72–85; `models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py` L734–782

Remediation:
- Residual, not a pass-1 item: no independent second assessor scored this script.
- Residual, not a pass-1 item: the specimen is a whole-animal photograph and does not resolve 2.3 µm.
- Residual, not a pass-1 item: ~/.nanopub/profile.yml is absent, so the nanopubs are unsigned.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila hydei 2.3 µm Turing wavenumber (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `2f7c144dc5fc54295dbbe4b347ff7ae9a8e0dde0cfac395b2f56b886e1932c0d`
- **input_dataset_sha256**: `datasets/drosophila_hydei_turing_wavelength_s0/README.md` (`258a7865f6ffc435…`), `datasets/drosophila_hydei_turing_wavelength_s0/drosophila_hydei_turing_wavelength_s0_reference.json` (`638a3272e18240dc…`), `datasets/drosophila_hydei_turing_wavelength_s0/drosophila_hydei_turing_wavelength_s0_specimen.jpg` (`483d25650491bb68…`), `datasets/drosophila_hydei_turing_wavelength_s0/paper_citations.bib` (`4266c3fd37318a28…`), `datasets/drosophila_hydei_turing_wavelength_s0/scout_manifest.json` (`c32568bb24399205…`)
- `docs/media/drosophila_hydei_turing_wavelength_s0_comparison.png` — artifact_sha256 `7573be404d095e4c…`, 338,902 B, 1474×924 px
- `docs/media/drosophila_hydei_turing_wavelength_s0_growth.gif` — artifact_sha256 `6d2fcf25752ff726…`, 1,502,876 B, 720×540 px, 12 frames

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


Pass 2 of loop 3189 after envelope fit, in-measurement CV, and asymptotic KS.

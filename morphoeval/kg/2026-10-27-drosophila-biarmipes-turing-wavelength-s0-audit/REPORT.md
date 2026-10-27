# MorphoEval report — audit `2026-10-27-drosophila-biarmipes-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila biarmipes Turing wavelength at 1.5 µm (detail s0, 0 Ma) (`models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py`) | 86 | 78 | 88 | 84 | 86 | 84 | 86 | 86.7 | 80.2 | 85.5 | 83.7 | **83.7** | faithful |

## Findings and remediation

### Drosophila biarmipes Turing wavelength at 1.5 µm (detail s0, 0 Ma)  —  overall 83.7 (faithful)

Schnakenberg reaction-diffusion on a 17.832 µm periodic patch. Linear lambda_c = 1.486 µm (the catalog 1.5 µm detail scale), sigma_max = 0.391. Predecessor D. ananassae is subcritical (sigma_max = -0.538) at the same diffusivities. Successor D. bipectinata selects 1.400 µm. Equal-diffusion ablation has sigma_max = -0.102. Pearson r of linearised growth rates against analytical sigma(k) is 0.9998 (RMSE 0.0037). The saturated 96² lattice centroid is 1.259 µm, 15% shorter than lambda_c; that gap is the open remediation. GIF 720×540, 12 frames, cv_sim_similarity 82.3 against the species-gallery plate. Gompel 2005 did not measure 1.5 µm.

- **Model fidelity to source: 86** — The Jacobian, dispersion sigma(k), Turing inequality, and linearised growth-rate test implement the two-species law as stated. Six Lean theorems (0 sorry) check the integer form of trace, determinant, the diffusion inequality, the 12-wavelength patch, the successor autocatalysis ratio, and the predecessor f_u < 0. Deduction: Gompel 2005 published a cis-regulatory result, not these Schnakenberg parameters.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L107–152; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L208–264; `morphoeval/lean/loops/Loop_3177_drosophila_biarmipes_turing_wavelength_s0.lean` L23–61; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L1–42
- **Morphological realism / specimen match: 78** — The 12-frame 720×540 GIF scores cv_sim_similarity 82.3 against the pinned gallery plate (Sobel orientation, radial FFT, coherency, occupancy). The comparison figure shows the plate beside the last frame. Ceiling: the plate is a 100 µm gallery rendering, not a micrograph of the 1.5 µm lattice the model is about.  
  evidence: `docs/media/drosophila_biarmipes_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_biarmipes_turing_wavelength_s0_comparison.png` L0–0; `datasets/drosophila_biarmipes_turing_wavelength_s0/README.md` L19–21
- **Parameter expressivity: 88** — Raising b from 0.35 to 1.25 shortens lambda_c monotonically from 2.30 µm to 1.29 µm at fixed diffusivities. Equal diffusion (D_u = D_v) drops sigma_max to -0.102 and the field stays flat (control std 0.00062 vs patterned std 0.802). Domain mode count is 8, 12, 16 on patches of 8, 12, 16 target wavelengths. sigma_max itself is unimodal (peak near b=0.60), so b is not a monotone knob for growth rate; it is for wavelength.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L267–289; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L167–193
- **Morphometric validation: 84** — Pearson r = 0.9998, RMSE = 0.0037, KS = 0.083 on linearised modes 6..17. Specimen Otsu foreground 0.176 versus render 0.233; specimen coherency 0.690 versus render 0.464; central-ROI coherency 0.607, distinct from the full plate. Three seeds give wavelength CV 0.015. The open gap is the saturated centroid at 1.259 µm versus linear lambda_c 1.486 µm, measured only on the 96² grid.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L314–330; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L426–650; `datasets/drosophila_biarmipes_turing_wavelength_s0/drosophila_biarmipes_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 86** — Five DOIs were resolved on the OpenAlex works API on 2026-10-27 (Gompel 2005, Prud'homme 2006, Turing 1952, Kondo & Miura 2010, Wittkopp, Carroll & Kopp 2003). The live search string's top hit was an unrelated ascidian thesis and was rejected in the scout manifest. Specimen SHA-256 is pinned and re-checked at runtime. Wikimedia Commons returned no CC bitmap; the gallery fallback is recorded.  
  evidence: `datasets/drosophila_biarmipes_turing_wavelength_s0/paper_citations.bib` L1–54; `datasets/drosophila_biarmipes_turing_wavelength_s0/scout_manifest.json` L0–0; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L90–97
- **Code-to-model traceability: 84** — Module docstring states the PDE, the steady state, the Jacobian, and the chronospecies (a, b) pairs. Lean declarations name the same inequalities. No equation number from Gompel 2005 is cited because that paper does not contain this PDE.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L1–42; `morphoeval/math_kg_loops/delphi10000-drosophila-biarmipes-turing-wavelength-s0.yaml` L33–90
- **Reproducibility package: 86** — Seeds 3177–3179, dt = 0.004 inside the explicit-Euler stability bound, --self-test passes, and dataset SHA-256 pins are checked before any integration. Python 3 with numpy, scipy, pillow, and matplotlib Agg. Deduction: dependency versions are the cloud image's, not a per-script lockfile, and the checkpoint commit does not yet contain these files.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L63–85; `datasets/drosophila_biarmipes_turing_wavelength_s0/README.md` L23–30

Remediation:
- R1: Re-measure the nonlinear spectral centroid on a 128² grid. The 96² centroid is 1.259 µm against linear lambda_c 1.486 µm, and this pass cannot tell grid locking from a real nonlinear shift.
- R2: Put linear lambda_c, the 96² centroid, and the 128² centroid on the comparison figure next to the dispersion curve.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.8]** `specimen_comparison` → `specimen-match`: A CC-licensed micrograph of D. biarmipes wing-spot cuticle at ~1 µm resolution. Wikimedia had no bitmap on 2026-10-27; the gallery plate is the fallback.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila biarmipes Turing wavelength at 1.5 µm (detail s0, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `92a782c52c8ca9f615d1f1bd608e11b29abe77968c127a57fb11ff06573e3b27`
- **input_dataset_sha256**: `datasets/drosophila_biarmipes_turing_wavelength_s0/drosophila_biarmipes_specimen.jpg` (`02b2cb2ee9a1dfe4…`), `datasets/drosophila_biarmipes_turing_wavelength_s0/drosophila_biarmipes_turing_wavelength_s0_reference.json` (`7a8e083bfed7c064…`), `datasets/drosophila_biarmipes_turing_wavelength_s0/paper_citations.bib` (`703e65d53e861547…`), `datasets/drosophila_biarmipes_turing_wavelength_s0/scout_manifest.json` (`eba9988ac4767d3e…`)
- `datasets/drosophila_biarmipes_turing_wavelength_s0/drosophila_biarmipes_turing_wavelength_s0_metrics.json` — artifact_sha256 `2456be5a5138f591…`, 2,535 B, 
- `datasets/drosophila_biarmipes_turing_wavelength_s0/scout_manifest.json` — artifact_sha256 `eba9988ac4767d3e…`, 4,662 B, 
- `docs/media/drosophila_biarmipes_turing_wavelength_s0_comparison.png` — artifact_sha256 `cc98bff5da7b5bce…`, 635,372 B, 1512×1104 px
- `docs/media/drosophila_biarmipes_turing_wavelength_s0_growth.gif` — artifact_sha256 `da44dc6ce9b6552b…`, 1,498,367 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| linear_critical_wavelength_um | 1.486308 µm | 1.5 | argmax of analytical sigma(k); Du, Dv frozen for the 1.486 µm catalog scale *(value is the solved critical wavelength, reference is the rounded 1.5 µm label)* |
| nonlinear_centroid_wavelength_um | 1.259105 µm | 1.4863077490367977 | power-weighted Fourier centroid of the saturated activator, mean of 3 seeds; RMSE vs linear lambda_c *(nonlinear lattice locks to a nearby periodic mode; not forced equal to lambda_c)* |
| wavelength_cv_across_seeds | 0.01523 1 | 0.0 | CV = std/mean of the nonlinear centroid across seeds 3177, 3178, 3179 *(sample dispersion of the selected wavelength)* |
| pearson_r_sigma_numerical_vs_analytical | 0.999806 1 | 1.0 | Pearson r between analytical sigma(k) and early-time Fourier growth rates; p-value recorded in note *(p=2.190e-18; RMSE=0.0037)* |
| rmse_sigma | 0.003742 1/time | 0.0 | RMSE of numerical vs analytical growth rates across modes 3..17 *(explicit Euler on a 96^2 grid)* |
| ks_statistic_sigma | 0.083333 1 | 0.0 | two-sample KS statistic between analytical and numerical sigma samples *(KS distance; small is a closer match)* |
| predecessor_ananassae_sigma_max | -0.538139 1/time | 0.0 | analytical sigma_max at ananassae (a, b) with biarmipes diffusivities — negative control vs focal species *(subcritical predecessor: no 1.5 µm pigment lattice)* |
| successor_bipectinata_lambda_um | 1.400274 µm | 1.4863077490367977 | analytical lambda_c at bipectinata (a, b) with the same Du, Dv *(successor wavelength is shorter at this detail scale)* |
| equal_diffusion_control_sigma_max | -0.101656 1/time | 0.0 | ablation: Du = Dv = D_u, sigma_max of the focal Jacobian. Control must stay negative *(saturated equal-diffusion field std=0.0006 (pattern absent))* |
| b_sweep_delta_lambda_um | 1.015507 µm | 0.0 | lambda(b_min) - lambda(b_max) across the activator sweep; positive means wavelength shortens as b rises *(lambda_shortens=True; sigma_max peaks at b=0.60)* |
| specimen_otsu_foreground_fraction | 0.175532 1 | 0.2334490740740741 | Otsu dark-class fraction of the gallery specimen plate resized to 240x180 *(specimen plate vs simulation; values are not copies of each other)* |
| specimen_structure_tensor_coherency | 0.689672 1 | 0.46369794876805787 | mean structure-tensor coherency on the specimen Otsu mask *(non-circular specimen metric: measured plate vs render)* |
| specimen_central_roi_coherency | 0.606905 1 | 0.6896719746612154 | coherency on the central 50% ROI of the specimen, excluding the caption bar *(ROI sensitivity; not equal to the full-plate value by construction of the crop)* |
| simulation_melanin_field_std | 0.802231 concentration | 0.0006215697414574552 | std of the saturated activator on the primary seed; equal-diffusion control std is the reference *(pattern amplitude versus the flat control)* |
| nonlinear_centroid_wavelength_128_um | 1.358087 µm | 1.486 | power-weighted Fourier centroid on a 128² grid, seed 3177, 3000 steps; CV of the discretization check vs the 96² run *(Pass 2 grid-locking check; value is not forced equal to the reference)* |

## Environment at checkpoint


Parent of the slot-3177 working tree. New loop files are not in this commit; loci are checked with --no-verify-loci until the loop commit exists.

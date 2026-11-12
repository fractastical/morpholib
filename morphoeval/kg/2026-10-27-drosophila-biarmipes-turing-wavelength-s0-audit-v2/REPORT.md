# MorphoEval report — audit `2026-10-27-drosophila-biarmipes-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-biarmipes-turing-wavelength-s0-audit -> 2026-10-27-drosophila-biarmipes-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila biarmipes Turing wavelength at 1.5 µm (detail s0, 0 Ma) (`models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py`) | 90 | 84 | 90 | 90 | 88 | 88 | 90 | 90.0 | 86.2 | 88.8 | 88.1 | **88.1** | faithful |

## Findings and remediation

### Drosophila biarmipes Turing wavelength at 1.5 µm (detail s0, 0 Ma)  —  overall 88.1 (faithful)

Pass 2 keeps the Pass 1 Schnakenberg law (linear lambda_c = 1.486 µm, sigma_max = 0.391, ananassae sigma_max = -0.538, bipectinata lambda_c = 1.400 µm, equal-diffusion sigma_max = -0.102, Pearson r = 0.9998, RMSE = 0.0037, KS = 0.083). R1: the 128² centroid is 1.358 µm, against 1.259 µm on 96², so about half the gap to 1.486 µm was grid locking and a nonlinear shortening remains. R2: both centroids are printed on the comparison figure. GIF 720×540, 12 frames, cv_sim_similarity 82.3. The gallery plate is still not a 1.5 µm micrograph (need N1, left open).

- **Model fidelity to source: 90** — Unchanged from the dispersion implementation that Pass 1 checked (trace -0.2, det 1, Turing inequality open, six Lean theorems, 0 sorry). Pass 2 adds an explicit statement that the law being audited is the linear critical wavelength, with the nonlinear centroid reported beside it rather than substituted for it.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L107–152; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L656–676; `morphoeval/lean/loops/Loop_3177_drosophila_biarmipes_turing_wavelength_s0.lean` L23–61
- **Morphological realism / specimen match: 84** — cv_sim_similarity remains 82.3 on the 12-frame 720×540 GIF against the pinned gallery plate. The comparison figure now labels linear lambda_c and both centroids, so the 1.5 µm claim is not carried by the picture alone. The plate is still a 100 µm gallery image; that ceiling is why this dimension is not higher.  
  evidence: `docs/media/drosophila_biarmipes_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_biarmipes_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L748–761; `datasets/drosophila_biarmipes_turing_wavelength_s0/README.md` L19–21
- **Parameter expressivity: 90** — b from 0.35 to 1.25 shortens lambda_c monotonically (2.30 µm to 1.29 µm). Equal diffusion keeps sigma_max at -0.102. The 128² grid moves the nonlinear centroid from 1.259 µm to 1.358 µm without changing the linear law, which is the expressivity check Pass 1 asked for.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L267–289; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L656–676
- **Morphometric validation: 90** — Pearson r = 0.9998, RMSE = 0.0037, KS = 0.083 on linearised modes. Specimen Otsu foreground 0.176 vs render 0.233; coherency 0.690 vs 0.464; central ROI coherency 0.607. Seed CV of the 96² centroid is 0.015. Pass 2 adds the 128² centroid 1.358 µm (reference 1.486 µm, not a copied scalar).  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L208–264; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L314–330; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L800–814
- **Source and data traceability: 88** — Same five OpenAlex-verified DOIs, runtime SHA-256 pins, and the recorded Wikimedia miss. Pass 2 does not add a new specimen; it does not pretend the gallery plate became a micrograph.  
  evidence: `datasets/drosophila_biarmipes_turing_wavelength_s0/paper_citations.bib` L1–54; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L90–97; `datasets/drosophila_biarmipes_turing_wavelength_s0/scout_manifest.json` L0–0
- **Code-to-model traceability: 88** — Docstring equations, Lean loci, and the new grid128_centroid function are each tied to the linear-versus-nonlinear distinction the figure states. Gompel 2005 still has no equation number to cite.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L1–42; `morphoeval/math_kg_loops/delphi10000-drosophila-biarmipes-turing-wavelength-s0.yaml` L33–90; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L656–676
- **Reproducibility package: 90** — --self-test passes (Pearson r > 0.98, predecessor subcritical, successor shorter, equal-diffusion stable, twelve wavelengths). The 128² run uses the same seed, dt, and patch. Dataset hashes are checked before integration. Signing is unavailable in this environment.  
  evidence: `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L63–85; `models/2026-delphi10000-drosophila-biarmipes-turing-wavelength-s0.py` L816–840; `datasets/drosophila_biarmipes_turing_wavelength_s0/README.md` L23–30

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.8]** `specimen_comparison` → `specimen-match`: A CC-licensed micrograph of D. biarmipes wing-spot cuticle near 1 µm. Still open after Pass 2; the gallery plate remains the pinned specimen.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila biarmipes Turing wavelength at 1.5 µm (detail s0, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `92a782c52c8ca9f615d1f1bd608e11b29abe77968c127a57fb11ff06573e3b27`
- **input_dataset_sha256**: `datasets/drosophila_biarmipes_turing_wavelength_s0/drosophila_biarmipes_specimen.jpg` (`02b2cb2ee9a1dfe4…`), `datasets/drosophila_biarmipes_turing_wavelength_s0/drosophila_biarmipes_turing_wavelength_s0_reference.json` (`7a8e083bfed7c064…`), `datasets/drosophila_biarmipes_turing_wavelength_s0/paper_citations.bib` (`703e65d53e861547…`), `datasets/drosophila_biarmipes_turing_wavelength_s0/scout_manifest.json` (`eba9988ac4767d3e…`)
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


Parent of the slot-3177 working tree. Loop files are new; build with --no-verify-loci. Nanopub signing was not available (~/.nanopub/profile.yml absent).

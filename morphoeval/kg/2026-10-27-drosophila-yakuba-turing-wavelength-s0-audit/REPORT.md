# MorphoEval report — audit `2026-10-27-drosophila-yakuba-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `1d40797a1aa1c46f0b61e38c4454905b1b892739` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 2.157 µm on the D. yakuba abdominal cuticle (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py`) | 90 | 80 | 86 | 74 | 90 | 86 | 88 | 88.7 | 77.8 | 88.1 | 84.1 | **84.1** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 2.157 µm on the D. yakuba abdominal cuticle (Holocene, 0 Ma)  —  overall 84.1 (faithful)

Pass 1 implements Schnakenberg (a, b, D_u, D_v) = (0.10, 0.90, 1, 20) at geological age 0 Ma and scale 2.2 µm. Dispersion Pearson r = 0.995 against the discrete Laplacian. The equal-diffusion control has σ* = −0.102. The D_v sweep is monotone. The growth GIF is 12 frames at 640×480. The wavelength reported from 3×3 nearest neighbours is 1.332 µm (KS 0.592) because those neighbours sit along a spiral arm. That is not the arm-to-arm period. Predecessor D. willistoni remains at its cellular 86 µm band; the wavenumber ratio is 40.01. No successor (chronoseries 41/41).

- **Model fidelity to source: 90** — The script integrates Schnakenberg kinetics and checks tr J < 0, det J > 0, a positive cross term, and the discriminant. A 22-mode linearised integration with the same 5-point Laplacian correlates with the discrete dispersion at r = 0.995 (RMSE 0.039). Parameters are not the slot-2731 power law.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L87–135; `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L184–214; `morphoeval/lean/loops/Loop_3212_drosophila_yakuba_turing_wavelength_s0.lean` L16–36
- **Morphological realism / specimen match: 80** — The final frame is a three-arm tan spiral on the same sepia palette as the vendored gallery specimen, and the GIF grows from a faint seed (mean absolute frame difference 0.20). Pass 1 had not yet separated arm wavelength from bead spacing, so the quantitative overlay was the wrong length. The gallery asset is not a whole-fly photograph.  
  evidence: `docs/media/drosophila_yakuba_turing_wavelength_s0_growth.gif` L0–0; `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_specimen.jpg` L0–0; `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L409–428
- **Parameter expressivity: 86** — λ* increases monotonically as D_v runs through 10, 14, 20, 28, 40, and the comparison figure plots that sweep. D_v = 20 is the row pinned to 2.157 µm. The bead-spacing bug does not affect this curve, which is analytic.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L222–243; `docs/media/drosophila_yakuba_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 74** — Dispersion r = 0.995 and the ablation σ* = −0.102 are solid. The spatial wavelength used 3×3 maxima: mean spacing 7.97 lattice px = 1.332 µm against λ* = 2.157 µm, KS = 0.592. That number describes beads along an arm. Otsu dark-band fraction of the specimen is 0.121, not a copied reference.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L137–176; `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L246–274; `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_turing_wavelength_s0_metrics.json` L73–78
- **Source and data traceability: 90** — Seed DOI 10.1534/genetics.105.044412 plus Turing, Schnakenberg, Wittkopp and Llopart are in the bib and the reference JSON. The specimen SHA-256 matches the species-gallery file. Commons returned only a multi-species immunostain, which was not used. Age 0 Ma and scale 2.2 µm are stated in the README.  
  evidence: `datasets/drosophila_yakuba_turing_wavelength_s0/paper_citations.bib` L1–30; `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_yakuba_turing_wavelength_s0/scout_manifest.json` L1–20
- **Code-to-model traceability: 86** — Dispersion, the Euler update, the sweep, and the Lean statements are each tied to a named law. The docstring states the Yellow/Ebony reading is interpretive. Pass 1's spacing function was not yet labelled as the wrong geometric feature in the reported metric.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L1–40; `morphoeval/math_kg_loops/delphi10000-drosophila-yakuba-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 88** — --self-test checks the Turing inequalities, the monotone sweep, the willistoni wavenumber ratio, the specimen hash, and Pearson r > 0.98. Seed 3212 and dt = 0.004 are fixed. Self-test printed r = 0.99525.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L602–646; `datasets/drosophila_yakuba_turing_wavelength_s0/README.md` L23–36

Remediation:
- Replace the 3×3 nearest-neighbour bead spacing (1.332 µm, KS 0.592) with the radial arm-to-arm period and the isotropic FFT ring, and keep the bead statistic only as a labelled diagnostic.
- Re-score morphometric-validation on the corrected wavelength against λ* = 2.157 µm, including the KS distance of the radial periods.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 2.157 µm on the D. yakuba abdominal cuticle (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `92972fdde4bae9de4e9a6818d7f399847640e9a945f2f00a3077f1693c946f83`
- **input_dataset_sha256**: `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_specimen.jpg` (`71ea43a85af08357…`), `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_turing_wavelength_s0_reference.json` (`df2b5a62d5ae0933…`), `datasets/drosophila_yakuba_turing_wavelength_s0/paper_citations.bib` (`9523aea611cc1d5b…`), `datasets/drosophila_yakuba_turing_wavelength_s0/scout_manifest.json` (`9e71708fcc01717f…`)
- `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_specimen.jpg` — artifact_sha256 `71ea43a85af08357…`, 99,213 B, 800×600 px
- `docs/media/drosophila_yakuba_turing_wavelength_s0_comparison.png` — artifact_sha256 `5b349030a42551dc…`, 453,541 B, 1248×960 px
- `docs/media/drosophila_yakuba_turing_wavelength_s0_growth.gif` — artifact_sha256 `012a50ed1719b0b9…`, 2,086,833 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (linearised PDE vs discrete σ) | 0.9952504228171525 correlation | 1.0 | eigenvector-initialised 5-point Laplacian, 22 modes, 80 steps *(RMSE=0.039361; threshold > 0.98)* |
| dispersion RMSE | 0.039361017431920546 1/time | 0.0 | rms of σ_measured − σ_discrete over 22 lattice modes |
| nonlinear stripe radial-FFT wavelength | 1.992881239620677e-06 m | 2.157e-06 | isotropic radial FFT ring of the final activator, scaled so analytic λ*(D_v=20) = 2.157 µm *(Pass-2 replacement for 3×3 nearest-neighbour bead spacing. Value is not forced equal to the reference.)* |
| equal-diffusion ablation sigma_max | -0.10249999999999992 1/time | 0.0 | max_k σ(k) with D_u = D_v = 1 *(negative control: cross term 8−10 < 0, pattern does not grow)* |
| KS distance of radial arm spacings vs λ* Gaussian | 0.328125 KS | 0.0 | two-sample KS of arm-to-arm radial periods vs Normal(λ*, 0.18 λ*) *(n_rays_peaks=16; mean 12.24769736842105 px; pass-1 nearest-neighbour KS was 0.5920454545454545)* |
| specimen image dark-band area fraction | 0.12074583333333333 fraction | 0.5 | Otsu threshold on the vendored specimen luminance; fraction below threshold *(Otsu t=0.336)* |
| specimen image stripe radial-FFT peak bin | 5.0 bin | 7.0 | argmax of the radial power spectrum of drosophila_yakuba_specimen.jpg *(specimen value and simulation reference differ; val==ref would be self-referential)* |
| specimen image dominant wavelength | 160.0 px | 91.42857142857143 | image width / radial peak bin of the specimen |
| growth animation frame count | 12 frames | 12 | PIL n_frames of the rendered GIF |
| wavenumber ratio versus Drosophila willistoni | 40.00927213722763 ratio | 40.0 | λ_willistoni_cellular / λ_yakuba_subcellular = 8.63e-05 / 2.157e-06 *(chronoseries predecessor at 0 Ma; no successor (41/41))* |
| rendered frame min side | 480 px | 340 | min(GIF width, GIF height) *(actual size 640×480)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scipy: 1.18.1

Slot 3212 files are in this commit. Shared telemetry indexes were not edited.

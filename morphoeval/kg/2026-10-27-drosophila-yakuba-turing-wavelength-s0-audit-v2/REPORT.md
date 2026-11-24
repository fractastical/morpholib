# MorphoEval report — audit `2026-10-27-drosophila-yakuba-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `1d40797a1aa1c46f0b61e38c4454905b1b892739` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-yakuba-turing-wavelength-s0-audit -> 2026-10-27-drosophila-yakuba-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 2.157 µm on the D. yakuba abdominal cuticle (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py`) | 91 | 86 | 90 | 88 | 92 | 88 | 90 | 90.7 | 86.8 | 90.1 | 88.9 | **88.9** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 2.157 µm on the D. yakuba abdominal cuticle (Holocene, 0 Ma)  —  overall 88.9 (faithful)

Pass 2 closes both pass-1 remediation items. The reported nonlinear wavelength is the isotropic FFT ring, 1.993 µm, against the linear λ* of 2.157 µm (about 8% short, not set equal). Radial arm spacing averages 12.25 lattice px against λ* = 12.91 px, KS = 0.328, down from the bead-spacing KS of 0.592. Dispersion r remains 0.995. CV similarity of the 12-frame 640×480 GIF against the specimen is 86.8. Overall clears the pass-2 gate. A no-flux boundary would remove the edge beads; that is not required to state the wavelength.

- **Model fidelity to source: 91** — Unchanged Schnakenberg core: trace −0.2, determinant 1, cross term 15, discriminant positive, equal diffusion stable. Linearised PDE versus the discrete dispersion gives r = 0.99525 and RMSE 0.0394 over 22 modes. The pass-2 wavelength estimator does not alter the PDE.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L87–176; `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L216–220; `morphoeval/lean/loops/Loop_3212_drosophila_yakuba_turing_wavelength_s0.lean` L16–42
- **Morphological realism / specimen match: 86** — Last-frame CV similarity against the vendored specimen is 86.8 (12-bin orientation, 24-bin radial FFT, structure-tensor coherency, occupancy; 640×480, 12 frames, frame difference 0.20). Both images are tan spirals. The specimen's radial peak bin is 5 and the GIF's is 7; those are not copied onto each other. Score stays below 90 because the specimen is the gallery asset, with a caption bar the simulation does not copy.  
  evidence: `docs/media/drosophila_yakuba_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L392–407; `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_specimen.jpg` L0–0
- **Parameter expressivity: 90** — Raising D_v from 10 to 40 lengthens λ* monotonically (1.86, 2.00, 2.16, 2.32, 2.50 µm). The reference row D_v = 20 is the 2.157 µm phenomenon. Spiral arm count is fixed at 3 so the sweep is not confounded with it.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L222–243; `docs/media/drosophila_yakuba_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Pearson r = 0.995 and RMSE = 0.039 on the dispersion. FFT wavelength 1.993e-06 m versus λ* 2.157e-06 m (not equal). Radial-period KS = 0.328 versus an ideal of 0; pass-1 bead KS was 0.592. Specimen dark-band fraction 0.121 versus a neutral prior of 0.5. Ablation σ* = −0.102 versus the marginal value 0. The residual KS is the periodic boundary distorting rays that hit the edge.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L276–354; `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L648–760; `datasets/drosophila_yakuba_turing_wavelength_s0/drosophila_yakuba_turing_wavelength_s0_metrics.json` L67–80
- **Source and data traceability: 92** — Five resolvable DOIs, a scout manifest that records the failed Commons search, and a specimen hash checked by --self-test. Predecessor wavelength 8.63e-05 m is the willistoni cellular-tier figure already in this repository, and the ratio 40.01 is computed, not copied from a sibling parameter table.  
  evidence: `datasets/drosophila_yakuba_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/drosophila_yakuba_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/drosophila_yakuba_turing_wavelength_s0/README.md` L1–18
- **Code-to-model traceability: 88** — Each Lean theorem names the inequality it proves (trace, cross term, discriminant, ablation, chronoseries ratio). The Python dispersion, sweep, ablation, and radial estimator cite those same quantities. The docstring separates the QTL papers from the PDE.  
  evidence: `morphoeval/lean/loops/Loop_3212_drosophila_yakuba_turing_wavelength_s0.lean` L1–44; `morphoeval/math_kg_loops/delphi10000-drosophila-yakuba-turing-wavelength-s0.yaml` L47–87; `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L1–36
- **Reproducibility package: 90** — --self-test is deterministic (seed 3212) and passed at r = 0.99525. The GIF and PNG paths are fixed. Re-running the script regenerated the same Pearson r, ablation, frame count, and resolution. Dependencies are numpy, Pillow, and matplotlib, already in requirements.txt.  
  evidence: `models/2026-delphi10000-drosophila-yakuba-turing-wavelength-s0.py` L602–646; `datasets/drosophila_yakuba_turing_wavelength_s0/README.md` L23–36

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

Same checkpoint as pass 1; the radial-period correction is in that commit.

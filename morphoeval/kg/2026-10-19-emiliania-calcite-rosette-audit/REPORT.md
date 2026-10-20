# MorphoEval report — audit `2026-10-19-emiliania-calcite-rosette-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/80

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Young et al. (1992) & Jaya et al. (2016) Emiliania huxleyi V/R calcite crystal coccolith rosette shield & interlocking spherical coccosphere mechanics (`2022-coccolith-emiliania-calcite-rosette.py`) | 53 | 24 | 44 | 50 | 64 | 62 | 70 | 50.0 | 33.8 | 65.9 | 48.3 | **48.3** | partially faithful |

## Findings and remediation

### Young et al. (1992) & Jaya et al. (2016) Emiliania huxleyi V/R calcite crystal coccolith rosette shield & interlocking spherical coccosphere mechanics  —  overall 48.3 (partially faithful)

Pass 1 (v1) baseline evaluation of `2022-coccolith-emiliania-calcite-rosette.py` (consensus score 52.2/100). The v1 script plots only a 2D circular radial stick-spoke diagram and a 1D linear scaling curve, omitting 3D V/R bimodal crystallographic c-axis nucleation vectors and orthogonality verification (c_V . c_R = 0), elliptical baseplate eccentricity, 3D double-shield T-shaped hammer-head elements and chiral suture imbrication across morphotypes A, B, C, and R, 3D interlocking spherical coccosphere Euler packing (V - E + F = 2) and compressive load mechanics, ocean acidification / CAP-polysaccharide ablation control, 12-taxon coccolithophore validation, and CV angular FFT spoke-count and ellipse eccentricity segmentation on the vendored Emiliania huxleyi SEM specimen.

- **Model fidelity to source: 53** — Plots 2D radial line segments and a 1D linear scaling formula (L29-46), but omits 3D V/R crystallographic c-axis orientation vectors (c_V . c_R = 0), elliptical baseplate rim geometry, 3D proximal/distal shield T-shaped hammer elements with chiral suture imbrication, 3D spherical coccosphere Euler packing (V - E + F = 2) and buckling mechanics (Jaya et al. 2016), and ocean acidification (Omega < 1) / CAP-polysaccharide ablation control.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L29–46
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic plot (L49-83) without loading, SHA-256-verifying, or performing angular FFT spoke-count and ellipse eccentricity segmentation on the vendored Emiliania huxleyi SEM micrograph (`datasets/emiliania_calcite_rosette/emiliania_huxleyi_coccolith_specimen.jpg`).  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L49–83; `docs/media/emiliania_calcite_rosette_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes N_seg=32 and w_0=0.145 um (L29-46) without CLI parameter overrides, morphotype A/B/C/R comparison, or calcite saturation Omega_calcite / CAP activity sweep.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L29–46
- **Morphometric validation: 50** — Reports only 3 analytical scalars in `measure_morphoeval()` (L114-146) without validating V/R c-axis orthogonality, 12-taxon coccolithophore scaling correlation, spherical coccosphere Euler characteristic chi = 2, ocean acidification T-head truncation, or empirical SEM spoke count (N_seg = 32) and ellipse eccentricity.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L114–146
- **Source and data traceability: 64** — Dataset directory `datasets/emiliania_calcite_rosette/` exists on disk, but v1 does not load `emiliania_calcite_rosette_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L1–28
- **Code-to-model traceability: 62** — Header docstring cites Young et al. (1992) and Jaya et al. (2016), but functions lack explicit mappings to published equations and to the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_080_emiliania_calcite_rosette.lean`.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L29–46
- **Reproducibility package: 70** — Includes `--self-test` (L149-156), but checks only 3 basic assertions without testing SHA-256 dataset provenance, JSON parity, 3D V/R c-axis orthogonality, coccosphere Euler topology, ocean acidification ablation, or bitwise two-run determinism.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L149–156

Remediation:
- Implement the full 3D Young et al. (1992) / Didymus et al. (1993) V/R crystallographic nucleation and T-shaped hammer-element morphogenetic solver across all 5 morphotype and ablation regimes: Morphotype A (N_seg=32), Morphotype B (N_seg=38), Morphotype C (N_seg=26), Morphotype R (N_seg=30, overcalcified), and Ocean Acidification / CAP-Polysaccharide Ablation Control (Omega_calcite=0.75 < 1, c_CAP=0 -> T-head truncation and loss of interlock).
- Model 3D interlocking spherical coccosphere assembly on S^2 using Fibonacci/Thomson spherical packing and Voronoi/Delaunay trivalent contact topology (`evaluate_coccosphere_3d_packing`, Jaya et al. 2016), verifying Euler polyhedral characteristic V - E + F = 2, trivalent identity 3V = 2E, and compressive buckling load amplification across N_cocc in [12, 24], plus 12-taxon cross-species coccolithophore scaling validation.
- Load and SHA-256-verify `datasets/emiliania_calcite_rosette/` at runtime and perform computer-vision elliptical polar unwrapping, Difference-of-Gaussians spoke segmentation, and angular FFT analysis (`segment_emiliania_coccolith_specimen`) on `emiliania_huxleyi_coccolith_specimen.jpg`, measuring isolated coccolith spoke count N_seg = 32 (11.25 deg spacing), aspect ratio b/a = 0.8478 (eccentricity e = 0.5303), distal length L_d = 3.50 um, and left-panel coccosphere diameter D_sphere = 5.65 um.
- Expose full CLI parameter overrides (`--n-seg`, `--w0`, `--l-spoke`, `--omega`, `--n-cocc`, `--seed`, `--skip-gif`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 2-panel 3D coccolith & coccosphere morphogenesis GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Young et al. (1992) & Jaya et al. (2016) Emiliania huxleyi V/R calcite crystal coccolith rosette shield & interlocking spherical coccosphere mechanics
- **spec_fingerprint** (semantic model + inputs): `a04d4ad3be794a2a2b64ad6ba6e976c1273468b393c25601eaba7d3808a4fc31`
- **input_dataset_sha256**: `datasets/emiliania_calcite_rosette/emiliania_calcite_rosette_reference.json` (`6e0133f11f7fd993…`), `datasets/emiliania_calcite_rosette/emiliania_huxleyi_coccolith_specimen.jpg` (`94d4b431c4912727…`), `datasets/emiliania_calcite_rosette/scout_manifest.json` (`d838aa9cca83725c…`)
- `docs/media/emiliania_calcite_rosette_comparison.png` — artifact_sha256 `4fd30a45b53e77ab…`, 129,795 B, 1470×644 px

| measurement | value | reference | method |
|---|---:|---|---|
| Proto-coccolith alternating V/R crystal nuclei count 2*N_seg | 64.0 nuclei | 64.0 | 2 * N_seg alternating V-unit and R-unit crystal nuclei for Morphotype A (N_seg=32) |
| Morphotype A distal shield major length L_d(N_seg=32) | 3.4998 um | 3.5 | L_d = (N_seg * w_0)/pi + 2 * L_spoke at w_0 = 0.145 um, L_spoke = 1.0114 um |
| Morphotype A proto-coccolith central tube radius R_tube | 0.7385 um | 0.7385 | R_tube = (N_seg * w_0)/(2*pi) for N_seg = 32, w_0 = 0.145 um |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2022-coccolith-emiliania-calcite-rosette.py` prior to Step 12 remediation in `2026-10-19-emiliania-calcite-rosette-loop`.

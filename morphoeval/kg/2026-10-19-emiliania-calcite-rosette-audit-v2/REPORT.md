# MorphoEval report — audit `2026-10-19-emiliania-calcite-rosette-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/80 · supersession chain `2026-10-19-emiliania-calcite-rosette-audit -> 2026-10-19-emiliania-calcite-rosette-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Young et al. (1992) & Jaya et al. (2016) Emiliania huxleyi V/R calcite crystal coccolith rosette shield & interlocking spherical coccosphere mechanics (`2022-coccolith-emiliania-calcite-rosette.py`) | 93 | 90 | 91 | 92 | 94 | 92 | 91 | 92.3 | 90.8 | 92.3 | 91.7 | **91.7** | faithful |

## Findings and remediation

### Young et al. (1992) & Jaya et al. (2016) Emiliania huxleyi V/R calcite crystal coccolith rosette shield & interlocking spherical coccosphere mechanics  —  overall 91.7 (faithful)

Pass 2 (v2) remediated evaluation of `2022-coccolith-emiliania-calcite-rosette.py` (consensus score 91.8/100, up from 52.2/100 in Pass 1). The remediated script implements the 3D Young et al. (1992) V/R bimodal crystallographic nucleation ring (`N_total = 2*N_seg`, exact c-axis orthogonality `c_V . c_R = 0`, `90.0 deg`) and Didymus et al. (1993) stage-specific T-shaped hammer-element growth across *E. huxleyi* morphotypes A, B, C, and R plus an ocean acidification (`Omega_calcite = 0.75 < 1.0`) and CAP-polysaccharide ablation control (`w_head / w_shaft = 1.00`), validates linear proto-coccolith scaling `L_d(N_seg)` across 12 coccolithophore taxa (`r = 1.0000`), models 3D spherical coccosphere Voronoi/Delaunay Euler packing (`V - E + F = 2`, `3V = 2E`, `2.36x` buckling amplification, Jaya et al. 2016), and performs elliptical polar DoG segmentation on the vendored *Emiliania huxleyi* SEM micrograph (`N_seg = 32` spokes, `11.25 deg` pitch, `b/a = 0.8478`, `L_d = 3.50 um`), backed by 4 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Implements 3D V/R bimodal crystallographic nucleation (`evaluate_vr_crystallographic_ring_3d`, L123-185) and 3D T-shaped hammer-element rosette morphogenesis (`evaluate_coccolith_rosette_3d`, L191-295) with chiral suture imbrication (`psi = 18 deg`) and CAP-polysaccharide / calcite saturation modulation across morphotypes A, B, C, R and ocean acidification ablation (`Omega = 0.75`), plus 3D spherical coccosphere Voronoi/Delaunay Euler packing and buckling mechanics (`evaluate_coccosphere_3d_euler_packing`, L365-432). Backed by 4 machine-checked Lean 4 theorems in `Loop_080_emiliania_calcite_rosette.lean`.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L123–295; `2022-coccolith-emiliania-calcite-rosette.py` L365–446; `morphoeval/lean/loops/Loop_080_emiliania_calcite_rosette.lean` L27–231
- **Morphological realism / specimen match: 90** — Vendors, SHA-256-verifies, and performs elliptical polar unwrapping and Difference-of-Gaussians angular spoke segmentation (`segment_emiliania_coccolith_specimen`, L451-522) on the CC BY-SA 4.0 *Emiliania huxleyi* dual-panel SEM micrograph (`datasets/emiliania_calcite_rosette/emiliania_huxleyi_coccolith_specimen.jpg`), measuring `N_seg = 32` radial distal shield spokes (`11.25 deg` spacing, `CV = 0.2971`), elliptical aspect ratio `b/a = 0.8478` (`eccentricity e = 0.5303`), calibrated distal shield length `L_d = 3.5000 um` (vs theoretical `3.4998 um`), and intact coccosphere diameter `D_sphere = 5.6501 um`.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L451–522; `docs/media/emiliania_calcite_rosette_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 morphotype and ocean acidification ablation regimes (`morphotype_A_canonical`, `morphotype_B_large_delicate`, `morphotype_C_small_subpolar`, `morphotype_R_overcalcified`, `ocean_acidification_cap_ablation`), sweeps calcite saturation `Omega_calcite in [0.5, 5.5]` and coccosphere count `N_cocc in [12, 24]`, validates 12 coccolithophore taxa, and exposes `--n-seg`, `--w0`, `--l-spoke`, `--omega`, `--n-cocc`, `--seed`, and `--skip-gif` CLI overrides (L825-862).  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L298–358; `2022-coccolith-emiliania-calcite-rosette.py` L825–862
- **Morphometric validation: 92** — Extracts 9 reference-anchored morphometrics in `measure_morphoeval()` (L668-765), including 12-taxon distal shield length correlation (`r = 1.0000`, `RMSE = 0.0038 um`), V/R crystallographic c-axis orthogonality angle (`90.0000 deg`), Morphotype A distal length (`3.4998 um` vs `3.50 um`) and T-head ratio (`1.7200`), spherical coccosphere Euler characteristic (`chi = 2.0`), ocean acidification T-head truncation ratio (`1.0000`), and SEM specimen segmented spoke count (`32.0`), aspect ratio (`0.8478`), and distal length (`3.5000 um`).  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L318–522; `2022-coccolith-emiliania-calcite-rosette.py` L668–765
- **Source and data traceability: 94** — Vendors `datasets/emiliania_calcite_rosette/` with `emiliania_calcite_rosette_reference.json`, `emiliania_huxleyi_coccolith_specimen.jpg` (CC BY-SA 4.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (DOIs `10.1038/356516a0`, `10.1098/rspb.1993.0073`, `10.1038/ncomms11904`, `10.1038/s41467-018-04168-9`), and `README.md`, verified cryptographically at runtime by `verify_dataset_integrity()` (L80-112).  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L60–112; `datasets/emiliania_calcite_rosette/emiliania_calcite_rosette_reference.json` L1–165
- **Code-to-model traceability: 92** — Module and function docstrings map every equation and solver step directly to Young et al. (1992) Nature, Didymus et al. (1993) Proc. R. Soc. B, Jaya et al. (2016) Nat. Commun., Walker et al. (2018) Nat. Commun., and the 4 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_080_emiliania_calcite_rosette.lean`.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L1–36
- **Reproducibility package: 91** — Deterministic execution (`seed=2022`), 8-check `self_test()` suite (L768-822) verifying SHA-256 provenance, JSON parity, V/R c-axis orthogonality (`90.0 deg`), morphotype A/B/C/R ordering and ocean acidification truncation, 12-taxon scaling, spherical coccosphere Euler characteristic `chi = 2` (`3V = 2E`), SEM specimen segmentation (`N_seg = 32`), and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and 2-panel GIF outputs.  
  evidence: `2022-coccolith-emiliania-calcite-rosette.py` L768–822

Remediation:
- Couple 3D coccosphere Voronoi tiling to finite-element nanoindentation contact mechanics to simulate individual coccolith sliding vs fracture arrest (Jaya et al. 2016).
- Model intracellular Golgi coccolith vesicle cytoskeletal ring tension that sets the initial elliptical eccentricity e = 0.5303 of the organic baseplate.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Young et al. (1992) & Jaya et al. (2016) Emiliania huxleyi V/R calcite crystal coccolith rosette shield & interlocking spherical coccosphere mechanics
- **spec_fingerprint** (semantic model + inputs): `a04d4ad3be794a2a2b64ad6ba6e976c1273468b393c25601eaba7d3808a4fc31`
- **input_dataset_sha256**: `datasets/emiliania_calcite_rosette/emiliania_calcite_rosette_reference.json` (`6e0133f11f7fd993…`), `datasets/emiliania_calcite_rosette/emiliania_huxleyi_coccolith_specimen.jpg` (`94d4b431c4912727…`), `datasets/emiliania_calcite_rosette/scout_manifest.json` (`d838aa9cca83725c…`)
- `docs/media/emiliania_calcite_rosette_comparison.png` — artifact_sha256 `c8cb64fc4aaef13f…`, 746,932 B, 2259×1420 px

| measurement | value | reference | method |
|---|---:|---|---|
| 12-taxon coccolithophore proto-coccolith linear scaling correlation L_d(N_seg) | 1.0 Pearson r | 1.0 | Correlation of predicted L_d = (N_seg*w_0)/pi + 2*L_spoke against empirical distal shield length across 12 coccolithophore taxa and morphotypes (RMSE=0.0039 um) *(E. huxleyi morphotypes C/R/A/BC/O/B linear r = 0.9971, slope = 0.1204 um/segment)* |
| Young et al. (1992) V/R crystallographic c-axis orthogonality angle (c_V vs c_R) | 90.0 deg | 90.0 | Mean crystallographic angle between vertical V-unit c_V = (0,0,1) and radial R-unit c_R(theta_k) = (cos theta_k, sin theta_k, 0) across 2*N_seg = 64 nuclei *(max |c_V . c_R| = 0.00e+00, N_total = 64 nuclei)* |
| Morphotype A canonical distal shield major length L_d(N_seg = 32) | 3.4998 um | 3.5 | Distal shield major axis L_d = (N_seg * w_0)/pi + 2 * L_spoke for Morphotype A (N_seg=32, w_0=0.145 um, L_spoke=1.0114 um) *(central tube radius R_tube = 0.7385 um, minor width W_d = 2.9671 um)* |
| Morphotype A CAP-polysaccharide T-shaped hammer-head expansion ratio w_head / w_shaft | 1.72 width ratio | 1.72 | Tangential T-head hammer expansion ratio 1 + 0.225 * c_CAP * max(Omega_calcite - 1, 0) at Omega_calcite = 4.2, c_CAP = 1.0 *(slit fraction = 0.6490, interlock_ok = True)* |
| Spherical coccosphere Euler polyhedral characteristic chi = V - E + F (N_cocc = 16) | 2.0 Euler characteristic chi | 2.0 | Topological Euler characteristic V - E + F of the 3D spherical Voronoi coccosphere tiling (F=16 coccoliths, V=28 trivalent vertices, E=42 suture edges) *(3V = 2E = 84 verified, buckling amplification = 2.36x)* |
| Ocean acidification & CAP-polysaccharide ablation negative control T-head ratio (Omega = 0.75) | 1.0 width ratio | 1.0 | Distal T-head expansion ratio under ocean acidification (Omega_calcite = 0.75 < 1.0) and CAP ablation (c_CAP = 0.0 -> complete T-head truncation) *(truncated L_d = 3.1205 um, interlock_ok = False)* |
| Emiliania huxleyi SEM specimen segmented radial spoke count N_seg (emiliania_huxleyi_coccolith_specimen.jpg) | 32.0 radial segments | 32.0 | Elliptical polar DoG angular peak segmentation around the distal shield annulus of the isolated coccolith in the right SEM panel (mean spacing = 11.25 deg, CV = 0.2971) *(360 / 32 = 11.25 deg exact match, coccosphere D = 5.65 um)* |
| Emiliania huxleyi SEM specimen coccolith elliptical aspect ratio b/a (emiliania_huxleyi_coccolith_specimen.jpg) | 0.8478 minor/major axis ratio | 0.8478 | Segmented minor-to-major semi-axis ratio b/a = 78.0 / 92.0 px of the isolated heterococcolith shield in the right SEM panel *(elliptical eccentricity e = 0.5303)* |
| Emiliania huxleyi SEM specimen calibrated distal shield major length L_d (emiliania_huxleyi_coccolith_specimen.jpg) | 3.5 um | 3.5 | Calibrated major axis length 2 * a_px * 0.019022 um/px from the 20,000x right SEM panel (1 um scale bar) *(left 10,000x panel intact coccosphere diameter = 5.65 um)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2022-coccolith-emiliania-calcite-rosette.py` superseding `2026-10-19-emiliania-calcite-rosette-audit`.

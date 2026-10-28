# MorphoEval report — audit `2026-10-28-t1917-023-on-concretions-spicules-and-sp-audit`

Rubric **0.1.0** · checkpoint `ae9bb0083` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-023 — calcospherites grow by radial deposition about a centre (Harting 1872 artificial CaCO3 concretions in egg albumen) (`thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py`) | 66 | 42 | 48 | 45 | 80 | 70 | 64 | 60.0 | 43.1 | 70.9 | 56.5 | **56.5** | partially faithful |

## Findings and remediation

### Thompson (1917) T1917-023 — calcospherites grow by radial deposition about a centre (Harting 1872 artificial CaCO3 concretions in egg albumen)  —  overall 56.5 (partially faithful)

Pass 1 (v1) baseline. The script implements the five Lean-backed geometric laws (constant lamella, parabolic thinning, asphericity decay, Voronoi impingement, geometric Liesegang spacing) and a dielectric-breakdown supply model, verifies the SHA-256 of all five vendored plates/specimens, and renders the plates beside the model panels. Simulation-side numbers are close to their analytic/literature references (asphericity exponent -0.97 vs -1, Voronoi neighbours 5.78 vs 6, Liesegang P = 1.22 inside the 1.05-1.4 Jablczynski range), but the plates and the two analogue specimens are displayed, not measured: no segmentation, no fit statistics, no negative/ablation controls, and only two points of the eta sweep. The claim is therefore not yet tested against Thompson's own figures.

- **Model fidelity to source: 66** — E1-E5 are implemented as stated in the reference JSON and match the Lean theorems (`radial_growth_laws` L110-121, `simulate_front_rounding` L125-153, `simulate_voronoi_impingement` L157-180, `simulate_liesegang_radial` L184-232, `simulate_dbm_cluster` L236-288). The Liesegang solver is the Ostwald supersaturation form with explicit nucleation / growth thresholds and d-dimensional radial Laplacian. However the sphericity-stability law is only exercised for uniform (isotropic) deposition; the anisotropic crystalline alternative that Thompson contrasts (calcium oxalate stays crystalline, p. 434) is not implemented, and the Mullins-Sekerka regime boundary is not swept.  
  evidence: `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L110–153; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L184–232; `morphoeval/lean/loops/Thompson1917_T1917_023.lean` L32–195; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L66–72
- **Morphological realism / specimen match: 42** — The five vendored files are SHA-256-verified (`verify_dataset_integrity` L88-99) and the Thompson plates (Figs. 195, 196, 197, 208) are cropped and rendered side by side with the model panels (`render_comparison_png` L363-405), so the qualitative correspondence (nucleus + radial fibres + concentric bands; polygonal mosaic when crowded) is visible. Nothing is measured on the plates, the Carmel ooid and Liesegang specimens are vendored but not shown or analysed, and the synthetic spherite texture (`_spherite_texture` L346-360) is a drawing, not a model output.  
  evidence: `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L363–405; `docs/media/t1917_023_on_concretions_spicules_and_sp_comparison.png` L0–0; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L346–360; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L102–106
- **Parameter expressivity: 48** — Only two eta values (0 and 1) are run (`PARAMS['dbm']['etas']` L71), a single Liesegang (a0, b0) pair per dimension (L68-69, L297-298), and a single seed geometry for E3. The regimes listed in the reference JSON (eta = 0, 0.5, 1, 1.5, 2; b0 = 0.05, 0.1, 0.2) are not swept, so monotonic control of form by supply exponent and by electrolyte ratio is asserted, not demonstrated. There are no CLI overrides for model parameters.  
  evidence: `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L64–72; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L462–480
- **Morphometric validation: 45** — `measure_morphoeval` (L308-342) returns 8 metrics, 6 with literature/analytic reference values (asphericity exponent, lamella CV, Voronoi neighbours and area CV, Jablczynski P, DLA exponent). All are simulation-side; none is measured on a specimen, and there is no r / RMSE / KS / CV statistic comparing model to plate.  
  evidence: `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L308–342; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L308–342
- **Source and data traceability: 80** — `datasets/thompson_1917/t1917-023/` holds the scout manifest (IA SHA-1/SHA-256 digests, leaf-to-page rule, OCR quotes with page numbers), the reference JSON with every vendored file's SHA-256, a 21-entry BibTeX, and the Commons licence/author records for both analogue specimens; the script verifies the digests at runtime. Harting (1872) itself is cited through Thompson, not fetched.  
  evidence: `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L88–99; `datasets/thompson_1917/t1917-023/scout_manifest.json` L1–120
- **Code-to-model traceability: 70** — Module docstring (L1-33) and section headers map E1-E6 to Thompson page numbers and to the Lean module; function docstrings cite Zener / Mullins-Sekerka, Keller-Rubinow/Ostwald, Niemeyer et al. Figure numbers (195-197, 208) are pinned to pixel boxes (L55-61) but the link from each measured number back to a specific Thompson sentence is not yet explicit.  
  evidence: `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L1–33; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L55–61
- **Reproducibility package: 64** — Deterministic seeds (SEED = 1917; per-eta RNG), 8-check `run_self_test` (L436-459) passing in 2.3 s, full render in 6.4 s, documented invocation, `--quick` mode. Dependencies are the repo's requirements.txt but versions are not pinned in the script, and no result.json / verdict artefact is written.  
  evidence: `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L436–459; `thompson/claims/t1917-023_on-concretions-spicules-and-spicular-skeletons.py` L462–480

Remediation:
- Segment Thompson's plates: boundary-radius CV and radial-fibre alignment of the Fig. 196 spherite, nucleus detection + radial alignment in Fig. 195, cell neighbour count / area CV of the Fig. 197 mosaic, ring radii of Fig. 208.
- Analyse the two analogue specimens: ellipse fit, concentric ring count and radial-fibre alignment of the Carmel ooids; ring radii of the silver-chromate Liesegang specimen with geometric-vs-arithmetic spacing fits (r, RMSE) and Jablczynski P.
- Run the full eta sweep (0, 0.5, 1, 1.5, 2) and the Liesegang b0 sweep (0.05, 0.1, 0.2) to show monotone control of solidity / exponent and of P.
- Add negative / ablation controls: anisotropic (chessboard-metric) deposition that does not round off; threshold-free (K_ss = K_sp) precipitation that gives a continuous deposit instead of bands.
- Add goodness-of-fit statistics (Pearson r, RMSE, two-sample KS, CV) between simulated and specimen distributions, expose CLI parameter overrides, and write datasets/thompson_1917/t1917-023/result.json with the claim verdict.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-023 — calcospherites grow by radial deposition about a centre (Harting 1872 artificial CaCO3 concretions in egg albumen)
- **spec_fingerprint** (semantic model + inputs): `09c3115074722edd88f58e645bafec8f465e6fa5a4c47f5f66ece6a0c71a3acd`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-023/carmel_ooids_thin_section_specimen.jpg` (`3f1e500de229723d…`), `datasets/thompson_1917/t1917-023/liesegang_rings_silver_chromate_gelatine_specimen.jpg` (`2f41645ff68f6000…`), `datasets/thompson_1917/t1917-023/paper_citations.bib` (`7a05770584be0d5e…`), `datasets/thompson_1917/t1917-023/scout_manifest.json` (`89dbc647da5d51c9…`), `datasets/thompson_1917/t1917-023/t1917_023_on_concretions_spicules_and_sp_reference.json` (`23dafa48d0e5c552…`), `datasets/thompson_1917/t1917-023/thompson_1917_p421_figs195-196_harting_calcospherites.jpg` (`f05b4d1f9ce006b8…`), `datasets/thompson_1917/t1917-023/thompson_1917_p422_fig197_aggregated_calcospherites.jpg` (`65eabed4733a09dd…`), `datasets/thompson_1917/t1917-023/thompson_1917_p432_fig208-209_spherocrystal_otoliths.jpg` (`cbfffad55d3f82b7…`)
- `docs/media/t1917_023_on_concretions_spicules_and_sp_comparison.png` — artifact_sha256 `99d274958a29f4bd…`, 1,067,896 B, 2200×1100 px

| measurement | value | reference | method |
|---|---:|---|---|
| E3 asphericity decay exponent d ln A / d ln R (square nucleus, uniform normal deposition) | -0.9727  | -1.0 | log-log slope of (max rho - min rho)/mean rho vs mean R over distance-transform level sets |
| E1 lamella-thickness CV, interface-controlled law | 0.0  | 0.0 | std/mean of successive R_{n+1}-R_n |
| E4 mean neighbour count, equal-rate Voronoi mosaic | 5.7778 neighbours | 6.0 | interior cells of 60-centre Voronoi map |
| E4 cell-area CV, equal-rate Voronoi mosaic | 0.5609  | 0.529 | std/mean of interior cell areas |
| E5 Liesegang spacing coefficient P (2-D, a0=50, b0=0.1) | 1.2197  | 1.15 | mean r_{n+1}/r_n over precipitate band centres *(11 bands)* |
| E5 Liesegang band count (3-D spherical, a0=100) | 7.0 bands | — | precipitate maxima above 10% of peak |
| E6 Eden (eta=0) envelope radius CV | 0.1346  | — | 36-sector outer envelope std/mean |
| E6 DLA (eta=1) mass-radius exponent | 1.6347  | 1.71 | log N(r) vs log r fit, r <= 0.75 R_max |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- lean: 4.35.0

Pass 1 (v1) baseline evaluation of the Thompson (1917) claim-validation script before Step 12 closed-loop remediation in `2026-10-28-t1917-023-on-concretions-spicules-and-sp-loop`.

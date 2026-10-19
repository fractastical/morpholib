# MorphoEval report — audit `2026-10-18-phyllotaxis-audit`

Rubric **0.1.0** · checkpoint `c89d18d` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasProbability` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for ProbKnow template conformance). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Vogel (1979) & Douady–Couder (1992) Golden-Angle Phyllotaxis (`1979-vogel-phyllotaxis.py`) | 72 | 55 | 68 | 45 | 80 | 66 | 78 | 70.7 | 51.2 | 75.5 | 64.3 | **64.3** | mostly faithful |

## Findings and remediation

### Vogel (1979) & Douady–Couder (1992) Golden-Angle Phyllotaxis  —  overall 64.3 (mostly faithful)

Initial v1 implementation of Vogel (1979) golden-angle phyllotaxis and Douady & Couder (1992) Hofmeister inhibitory dynamics, paired with Swinton et al. (2016) Royal Society Helianthus annuus survey tables and two Wikimedia Commons specimen photographs. Independent audit by Claude Sonnet 5.5 High scored v1 at 64.3 overall, surfacing four concrete bugs and missing measurements targeted for Step 12 remediation.

- **Model fidelity to source: 72** — Vogel's (1979) planar generative spiral r_n = c*sqrt(n), theta_n = n*alpha_0 and Delaunay rising Fibonacci parastichy cascade (8,13)->(13,21)->(21,34)->(34,55) are implemented accurately. Douady & Couder's (1992) Hofmeister inhibitory energy minimization converges to 137.59 deg (Fibonacci) and 99.38 deg (Lucas), though the Lucas branch is seeded near its basin (99.8 deg) rather than emerging purely from plastochron bifurcation.  
  evidence: `1979-vogel-phyllotaxis.py` L109–155; `1979-vogel-phyllotaxis.py` L274–336
- **Morphological realism / specimen match: 55** — Two real CC-licensed Helianthus annuus capitulum photographs are vendored and displayed in Panel A with verified SHA-256 digests, and Swinton et al.'s (2016) 768 photoreviewed parastichy counts are plotted in Panel F. However, in v1 the specimen photograph is only shown via imshow() without pixel/Fourier parastichy extraction, and the simulated ensemble produces 0% Fib-1 and Fib+1 defects.  
  evidence: `1979-vogel-phyllotaxis.py` L552–581; `1979-vogel-phyllotaxis.py` L778–804; `vogel_phyllotaxis_comparison.png` L0–0
- **Parameter expressivity: 68** — Divergence angle alpha_deg, floret count n_florets, jugacy J, and plastochron schedule G_start/G_end are parameterized and swept across Panels B, C, D, and E. However, angular/radial noise sigma is not systematically swept and no parameter regime is shown generating the empirical Fib-1 > Fib+1 grain-defect asymmetry.  
  evidence: `1979-vogel-phyllotaxis.py` L659–686; `1979-vogel-phyllotaxis.py` L691–703
- **Morphometric validation: 45** — While Delaunay parastichy pairs and NND-CV behave properly near the golden angle, the independent audit discovered that compute_packing_metrics() clips Voronoi cell areas to 0.05 < area < 20.0, allowing degenerate rational 120 deg spokes to report a lower voronoi_cv_area (0.000314) than the golden angle (0.0445), and Panel C's CV labels were hardcoded strings inconsistent with the CSV.  
  evidence: `1979-vogel-phyllotaxis.py` L215–267; `vogel_phyllotaxis_metrics.csv` L0–0
- **Source and data traceability: 80** — DOIs for Vogel (1979), Douady & Couder (1992, 1996), Swinton et al. (2016), and Dryad (10.5061/dryad.f9k77) are recorded in phyllotaxis_reference.json and paper_citations.bib, and both Wikimedia Commons photographs have verified SHA-1 and SHA-256 digests. Historical surveys (Schoute 1938, Weisse 1897, Jean 1994) should also be added to paper_citations.bib.  
  evidence: `datasets/phyllotaxis/phyllotaxis_reference.json` L1–135; `datasets/phyllotaxis/paper_citations.bib` L1–41
- **Code-to-model traceability: 66** — Functions are clearly documented against paper sections, but the independent assessor caught three code-to-label mismatches: Panel E legend says initial angles 155/122/104 deg while code passes 175/130/99.8 deg; Panel C labels Sunflower 171 as (29,47) whereas JSON records (47,76); and Panel C CV labels were hardcoded rather than formatted from compute_packing_metrics().  
  evidence: `1979-vogel-phyllotaxis.py` L659–665; `1979-vogel-phyllotaxis.py` L729–738
- **Reproducibility package: 78** — Self-contained script with --self-test, deterministic RNG seeds, verified dataset image SHA-256 checks, static PNG, animated GIF, and CSV outputs. Can be improved by hashing phyllotaxis_reference.json and reporting output digests directly.  
  evidence: `1979-vogel-phyllotaxis.py` L375–474; `docs/media/vogel_phyllotaxis_growth.gif` L0–0

Remediation:
- Extract quantitative morphometrics (2D polar/radial Fourier parastichy spectrum and detected floret centroids) directly from the empirical specimen photograph (`helianthus_annuus_capitulum_specimen.jpg` and `helianthus_parastichy_annotated.jpg`) and compare against the Vogel model.
- Fix `compute_packing_metrics()` so degenerate rational angles (e.g. 120 deg) are penalized by bounding the Capitulum disk (or using bounded Voronoi / NND + angular coverage uniformity) so the golden angle is a true global optimum across [90 deg, 170 deg], and format Panel C CV labels dynamically from the computed metrics.
- Model and test the Swinton et al. (2016) Fib-1 > Fib+1 grain-defect asymmetry mechanistically (via plastochron/radial transition-zone noise on finite capitula where outer-rim parastichy transitions are Incomplete by 1 spiral more often than supernumerary by 1 spiral) and report chi-square / binomial agreement against Table 1.
- Fix all code-traceability mismatches found by the second assessor (Panel E legend initial angles, Sunflower 171 (47,76) outer zone, F4 vs Fib+1 classification of 14, and BibTeX entries for Schoute 1938, Weisse 1897, and Jean 1994).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Vogel (1979) & Douady–Couder (1992) Golden-Angle Phyllotaxis
- **spec_fingerprint** (semantic model + inputs): `5df8c2b4cc4b9b4a86ea26aace00ebd26df8456a62d99c23287ddcea2644e06c`
- **input_dataset_sha256**: `datasets/phyllotaxis/helianthus_annuus_capitulum_specimen.jpg` (`c76483eea3e564fe…`), `datasets/phyllotaxis/helianthus_parastichy_annotated.jpg` (`7cd34354e23584cb…`), `datasets/phyllotaxis/phyllotaxis_reference.json` (`432d1eeebf62bdc7…`)
- `docs/media/vogel_phyllotaxis_growth.gif` — artifact_sha256 `8e8d60ceff1e2574…`, 877,508 B, 1120×540 px, 27 frames
- `vogel_phyllotaxis_comparison.png` — artifact_sha256 `98e2578361a0d040…`, 2,155,516 B, 2960×1840 px
- `vogel_phyllotaxis_metrics.csv` — artifact_sha256 `6f49a262bb096f02…`, 317 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| golden divergence angle α₀ = 360°/τ² | 137.5078 degrees | 137.50776405003785 | exact golden-section divergence angle 360 * (2 - τ) |
| outer-annulus [14.5, 22.5) opposed contact parastichy ratio (n/m) | 1.6177 ratio (55/34) | 1.61765 | Delaunay contact parastichy pair (34, 55) in outer annulus r in [14.5, 22.5) *(full radial cascade: [(8, 13), (13, 21), (21, 34), (34, 55)])* |
| Voronoi cell area CV at golden angle (137.5078°) | 0.0072 CV (std/mean) | 0.009777076602327022 | Voronoi polygon area dispersion across interior florets (0.18 R_max <= r <= 0.85 R_max, N=450) *(strictly lower than α=137.30° (0.0098) and α=137.60° (0.0074))* |
| Douady & Couder (1992) Hofmeister main-branch emergent divergence angle | 137.586 degrees | 137.507764 | mean steady-state divergence angle over final 25 plastochrons on main Fibonacci branch (G: 0.55 -> 0.04) |
| Douady & Couder (1992) Hofmeister Lucas-branch emergent divergence angle | 99.38 degrees | 99.501553 | mean steady-state divergence angle over final 25 plastochrons on secondary Lucas branch (G: 0.22 -> 0.04) |
| Swinton et al. (2016) photoreviewed Fibonacci-structure prevalence | 82.29 percent | 82.29 | 632 of 768 photoreviewed Helianthus annuus parastichy counts belonging to Fibonacci, Lucas, Double-Fib, or F4 sequences *(Fib-1 (49) vs Fib+1 (17) one-sided binomial p = 5.07e-05)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Initial v1 reproduction of Vogel (1979) planar spiral phyllotaxis, Douady & Couder (1992, 1996) Hofmeister inhibitory dynamics, and Swinton, Ochu & MSI Turing's Sunflower Consortium (2016) Helianthus annuus parastichy survey.

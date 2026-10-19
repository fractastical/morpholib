# MorphoEval report — audit `2026-10-18-phyllotaxis-audit-v2`

Rubric **0.1.0** · checkpoint `c89d18d` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-18-phyllotaxis-audit -> 2026-10-18-phyllotaxis-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Vogel (1979) & Douady–Couder (1992) Golden-Angle Phyllotaxis (`1979-vogel-phyllotaxis.py`) | 86 | 84 | 82 | 86 | 90 | 88 | 88 | 84.7 | 84.8 | 88.7 | 85.9 | **85.9** | faithful |

## Findings and remediation

### Vogel (1979) & Douady–Couder (1992) Golden-Angle Phyllotaxis  —  overall 85.9 (faithful)

Remediated v2 reproduction of Vogel (1979) and Douady & Couder (1992, 1996) golden-angle phyllotaxis validated against Swinton et al. (2016) Royal Society Helianthus annuus parastichy data and direct Difference-of-Gaussians floret segmentation on a real capitulum photograph (621 segmented florets, 402 interior Voronoi cells, mean circularity 0.793 vs 0.853 model). All four defects surfaced in the v1 dual-agent audit are resolved and verified by automated assertions in `run_self_tests()`.

- **Model fidelity to source: 86** — Vogel's (1979) planar spiral law, rising Fibonacci parastichy cascade (8,13)->(13,21)->(21,34)->(34,55), Lucas (29,47)->(47,76) branch, and bijugate 2*(21,34) = (42,68) whorls are reproduced and verified by Delaunay contact-graph topology. Douady & Couder's (1992) Hofmeister repulsive energy minimization converges from disparate initial angles (175 deg and 130 deg -> 137.59 deg; 99.8 deg -> 99.38 deg), and the noisy rim-dislocation ensemble mechanistically links outer-rim contact chain loss to Swinton et al.'s (2016) Fib-1 > Fib+1 grain-defect bias.  
  evidence: `1979-vogel-phyllotaxis.py` L110–154; `1979-vogel-phyllotaxis.py` L383–441; `1979-vogel-phyllotaxis.py` L470–562
- **Morphological realism / specimen match: 84** — Panel A segments 621 florets directly from `helianthus_annuus_capitulum_specimen.jpg` via Difference-of-Gaussians filtering and measures 402 bounded interior Voronoi cells (mean circularity 0.7934 vs 0.8525 in the ideal Vogel lattice and 0.7045 in the 137.3 deg control). Panel F compares all 6 parastichy categories against Swinton et al. (2016) Table 1 (N=768 photoreviewed counts) and Schoute (1938, N=319), achieving Pearson r = 0.9950 and reproducing Fib-1 (68) > Fib+1 (22) (p = 6.25e-7).  
  evidence: `1979-vogel-phyllotaxis.py` L310–377; `1979-vogel-phyllotaxis.py` L754–799; `vogel_phyllotaxis_comparison.png` L0–0
- **Parameter expressivity: 82** — Sweeps divergence angle alpha across [136.6, 138.4] deg plus Lucas (99.502 deg), rational (120.0 deg), and bijugate (J=2) regimes, plastochron schedule G(n) across [0.65, 0.04], floret count N across [30, 600] in the growth animation, and angular/radial noise in the 384-head dislocation ensemble.  
  evidence: `1979-vogel-phyllotaxis.py` L856–922; `1979-vogel-phyllotaxis.py` L927–960
- **Morphometric validation: 86** — Bounded Voronoi polygon clipping with 72 outer guard vertices completely resolves the v1 unbounded-cell artifact: `rational_120.0` now scores `voronoi_cv_area = 0.6207` and `mean_circularity = 0.0284`, while the golden angle (`137.5078 deg`) simultaneously minimizes `voronoi_cv_area` (`0.007209`), minimizes `nnd_cv` (`0.011108`), and maximizes Voronoi circularity (`0.852513`). All 7 morphometrics are independently verified in `2026-10-18-phyllotaxis-audit-v2.measurements.yaml`.  
  evidence: `1979-vogel-phyllotaxis.py` L223–303; `1979-vogel-phyllotaxis.py` L634–654; `vogel_phyllotaxis_metrics.csv` L0–0
- **Source and data traceability: 90** — All primary and historical sources (Vogel 1979, Douady & Couder 1992/1996, Swinton et al. 2016, Dryad dataset 10.5061/dryad.f9k77, Schoute 1938, Weisse 1897, Jean 1994, and both Wikimedia Commons specimen photographs) are cited in `datasets/phyllotaxis/paper_citations.bib` and locked by SHA-256 digests verified on every run of `--self-test`.  
  evidence: `1979-vogel-phyllotaxis.py` L586–596; `datasets/phyllotaxis/paper_citations.bib` L1–88
- **Code-to-model traceability: 88** — Every code-to-label discrepancy noted by the second assessor in v1 has been eliminated: Panel C and Panel E format their labels directly from the computed dictionary values, Sunflower 171 is verified on its exact (47, 76) outer-zone Lucas pair, and `classify_parastichy_number()` cleanly separates `Fibonacci + 1` (14) from `F4` (9, 23, 37, 60, 97).  
  evidence: `1979-vogel-phyllotaxis.py` L614–626; `1979-vogel-phyllotaxis.py` L969–994
- **Reproducibility package: 88** — Deterministic RNG seeds (`1979`, `2016`, `42`), fast `--self-test` (<1.5 s), explicit virtualenv invocation instructions in docstring, SHA-256 locks on all input JSON/JPEG files, and SHA-256 digest reporting for all generated PNG, GIF, and CSV artifacts.  
  evidence: `1979-vogel-phyllotaxis.py` L1247–1277; `docs/media/vogel_phyllotaxis_growth.gif` L0–0

Remediation:
- Extend specimen image analysis from 2D planar projection to 3D dome surface profilometry when micro-CT capitulum scans become available.
- Couple the Douady–Couder Hofmeister potential directly to a continuum auxin-PIN1 reaction-advection-diffusion solver on the meristematic dome (Jönsson et al. 2006 / Newell et al. 2008).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Vogel (1979) & Douady–Couder (1992) Golden-Angle Phyllotaxis
- **spec_fingerprint** (semantic model + inputs): `5df8c2b4cc4b9b4a86ea26aace00ebd26df8456a62d99c23287ddcea2644e06c`
- **input_dataset_sha256**: `datasets/phyllotaxis/helianthus_annuus_capitulum_specimen.jpg` (`c76483eea3e564fe…`), `datasets/phyllotaxis/helianthus_parastichy_annotated.jpg` (`7cd34354e23584cb…`), `datasets/phyllotaxis/phyllotaxis_reference.json` (`432d1eeebf62bdc7…`)
- `docs/media/vogel_phyllotaxis_growth.gif` — artifact_sha256 `8e8d60ceff1e2574…`, 877,508 B, 1120×540 px, 27 frames
- `vogel_phyllotaxis_comparison.png` — artifact_sha256 `3ab57a6be31aab64…`, 2,217,939 B, 2960×1840 px
- `vogel_phyllotaxis_metrics.csv` — artifact_sha256 `cb2404142073f9a5…`, 449 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| golden divergence angle α₀ = 360°/τ² | 137.5078 degrees | 137.50776405003785 | exact golden-section divergence angle 360 * (2 - τ) |
| outer-annulus [14.5, 22.5) opposed contact parastichy ratio (n/m) | 1.6177 ratio (55/34) | 1.61765 | Delaunay contact parastichy pair (34, 55) in outer annulus r in [14.5, 22.5) *(full radial cascade: [(8, 13), (13, 21), (21, 34), (34, 55)])* |
| Voronoi cell area CV at golden angle (137.5078°) | 0.0072 CV (std/mean) | 0.009777076602327238 | Voronoi polygon area dispersion across interior florets (0.18 R_max <= r <= 0.85 R_max, N=450) *(strictly lower than α=137.30° (0.0098) and α=137.60° (0.0074))* |
| Douady & Couder (1992) Hofmeister main-branch emergent divergence angle | 137.586 degrees | 137.507764 | mean steady-state divergence angle over final 25 plastochrons on main Fibonacci branch (G: 0.55 -> 0.04) |
| Douady & Couder (1992) Hofmeister Lucas-branch emergent divergence angle | 99.38 degrees | 99.501553 | mean steady-state divergence angle over final 25 plastochrons on secondary Lucas branch (G: 0.22 -> 0.04) |
| Swinton et al. (2016) photoreviewed Fibonacci-structure prevalence | 82.29 percent | 82.29 | 632 of 768 photoreviewed Helianthus annuus parastichy counts belonging to Fibonacci, Lucas, Double-Fib, or F4 sequences *(Fib-1 (49) vs Fib+1 (17) one-sided binomial p = 5.07e-05)* |
| empirical specimen photograph segmented floret Voronoi circularity ⟨4πA/P²⟩ | 0.7934 ratio (4πA/P²) | 0.8525133755444986 | Difference-of-Gaussians floret segmentation (N=621 florets, 402 interior Voronoi cells) on helianthus_annuus_capitulum_specimen.jpg *(specimen Voronoi area CV = 0.2530, hex neighbor fraction = 0.377)* |
| mechanistic rim-dislocation ensemble Pearson r vs Swinton (2016) Table 1 | 0.995 Pearson r | 1.0 | correlation across 6 parastichy classes (N=768 simulated vs N=768 photoreviewed; Fib-1=68 > Fib+1=22, p=6.25e-07) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Remediated v2 reproduction of Vogel (1979) planar spiral phyllotaxis, Douady & Couder (1992, 1996) Hofmeister inhibitory dynamics, direct Difference-of-Gaussians floret segmentation on Helianthus annuus specimen photography, and mechanistic rim-dislocation modeling of Swinton et al. (2016) Table 1.

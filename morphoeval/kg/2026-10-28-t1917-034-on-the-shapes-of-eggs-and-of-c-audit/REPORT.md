# MorphoEval report — audit `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit`

Rubric **0.1.0** · checkpoint `83b41df2a` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. XV: eggs as solids of revolution of a definite, species-varying family of ovoid meridian curves (Preston 1953 family on Biggins et al. 2018 / Deeming 2018 data) (`thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`) | 62 | 42 | 50 | 58 | 70 | 66 | 60 | 58.0 | 48.0 | 64.9 | 56.1 | **56.1** | partially faithful |

## Findings and remediation

### Thompson (1917) Ch. XV: eggs as solids of revolution of a definite, species-varying family of ovoid meridian curves (Preston 1953 family on Biggins et al. 2018 / Deeming 2018 data)  —  overall 56.1 (partially faithful)

Pass 1 (v1) baseline evaluation of `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` (overall weighted score: 56.1/100). The v1 script implements the Preston (1953) ovoid meridian family with Biggins et al. (2018) shape indices, a closed-form polar-asymmetry index (r = 0.99997 against the published numeric index on 1389 eggs) and an exact volume functional, reproduces the published guillemot C126 coefficients, shows the 30-species elongation table, the 1389-egg morphospace and Thompson's relative-egg-size prediction (Deeming PC2 vs egg/female mass, Spearman rho = 0.391, p = 6.7e-8), and animates sphere -> guillemot within the family. It does not exercise Thompson's envelope equation in code, performs no parameter sweep, never segments the five vendored specimen photographs (so the solid-of-revolution part of the claim is untested), has no negative / ablation control (Steiner Cartesian oval, Baker / Troscianko / Carter alternatives), no eta^2 / KS statistics, no runtime SHA-256 verification, and no CLI parameter overrides.

- **Model fidelity to source: 62** — Implements the Preston family y = p(x) sqrt(1 - x^2), the Biggins indices and the closed-form pole radius / volume exactly as formalised in Lean Laws 1-2 (L46-81), but Thompson's own governing content - the envelope equation p_n + T(1/r + 1/r') = P, the pole-curvature closure of the family and the cone-ended-cylinder equilibrium (Lean Laws 3-4) - is not exercised in code, and no generative mechanical model produces the family from pressure asymmetry.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L46–81; `morphoeval/lean/loops/Thompson1917_T1917_034.lean` L40–191; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L224–258
- **Morphological realism / specimen match: 42** — The vendored specimen set (`biggins2018_C126_uria_aalge_silhouette.jpg` and four Museum Wiesbaden photographs of owl, grebe, lapwing and guillemot eggs) is SHA-256 pinned but never loaded; the figure shows only published fitted curves, so neither the axisymmetry (solid-of-revolution) test nor a photograph-to-family fit exists.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L160–222; `docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png` L0–0; `datasets/thompson_1917/t1917-034/t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json` L1–432
- **Parameter expressivity: 50** — Spans the published 10-egg range set and 1389 eggs of 30 species (L98-147), but there is no sweep over the model's own parameters (pole pressure contrast / c1, c3), no exploration of Thompson's spherical / ellipsoidal / cylindrical / ovoid / conical classes as regions of the family, and no CLI overrides (L310-318).  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L98–147; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L310–318
- **Morphometric validation: 58** — `measure_morphoeval()` (L264-292) returns 5 reference-anchored measurements (min elongation 1.035 >= 1; C126 elongation 1.8607 vs 92.5/49.8; closed-form PA r = 1.0000; species elongation range 0.43; Spearman rho = 0.391) but no RMSE / KS goodness-of-fit of the family against outlines, no between-species eta^2 or Kruskal-Wallis, and no negative-control delta.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L264–292
- **Source and data traceability: 70** — Reads the vendored CC0 Dryad tables (`biggins2018_*`, `deeming2018_*`) whose upstream and vendored SHA-256 digests are pinned in the reference JSON and scout manifest, and cites Thompson 1917 / Preston 1953 / Biggins 2018 / Deeming 2018 in the module header (L1-27), but performs no runtime digest verification and never opens the reference JSON (REF_JSON defined at L39, unused).  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L35–39; `datasets/thompson_1917/t1917-034/scout_manifest.json` L1–109
- **Code-to-model traceability: 66** — Module docstring cites the sources and Lean namespace, and the family block names the Lean definitions (L42-45), but individual functions do not map to Thompson's page-level statements (pp. 652, 655, 657-659) or to the Lean theorem names, and the Deeming PC axis identification is undocumented in code.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L1–45; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L160–222
- **Reproducibility package: 60** — `--self-test` (L295-307) checks sphere indices, the closed-form volume against numerical integration and the C126 indices against Biggins' values, and the run is deterministic (no RNG), but there is no dataset-integrity check, no two-run determinism assertion, no --verify-only / --skip-gif flags, and the GIF is 1.08 MB.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L295–307

Remediation:
- Exercise Thompson's envelope equation in code: implement the pole / equator curvature closure of the Preston family (given pole pressure contrast, equatorial pressure and volume solve for c0..c3), sweep the pole pressure contrast to generate sphere -> ovoid -> conical eggs, and compute the implied (P - p_n)/T mean-curvature profile for every Biggins egg (uniform pressure <=> sphere; pole tension sign).
- Segment the five vendored specimen photographs (Otsu / multi-Otsu threshold, convex hull, watershed), extract upper and lower meridian profiles, test axisymmetry (RMS upper-vs-lower / half-breadth), fit the Preston family by Biggins' weighted least squares and compare C126 against the published coefficients; place owl, grebe, lapwing and guillemot in the morphospace.
- Add negative / ablation controls: Steiner's Cartesian oval r + m r' = c on the guillemot outline (Thompson p. 654 footnote), prolate ellipsoid (c1 = c2 = c3 = 0) and Baker / Troscianko / Carter error ratios from biggins2018_comparison_of_fits.csv.
- Add statistics: Preston RMSE / breadth distribution, between-species eta^2 and Kruskal-Wallis for Elongation / Pointedness / PolarAsymmetry (30 species) and order-level eta^2 for Deeming PC1 / PC2, KS test of species elongation distributions (guillemot vs penguin), Pearson r of segmented vs published C126 profile.
- Add runtime SHA-256 verification of the 13 pinned dataset files, CLI overrides (--pole-contrast, --equator-contrast, --frames, --skip-gif, --verify-only), expand self_test() with integrity and two-run determinism checks, document page / theorem cross-references per function, and write datasets/thompson_1917/t1917-034/result.json.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. XV: eggs as solids of revolution of a definite, species-varying family of ovoid meridian curves (Preston 1953 family on Biggins et al. 2018 / Deeming 2018 data)
- **spec_fingerprint** (semantic model + inputs): `7f3cb7ceffd0305bc161370b241783acb708b5e9be3e83c757bf5c363fc7a7f4`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-034/biggins2018_C126_uria_aalge_silhouette.jpg` (`0a5c73b782104608…`), `datasets/thompson_1917/t1917-034/biggins2018_comparison_of_fits.csv` (`b2712450e2f2132e…`), `datasets/thompson_1917/t1917-034/biggins2018_data_for_comparing_indices.csv` (`39a776b94bad894a…`), `datasets/thompson_1917/t1917-034/biggins2018_range_of_egg_shapes.csv` (`3d0617dfd8cd0c20…`), `datasets/thompson_1917/t1917-034/deeming2018_jab_egg_shape_composition.csv` (`6367ba5252e52fcd…`), `datasets/thompson_1917/t1917-034/mwnh_0308_vanellus_vanellus_eggs.jpg` (`426b26bbad6c67d2…`), `datasets/thompson_1917/t1917-034/mwnh_0481_uria_aalge_eggs.jpg` (`fcaadba0ae209c4c…`), `datasets/thompson_1917/t1917-034/mwnh_0646_strix_aluco_egg.jpg` (`139f2249abb44e77…`), `datasets/thompson_1917/t1917-034/mwnh_1921_podiceps_cristatus_eggs.jpg` (`866eb96ea90025cc…`), `datasets/thompson_1917/t1917-034/paper_citations.bib` (`45e263a3bef66328…`), `datasets/thompson_1917/t1917-034/scout_manifest.json` (`83406c03aa7136ec…`), `datasets/thompson_1917/t1917-034/t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json` (`65a558575446b312…`)
- `docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png` — artifact_sha256 `6b6ab119d0c53b00…`, 301,794 B, 1540×1100 px

| measurement | value | reference | method |
|---|---:|---|---|
| minimum elongation L/B over 1399 eggs (prolate never oblate) | 1.03508405945417 ratio | 1.0 | min of Biggins et al. (2018) Elongation over 1389 main + 10 range eggs |
| C126 guillemot elongation from published Preston coefficients | 1.8607429282261319 ratio | 1.857429718875502 | 1/max y(x) of y = p(x) sqrt(1-x^2) |
| closed-form polar asymmetry vs Biggins numeric index (Pearson r, 1389 eggs) | 0.9999721506770683 r | 1.0 | R_pole = p(+/-1)^2 closed form (Lean Law 1) vs published PolarAsymmetry |
| species mean elongation range (max - min, 30 species) | 0.42995419192105877 ratio | — | species means of Biggins Elongation |
| Spearman rho: Deeming PC2 (asymmetry) vs log relative egg mass | 0.39114996513038175 rho | 0.0 | Spearman over n = 178 species, p = 6.72e-08 |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0
- lean: 4.35.0

Pass 1 (v1) evaluation of `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` prior to Step 12 closed-loop remediation in `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-loop`.

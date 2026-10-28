# MorphoEval report — audit `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2`

Rubric **0.1.0** · checkpoint `2514d4650` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib · supersession chain `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit -> 2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. XV: eggs as solids of revolution of a definite, species-varying family of ovoid meridian curves (Preston 1953 family on Biggins et al. 2018 / Deeming 2018 data) (`thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`) | 91 | 88 | 88 | 92 | 93 | 90 | 90 | 90.0 | 89.5 | 91.0 | 90.1 | **90.1** | faithful |

## Findings and remediation

### Thompson (1917) Ch. XV: eggs as solids of revolution of a definite, species-varying family of ovoid meridian curves (Preston 1953 family on Biggins et al. 2018 / Deeming 2018 data)  —  overall 90.1 (faithful)

Pass 2 (v2) evaluation of `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` (overall weighted score: 90.1/100, up from 56.1/100 in v1). The remediated script verifies the 13 pinned dataset files at runtime, inverts Thompson's envelope equation on every egg as the excess-pressure profile (P - p_n)/T = 2H(x), closes the Preston family from four curvature constraints (C126 recovered to 5.6e-16; sphere from uniform pressure), and drives a sphere -> ovoid -> conical pressure path. It segments 12 eggs of Thompson's owl, grebe, plover and guillemot classes from five photographs (axisymmetry residual 0.4 % on the back-lit C126 silhouette, museum median 4.2 %), re-fits the C126 photograph to c = (0.488, -0.245, 0.043, 0.117) vs published (0.488, -0.246, 0.039, 0.121) (E 1.8616 vs 1.8607, profile r = 0.99999), and shows Steiner's Cartesian oval 7.4x and the ellipsoid null 31x worse than Preston on the guillemot (Thompson's own footnote). Species signal: eta^2 = 0.67 / 0.83 / 0.47 for elongation / pointedness / polar asymmetry (p < 1e-92), order-level eta^2 0.38 / 0.57 (Deeming PC1 / PC2), KS D = 0.96; prolate-never-oblate holds (min E = 1.035 over 1411 eggs); asymmetry rises with relative egg mass (rho = 0.39, p = 7e-8). Verdict written to result.json: supported, with two flagged details (King Penguin is not spherical; Thompson's pole-tension remark has the sign reversed under his own equation).

- **Model fidelity to source: 91** — Implements the Preston family, indices and exact volume (L110-184) matching Lean Laws 1-2, and now exercises Thompson's own mechanics: the envelope equation inverted as 2H(x) = (P - p_n)/T with exact pole limits (L186-210), the four-constraint Thompson closure that recovers C126 and the sphere (L212-246), the empirical pressure-asymmetry path (L248-277) and the cone-balance ratio (L279-286), all cross-referenced to the 5 Lean theorems. The closure is kinematic (curvature <-> pressure) rather than a dynamic oviduct model with friction, which Thompson himself invokes.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L110–184; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L186–286; `morphoeval/lean/loops/Thompson1917_T1917_034.lean` L40–191; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L248–277
- **Morphological realism / specimen match: 88** — Segments all five vendored photographs (`_egg_candidates`, `segment_specimens`, L392-511) into 12 eggs spanning Thompson's four named classes, tests the solid-of- revolution property (C126 upper-vs-lower profile RMS 0.4 % of half-breadth; museum median 4.2 %, max 11.9 % on a shadowed lapwing egg), reproduces the published C126 coefficients from the raw silhouette (profile r = 0.99999) and recovers the ordering owl (E 1.27) < grebe (1.36-1.48) < guillemot (1.61 vs species mean 1.64). Polar asymmetry from hull fits of speckled eggs is inflated (3-8), so specimen PA is reported but not used for the verdict.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L392–511; `docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png` L0–0; `datasets/thompson_1917/t1917-034/result.json` L1–60; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L483–504
- **Parameter expressivity: 88** — Spans 1389 eggs of 30 species, the 10-egg range set, 181 Deeming species and 12 photographed eggs; sweeps the pole pressure contrast from the sphere to the 99th percentile (L248-277); exposes --pole-contrast, --equator-contrast, --x-star, --r-blunt, --frames, --skip-gif, --verify-only and --seed (L999-1030). Thompson's 'cylindrical' (megapode) class is represented only through the c2 sign, not by a dedicated specimen.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L248–277; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L999–1030
- **Morphometric validation: 92** — `measure_morphoeval()` (L816-871) returns 17 reference-anchored measurements: min elongation 1.035 (ref 1), axisymmetry 0.048 mean (ref 0), C126 photo E 1.8616 vs 1.8607 and PA 2.67 vs 2.63, profile r = 0.99999, Preston RMSE/B 0.0023 on C126 vs 0.0018 median, Cartesian oval 0.0172 (7.4x) and ellipsoid 0.0731 (31x) negative controls, Baker ratio 35x, eta^2 0.67 (p = 2e-100), order eta^2 0.38 / 0.57, KS D = 0.96 (p = 1e-135), class ordering, Spearman rho 0.39 (p = 7e-8), closure residual 5.6e-16, pressure-path r = 0.54 and CV of the pressure profile 0.32.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L816–871; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L559–633
- **Source and data traceability: 93** — `verify_dataset_integrity()` (L83-105) re-hashes the 13 pinned data files from the reference JSON at import of the analysis (and via --verify-only / self-test); Dryad DOIs, file ids, upstream SHA-256 digests, Commons authors and licences are in scout_manifest.json / reference JSON / README, and result.json repeats the dataset pins. The README hash is pinned but excluded from the runtime check as prose.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L83–105; `datasets/thompson_1917/t1917-034/t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json` L1–440
- **Code-to-model traceability: 90** — The module header maps each section to Thompson's pages (652, 654, 655, 657-659) and to the five Lean theorems (L1-57); function docstrings cite Biggins Analysis.R (estimator, fat.point), Lean law numbers and Thompson page numbers; the Math KG YAML links each law to code loci and Lean loci (alignment and integrity gates pass).  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L1–57; `morphoeval/math_kg_loops/thompson1917-t1917-034.yaml` L1–125
- **Reproducibility package: 90** — 20-assert `self_test()` (L956-996) covers dataset integrity, sphere / volume / reflection identities, Thompson closure of the sphere and C126, mean curvature of the sphere, cone- balance sign, specimen reproduction of C126, axisymmetry, the Cartesian-oval control and two-run bitwise determinism; --verify-only and --skip-gif flags; deterministic (no RNG); result.json written each run. Runtime is ~40 s because segmentation is recomputed per process.  
  evidence: `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L956–996; `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L874–954

Remediation:
- Replace the kinematic Thompson closure with a dynamic axisymmetric membrane in a peristaltic tube (wall friction + tangential tension gradient, Thompson pp. 657-659) so that the pressure-asymmetry path is solved rather than regressed.
- Photograph or source back-lit silhouettes of owl, grebe and plover eggs so polar asymmetry and axisymmetry are measured under the same controlled conditions as C126.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. XV: eggs as solids of revolution of a definite, species-varying family of ovoid meridian curves (Preston 1953 family on Biggins et al. 2018 / Deeming 2018 data)
- **spec_fingerprint** (semantic model + inputs): `16220681983e59a7518d8fd164735cf554e4313c23baed4858dbc080073234a4`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-034/biggins2018_C126_uria_aalge_silhouette.jpg` (`0a5c73b782104608…`), `datasets/thompson_1917/t1917-034/biggins2018_comparison_of_fits.csv` (`b2712450e2f2132e…`), `datasets/thompson_1917/t1917-034/biggins2018_data_for_comparing_indices.csv` (`39a776b94bad894a…`), `datasets/thompson_1917/t1917-034/biggins2018_range_of_egg_shapes.csv` (`3d0617dfd8cd0c20…`), `datasets/thompson_1917/t1917-034/deeming2018_jab_egg_shape_composition.csv` (`6367ba5252e52fcd…`), `datasets/thompson_1917/t1917-034/mwnh_0308_vanellus_vanellus_eggs.jpg` (`426b26bbad6c67d2…`), `datasets/thompson_1917/t1917-034/mwnh_0481_uria_aalge_eggs.jpg` (`fcaadba0ae209c4c…`), `datasets/thompson_1917/t1917-034/mwnh_0646_strix_aluco_egg.jpg` (`139f2249abb44e77…`), `datasets/thompson_1917/t1917-034/mwnh_1921_podiceps_cristatus_eggs.jpg` (`866eb96ea90025cc…`), `datasets/thompson_1917/t1917-034/paper_citations.bib` (`45e263a3bef66328…`), `datasets/thompson_1917/t1917-034/scout_manifest.json` (`83406c03aa7136ec…`), `datasets/thompson_1917/t1917-034/t1917_034_on_the_shapes_of_eggs_and_of_c_reference.json` (`27f65f272430a55c…`)
- `docs/media/t1917_034_on_the_shapes_of_eggs_and_of_c_comparison.png` — artifact_sha256 `d8c69f3b02b18347…`, 974,275 B, 1694×1346 px

| measurement | value | reference | method |
|---|---:|---|---|
| minimum elongation L/B over all eggs incl. segmented specimens (prolate never oblate) | 1.03508405945417 ratio | 1.0 | min over 1411 eggs (1389 Biggins + 10 range + 12 photographed); Thompson p. 652 null: oblate (E < 1) never occurs |
| axisymmetry residual RMS(upper - lower profile) / half-breadth, mean over segmented eggs | 0.0480223368747168 ratio | 0.0 | CV segmentation of 12 eggs in 5 photographs; std = 0.0285, max = 0.1186; solid-of-revolution test |
| C126 elongation from photograph segmentation vs published | 1.8615614438742696 ratio | 1.86074292548164 | Preston weighted LS on segmented silhouette (Biggins Analysis.R estimator) |
| C126 polar asymmetry from photograph vs published | 2.6698421705308917 ratio | 2.63450613688513 | closed form R_pole = p(+/-1)^2 on CV-fitted coefficients |
| C126 meridian profile Pearson r (segmented-fit vs published-fit) | 0.9999885635694227 r | 1.0 | 400-point y(x) profiles |
| Preston family RMSE / breadth on C126 outline | 0.002339414217032747 ratio | 0.00176654246895898 | geometric RMSE of segmented outline about the fitted Preston meridian |
| Steiner Cartesian oval RMSE / breadth on C126 (negative control, Thompson p. 654 footnote) | 0.017242702585420884 ratio | 0.002339414217032747 | r + m r' = c least squares; ratio vs Preston = 7.4x |
| prolate ellipsoid ablation RMSE / breadth on C126 (c1 = c2 = c3 = 0 null) | 0.07313691960230821 ratio | 0.002339414217032747 | ratio vs Preston = 31.3x |
| Baker (2002) 2-parameter alternative: median MSE ratio vs Preston (132 eggs) | 35.3165854333582 ratio | 1.0 | published cf.fitsB column; Troscianko ratio = 1.66, Carter ratio = 81.3 |
| between-species eta^2 of elongation (30 species) | 0.6667677204190067 fraction | 0.0 | one-way ANOVA eta^2; Kruskal-Wallis p = 2.4e-100; eta^2(polar asymmetry) = 0.471 (p = 4.1e-93) |
| order-level eta^2 of Deeming outline PC1 (elongation axis, 181 species) | 0.3840048266750948 fraction | 0.0 | orders with n >= 3; Kruskal-Wallis p = 5.6e-09; eta^2(PC2) = 0.574 |
| KS statistic: Uria aalge vs penguin elongation distributions | 0.9601636458779316 D | 0.0 | two-sample KS, p = 1.3e-135 |
| specimen ordering owl < grebe < guillemot elongation (Thompson p. 652 classes) | 1.0 bool | 1.0 | photo elongations: owl 1.269, grebe 1.428, guillemot 1.612 |
| Spearman rho: Deeming PC2 (asymmetry) vs log relative egg mass | 0.39114996513038175 rho | 0.0 | n = 178 species, p = 6.72e-08; Biggins species-level rho = 0.217 (p = 2.67e-01) |
| Thompson closure residual recovering C126 from its pressure descriptors | 5.551115123125783e-16 max abs residual | 0.0 | least squares on (delta_pole, delta_eq, x_star, r_blunt); sphere residual = 0.0e+00 |
| Thompson pressure path vs 30 species means: Pearson r of elongation at matched polar asymmetry | 0.5382957006738203 r | 1.0 | RMSE = 0.140; max closure residual along the path = 1.3e-15 |
| mean CV of the (P - p_n)/T profile over Biggins eggs (pressure non-uniformity; sphere = 0) | 0.3213574933955764 CV | 0.0 | std/mean of 2H(x) on 97 points, every 7th egg |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0
- lean: 4.35.0

Pass 2 (v2) evaluation of `thompson/claims/t1917-034_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` after completing Step 12 closed-loop remediation in `2026-10-28-t1917-034-on-the-shapes-of-eggs-and-of-c-loop`.

# MorphoEval report — audit `2026-10-18-venation-audit`

Rubric **0.1.0** · checkpoint `c89d18d` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasProbability` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for ProbKnow template conformance). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Runions et al. (2005) Leaf Venation Space-Colonization & Murray's Law (`2005-runions-leaf-venation.py`) | 45 | 20 | 40 | 30 | 68 | 62 | 55 | 43.3 | 23.8 | 61.2 | 40.9 | **40.9** | partially faithful |

## Findings and remediation

### Runions et al. (2005) Leaf Venation Space-Colonization & Murray's Law  —  overall 40.9 (partially faithful)

Initial v1 implementation of Runions et al. (2005) leaf venation space colonization, Murray (1926) / Price et al. (2013) vascular scaling, and Katifori & Magnasco (2012) loop nesting. Independent audit by Claude Sonnet 5.5 High scored v1 at 41.6 overall, identifying five concrete defects (unsegmented specimen images, triangle short-circuiting in closed anastomosis, synthetic 8x8 Katifori grid, circular Strahler sampling, and vacuous Lean 4 definitions) targeted for Step 12 remediation.

- **Model fidelity to source: 45** — Open space-colonization (Eq. 1) grows a valid dichotomous Ginkgo tree, but closed reticulate venation (L329-340) fuses new tips to any node within 1.32*d_k except the immediate parent rather than implementing Runions et al. (2005) Sec. 4 persistent V(s) source tagging, producing 571 degenerate triangles and 375 crossing edge pairs.  
  evidence: `2005-runions-leaf-venation.py` L279–344
- **Morphological realism / specimen match: 20** — Both specimen photographs have verified SHA-256 locks, but ginkgo_open_venation_specimen.jpg is never loaded or displayed and reticulate_vein_skeleton_specimen.jpg is only shown via imshow() with zero pixel segmentation, skeletonization, or empirical graph extraction.  
  evidence: `2005-runions-leaf-venation.py` L72–74; `2005-runions-leaf-venation.py` L708–718
- **Parameter expressivity: 40** — Runions Table 1 (15 parameter regimes) is vendored in venation_reference.json, but simulate_venation() hardcodes step_D, b_s, b_v, and d_k per mode without reading Table 1 or sweeping parameters.  
  evidence: `2005-runions-leaf-venation.py` L215–241
- **Morphometric validation: 30** — In v1, simulate_hierarchical_strahler_tree() samples directly from the JSON gamma_k column and correlates against the same JSON medians (circular r=0.9956), and compute_katifori_loop_nesting() runs on a synthetic 8x8 lattice rather than the simulated or segmented venation graph.  
  evidence: `2005-runions-leaf-venation.py` L518–560; `2005-runions-leaf-venation.py` L566–649
- **Source and data traceability: 68** — DOIs, BibTeX entries, and SHA-256 digests are complete for Runions (2005), Murray (1926), Price (2013), Katifori (2012), and Ronellenfitsch (2015), though venation_reference.json says '14 canonical regimes' in the role description while listing 15 regimes.  
  evidence: `datasets/venation/venation_reference.json` L1–165; `datasets/venation/paper_citations.bib` L1–55
- **Code-to-model traceability: 62** — Functions and Lean 4 theorems are cited in docstrings, but Lean Section 13 defines murray_root_capacity and murray_tip_sum identically and compute_murray_radii_and_strahler() has a 3-tuple return type annotation while returning a 4-tuple.  
  evidence: `morphoeval/lean/MathOfBiology.lean` L365–413; `2005-runions-leaf-venation.py` L397–461
- **Reproducibility package: 55** — Deterministic RNG seeds, SHA-256 checks, and --self-test are provided, but the unused scipy.spatial.Delaunay import and incomplete 8-row CSV limit standalone verification.  
  evidence: `2005-runions-leaf-venation.py` L67–67; `2005-runions-leaf-venation.py` L885–902

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Runions et al. (2005) Leaf Venation Space-Colonization & Murray's Law
- **spec_fingerprint** (semantic model + inputs): `9b9080151e73e5b8bf2e2b214040b93303d1faaea0b8391fd0aa193cc0743686`
- **input_dataset_sha256**: `datasets/venation/ginkgo_open_venation_specimen.jpg` (`5a620943881fe698…`), `datasets/venation/reticulate_vein_skeleton_specimen.jpg` (`6d06d3e545bd585b…`), `datasets/venation/venation_reference.json` (`4edb5b774f01935b…`)
- `runions_leaf_venation_comparison.png` — artifact_sha256 `79a46c05f0cf7c61…`, 1,136,399 B, 2475×1530 px

| measurement | value | reference | method |
|---|---:|---|---|
| open Ginkgo venation anastomosing cycle count L = E - V + 1 | 0 loops | 0 | Euler-Poincaré independent cycle rank E - V + 1 on open Voronoi space-colonization tree |
| Murray's cube law root-to-tip conservation ratio r_root^3 / (N_tips * r_tip^3) | 1.0 ratio | 1.0 | ratio of petiole root cubic radius r_root^3 to sum of terminal veinlet cubic radii sum r_tip^3 *(r_root^3 = 397.0, N_tips = 397)* |
| open venation binary junction median daughter/parent area ratio (A_1 + A_2)/A_0 | 1.2223 area ratio | 1.2599 | median (r_1^2 + r_2^2) / r_0^2 across binary bifurcations of the grown open venation tree |
| closed reticulate venation anastomosing loop count L = E - V + 1 | 902 areole loops | 150 | Euler cycle rank E - V + 1 of closed Relative Neighborhood Graph (RNG) space-colonization network *(V = 1056, E = 1957)* |
| Horton-Strahler order 1 vs order 8 median area-ratio shift | 0.2489 delta area ratio | 0.24 | decrease in median daughter/parent cross-sectional area ratio from terminal minor veins (order 1) to major midveins (order 8) *(Pearson r across Strahler orders = 0.9956)* |
| Katifori & Magnasco (2012) unweighted hierarchical loop nesting number i_u | 0.8234 nesting index i_u | 0.68 | mean subtree partition balance q_j = 1 - |r_j - s_j|/(r_j + s_j - 1) under iterative minimum-width vein removal *(high-noise / additive control i_u = 0.6155)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Initial v1 reproduction of Runions et al. (2005) space-colonization leaf venation, Murray (1926) / Price et al. (2013) vascular branching laws, and Katifori & Magnasco (2012) / Ronellenfitsch et al. (2015) hierarchical loop nesting.

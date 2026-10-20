# MorphoEval report — audit `2026-10-19-social-wasp-comb-hexagons-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/96

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Karsai & Pénzes (1993) J. Theor. Biol., Theraulaz & Bonabeau (1995) Science & Smith et al. (2021) PNAS stigmergic hexagonal wasp comb architecture, shared-wall pulp economy & 5-7 topological disclination pairs (`1998-bonabeau-social-wasp-comb-hexagons.py`) | 57 | 35 | 50 | 52 | 74 | 68 | 64 | 54.7 | 41.4 | 68.4 | 53.5 | **53.5** | partially faithful |

## Findings and remediation

### Karsai & Pénzes (1993) J. Theor. Biol., Theraulaz & Bonabeau (1995) Science & Smith et al. (2021) PNAS stigmergic hexagonal wasp comb architecture, shared-wall pulp economy & 5-7 topological disclination pairs  —  overall 53.5 (partially faithful)

Pass 1 (v1) baseline evaluation of `1998-bonabeau-social-wasp-comb-hexagons.py` (overall weighted score: 56.2/100). The v1 script implements the analytic centered hexagonal ring progression N(R) = 1 + 3R(R+1), total wall count W(R) = 3N(R) + 6R + 3, and wall-to-cell ratio W(R)/N(R) -> 3.0 across R in [0, 7], but generates cells via deterministic ringenumeration rather than simulating the stochastic Boltzmann stigmergic initiation rule P_init(k) ~ exp(beta*k) over k in {1, 2, 3} shared-wall perimeter pockets, omits the 7-point beta selectivity sweep and beta = 0 Eden random growth negative control, omits the Smith et al. (2021) worker-to-queen cell size transition inducing paired 5-7 topological defects, and never loads or segments the vendored Polistes wasp comb specimen (`polistes_wasp_comb_specimen.jpg`).

- **Model fidelity to source: 57** — Computes centered hexagonal cell counts N(R) = 1 + 3R(R+1), total shared walls W(R) = 3N(R) + 6R + 3, and pulp savings S_saved(R) = 6N(R) - W(R) (L34-53), backed by 3 Lean 4 theorems in `Loop_096_social_wasp_comb_hexagons.lean`. However, `generate_deterministic_hex_rings` (L56-68) enumerates concentric rings deterministically without simulating the stochastic Boltzmann wall-sharing rule P(k) ~ exp(beta*k), and lacks topological 5-7 disclination pairs at caste size transitions.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L34–53; `1998-bonabeau-social-wasp-comb-hexagons.py` L56–68
- **Morphological realism / specimen match: 35** — Although `datasets/social_wasp_comb_hexagons/polistes_wasp_comb_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or extracts brood cell cavities, Delaunay/Voronoi neighbor coordination numbers z_i, or hexatic order from the biological specimen image.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L71–86; `docs/media/social_wasp_comb_hexagons_comparison.png` L0–0
- **Parameter expressivity: 50** — Evaluates 8 concentric ring indices R in [0, 7] (L71-86), but omits the 7-point stigmergic selectivity sweep beta in [0.0, 5.5], omits the beta = 0 Eden random comb negative control, omits the worker-to-reproductive cell size ratio lambda in [1.00, 1.40] control, and exposes no CLI parameter overrides.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L71–86
- **Morphometric validation: 52** — `measure_morphoeval()` (L175-210) reports only 4 analytic ring ratios without Pearson r, p-value, RMSE, stigmergic selectivity beta correlation, Eden random vs wild-type wall-per-cell delta, 5-7 disclination dipole net Euler charge, or specimen brood cell coordination number <z>.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L175–210
- **Source and data traceability: 74** — Vendors `datasets/social_wasp_comb_hexagons/` with 5 primary DOIs in `paper_citations.bib` (`10.1006/jtbi.1993.1072`, `10.1126/science.269.5224.686`, `10.1093/beheco/11.4.387`, `10.1073/pnas.2103605118`, `10.1007/s004540010071`), `scout_manifest.json`, and `social_wasp_comb_hexagons_reference.json`, but lacks runtime SHA-256 verification in `1998-bonabeau-social-wasp-comb-hexagons.py`.  
  evidence: `datasets/social_wasp_comb_hexagons/social_wasp_comb_hexagons_reference.json` L1–165
- **Code-to-model traceability: 68** — Links module docstrings to Karsai & Pénzes (1993), Theraulaz & Bonabeau (1995), and the 3 Lean 4 theorems in `Loop_096_social_wasp_comb_hexagons.lean` (L1-11), but lacks trace links for stochastic Boltzmann pocket selection, Smith et al. (2021) 5-7 disclination pairs, and Eden random growth controls.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L1–11
- **Reproducibility package: 64** — Provides `--self-test` (`L213-218`) checking centered hexagonal numbers, but lacks SHA-256 asset verification, two-run bitwise determinism checks on the stochastic stigmergic builder, and topological defect assertions.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L213–218

Remediation:
- Replace deterministic ring enumeration with a stochastic Karsai-Penzes & Theraulaz-Bonabeau 2D/3D hexagonal lattice builder implementing Boltzmann stigmergic pocket selection P_init(j) = exp(beta * k_j) / sum_m exp(beta * k_m) over k in {1, 2, 3} shared-wall perimeter pockets.
- Implement both the 8-ring concentric comb economy trajectory R in [0, 7] (verifying W(R)/N(R) -> 3.0 and 50% pulp savings) and the 7-point stigmergic selectivity sweep beta in [0.0, 5.5] including the beta = 0.0 Eden random branched comb negative control.
- Implement the Smith et al. (2021) PNAS worker-to-reproductive cell size transition (lambda in [1.00, 1.40]) with spring-Voronoi lattice relaxation inducing paired pentagon-heptagon (5-7) topological disclinations with zero net Euler charge.
- Segment the vendored CC BY 2.0 Polistes paper wasp comb photograph (`polistes_wasp_comb_specimen.jpg`) via Difference-of-Gaussians cavity detection and Delaunay/Voronoi topology to extract brood cell count and interior coordination number <z>, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Karsai & Pénzes (1993) J. Theor. Biol., Theraulaz & Bonabeau (1995) Science & Smith et al. (2021) PNAS stigmergic hexagonal wasp comb architecture, shared-wall pulp economy & 5-7 topological disclination pairs
- **spec_fingerprint** (semantic model + inputs): `6471896f9ec341c70d5d90fdfc768c186b2e5e7ed95f0e8568fe6372d790ffc3`
- **input_dataset_sha256**: `datasets/social_wasp_comb_hexagons/paper_citations.bib` (`9b09023aaea52aa7…`), `datasets/social_wasp_comb_hexagons/polistes_wasp_comb_specimen.jpg` (`03befd4d1fbfa35a…`), `datasets/social_wasp_comb_hexagons/scout_manifest.json` (`63f51b932c469536…`), `datasets/social_wasp_comb_hexagons/social_wasp_comb_hexagons_reference.json` (`5d67bfb268fa3003…`)
- `docs/media/social_wasp_comb_hexagons_comparison.png` — artifact_sha256 `75731e309cf80570…`, 258,873 B, 1890×1400 px

| measurement | value | reference | method |
|---|---:|---|---|
| Founder cell (R = 0) initial wall-to-cell ratio W(0)/N(0) | 6.0 walls/cell | 6.0 | Analytic centered hexagonal comb formula at R = 0 |
| Intermediate comb (R = 3, N = 37) wall-to-cell ratio W(3)/N(3) | 3.5676 walls/cell | 3.5676 | Analytic centered hexagonal comb formula at R = 3 |
| Mature 6-ring comb (N = 127) wall-to-cell ratio W(6)/N(6) | 3.3071 walls/cell | 3.3071 | Analytic centered hexagonal comb formula at R = 6 |
| Mature 7-ring comb (N = 169) paper pulp wall savings percentage | 45.56 percent | 45.56 | Analytic pulp savings 100*(6*N - W)/(6*N) at R = 7 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `1998-bonabeau-social-wasp-comb-hexagons.py` prior to Step 12 closed-loop remediation in `2026-10-19-social-wasp-comb-hexagons-loop`.

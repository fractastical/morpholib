# MorphoEval report — audit `2026-10-19-social-wasp-comb-hexagons-audit-v2`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/96 · supersession chain `2026-10-19-social-wasp-comb-hexagons-audit -> 2026-10-19-social-wasp-comb-hexagons-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Karsai & Pénzes (1993) J. Theor. Biol., Theraulaz & Bonabeau (1995) Science & Smith et al. (2021) PNAS stigmergic hexagonal wasp comb architecture, shared-wall pulp economy & 5-7 topological disclination pairs (`1998-bonabeau-social-wasp-comb-hexagons.py`) | 95 | 93 | 94 | 94 | 95 | 93 | 94 | 94.7 | 93.4 | 94.1 | 94.0 | **94.0** | faithful |

## Findings and remediation

### Karsai & Pénzes (1993) J. Theor. Biol., Theraulaz & Bonabeau (1995) Science & Smith et al. (2021) PNAS stigmergic hexagonal wasp comb architecture, shared-wall pulp economy & 5-7 topological disclination pairs  —  overall 94.0 (faithful)

Pass 2 (v2) evaluation of `1998-bonabeau-social-wasp-comb-hexagons.py` (overall weighted score: 93.9/100, up from 56.2/100 in v1). The remediated script simulates stochastic Boltzmann stigmergic cell initiation P_init(j) ~ exp(beta*k_j) on a 2D/3D hexagonal lattice; validates the 8-ring centered hexagonal economy trajectory R in [0, 7] (Pearson r = 0.9997, p = 4.43e-11, RMSE = 0.0212 walls/cell, 44.88% pulp savings at R = 6) and the 7-point stigmergic selectivity sweep beta in [0.0, 5.5] (Pearson r = 0.9919, RMSE = 0.0487 walls/cell), including the beta = 0.0 Eden random branched comb negative control; models the Smith et al. (2021) worker-to-reproductive cell size step (lambda in [1.00, 1.40]) inducing paired 5-7 topological disclination dipoles with zero net Euler charge; and segments the SHA-256-locked CC BY 2.0 Polistes paper wasp comb specimen (`polistes_wasp_comb_specimen.jpg`, n = 169 brood cells, interior coordination <z_int> = 5.862 ± 0.893 vs hexagonal 6.0).

- **Model fidelity to source: 95** — Implements centered hexagonal ring progression and shared-wall economy (`compute_centered_hex_economy`, L126-152), a stochastic Boltzmann stigmergic 2D/3D hexagonal lattice builder (`simulate_stigmergic_wasp_comb`, L173-277), and worker-to-reproductive cell size transitions with paired 5-7 topological disclination dipoles (`simulate_caste_transition_disclinations`, L377-464), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_096_social_wasp_comb_hexagons.lean`.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L126–277; `1998-bonabeau-social-wasp-comb-hexagons.py` L377–464; `morphoeval/lean/loops/Loop_096_social_wasp_comb_hexagons.lean` L16–115
- **Morphological realism / specimen match: 93** — Vendors and SHA-256-verifies the CC BY 2.0 Polistes paper wasp comb photograph `datasets/social_wasp_comb_hexagons/polistes_wasp_comb_specimen.jpg`, segments n = 169 brood cell cavities via Difference-of-Gaussians filtering, and constructs the Delaunay neighbor graph with empirical interior coordination number <z_int> = 5.862 ± 0.893 (matching hexagonal coordination z = 6.0) and nearest-neighbor spacing 16.55 ± 4.26 px (`segment_polistes_comb_specimen`, L508-578).  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L508–578; `docs/media/social_wasp_comb_hexagons_comparison.png` L0–0
- **Parameter expressivity: 94** — Spans the 8-ring centered hexagonal economy progression R in [0, 7] (L283-319), the 7-point stigmergic selectivity sweep beta in [0.0, 5.5] across 6 Monte Carlo replicates per point (L322-371), 5 topological and ablation controls including lambda in [1.00, 1.40] and beta = 0.0 Eden random growth (L467-502), and exposes `--beta`, `--n-cells`, `--size-ratio-lambda`, and `--seed` CLI overrides (L665-691).  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L283–502; `1998-bonabeau-social-wasp-comb-hexagons.py` L665–691
- **Morphometric validation: 94** — `measure_morphoeval()` (L589-684) computes 9 reference-anchored morphometric measurements with uncertainty and statistical tests: 8-ring W(R)/N(R) Pearson r = 0.9997 (p = 4.43e-11, RMSE = 0.0212), 7-point beta sweep Pearson r = 0.9919 (p = 1.12e-5, RMSE = 0.0487), WT 6-ring W/N = 3.315 (ref 3.3071), 6-ring pulp savings 44.88% (ref 44.88%), Eden random negative control W/N = 4.252 (ref 4.18), lambda = 1.30 caste step 5-7 defect count 6 pairs (ref 6), net Euler topological charge 0 (ref 0), specimen interior coordination <z_int> = 5.862 ± 0.893 (ref 6.0), and specimen segmented brood cell count 169 (ref 169).  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L589–684
- **Source and data traceability: 95** — Vendors `datasets/social_wasp_comb_hexagons/` with 5 primary DOIs in `paper_citations.bib` (`10.1006/jtbi.1993.1072`, `10.1126/science.269.5224.686`, `10.1093/beheco/11.4.387`, `10.1073/pnas.2103605118`, `10.1007/s004540010071`), `scout_manifest.json`, `social_wasp_comb_hexagons_reference.json`, and `polistes_wasp_comb_specimen.jpg`, all verified at runtime via SHA-256 checks (`verify_dataset_integrity`, L90-108).  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L90–108; `datasets/social_wasp_comb_hexagons/social_wasp_comb_hexagons_reference.json` L1–165
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation and control to Karsai & Pénzes (1993, 2000), Theraulaz & Bonabeau (1995), Smith et al. (2021) PNAS, Hales (2001), and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_096_social_wasp_comb_hexagons.lean` (L1-40, L126-464).  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L1–40
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L690-748`) verifying SHA-256 dataset integrity, centered hexagonal ring & wall-sharing invariants, 8-ring W/N correlation, 7-point beta selectivity sweep & Eden negative control, 5-7 disclination Euler charge neutrality, specimen CV brood cell segmentation, and two-run bitwise determinism.  
  evidence: `1998-bonabeau-social-wasp-comb-hexagons.py` L690–748

Remediation:
- Couple the 2D/3D stigmergic hexagonal lattice builder to a continuous elasto-capillary pulp meniscus model resolving corner rounding during fresh pulp drying.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Karsai & Pénzes (1993) J. Theor. Biol., Theraulaz & Bonabeau (1995) Science & Smith et al. (2021) PNAS stigmergic hexagonal wasp comb architecture, shared-wall pulp economy & 5-7 topological disclination pairs
- **spec_fingerprint** (semantic model + inputs): `6471896f9ec341c70d5d90fdfc768c186b2e5e7ed95f0e8568fe6372d790ffc3`
- **input_dataset_sha256**: `datasets/social_wasp_comb_hexagons/paper_citations.bib` (`9b09023aaea52aa7…`), `datasets/social_wasp_comb_hexagons/polistes_wasp_comb_specimen.jpg` (`03befd4d1fbfa35a…`), `datasets/social_wasp_comb_hexagons/scout_manifest.json` (`63f51b932c469536…`), `datasets/social_wasp_comb_hexagons/social_wasp_comb_hexagons_reference.json` (`5d67bfb268fa3003…`)
- `docs/media/social_wasp_comb_hexagons_comparison.png` — artifact_sha256 `83089887ecb6d254…`, 1,143,280 B, 2775×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| Concentric ring progression (R = 0..7) wall-to-cell ratio W(R)/N(R) correlation | 0.9997 Pearson r | 1.0 | Pearson correlation between simulated stigmergic W(R)/N(R) (beta=5.5) and empirical reference across 8 concentric rings N in [1, 169] (p=4.43e-11, RMSE=0.0212) *(RMSE = 0.0212 walls/cell, p = 4.43e-11)* |
| Stigmergic selectivity sweep (beta = 0.0..5.5) wall-to-cell ratio correlation | 0.9919 Pearson r | 1.0 | Pearson correlation between Monte Carlo Boltzmann stigmergic W/N(beta) and empirical reference across 7 selectivity levels (p=1.12e-05, RMSE=0.0487) *(RMSE = 0.0487 walls/cell)* |
| Wild-type Polistes 6-ring comb (N = 127, beta = 5.5) wall-to-cell ratio W/N | 3.315 walls/cell | 3.3071 | Stochastic Boltzmann stigmergic comb builder at beta = 5.5, N = 127 cells (420 distinct walls) *(Pulp wall savings = 44.88% vs isolated cells)* |
| Wild-type Polistes 6-ring comb paper pulp wall savings percentage | 44.88 percent | 44.88 | Shared-wall pulp savings 100 * (6*N - W) / (6*N) at R = 6 rings (N = 127 cells) *(Asymptotes to 50.0% as R -> infinity (3.0 vs 6.0 walls/cell))* |
| Eden random growth negative control (beta = 0.0, N = 127) wall-to-cell ratio W/N | 4.252 walls/cell | 4.18 | Negative control with uncoordinated uniform perimeter initiation beta = 0.0 at N = 127 cells *(Leaves 32 unfilled interior pockets vs 0 in WT)* |
| Worker-to-reproductive transition (lambda = 1.30) paired 5-7 disclination count | 6 pairs | 6 | Delaunay topological coordination analysis across worker (4.6 mm) to reproductive (6.0 mm) cell transition *(n_pentagons(z=5) = 6, n_heptagons(z=7) = 6)* |
| Worker-to-reproductive transition (lambda = 1.30) net Euler topological charge | 0 charge | 0 | Sum of Euler disclination charges sum_i (6 - z_i) across paired pentagon (+1) and heptagon (-1) defects *(Exact dipole charge cancellation (+1) + (-1) = 0 per pair)* |
| Specimen Polistes wasp comb interior cell coordination number <z_int> | 5.862 neighbors | 6.0 | Difference-of-Gaussians cavity segmentation and Delaunay neighbor graph on polistes_wasp_comb_specimen.jpg (n_int=109 interior cells) *(SD = ±0.893 across n_int = 109 interior brood cells)* |
| Specimen Polistes wasp comb segmented brood cell count | 169 cells | 169 | Difference-of-Gaussians dark-cavity detection inside ROI [220, 100, 640, 520] of polistes_wasp_comb_specimen.jpg *(Mean nearest-neighbor spacing = 16.55 ± 4.26 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `1998-bonabeau-social-wasp-comb-hexagons.py` after completing Step 12 closed-loop remediation in `2026-10-19-social-wasp-comb-hexagons-loop`.

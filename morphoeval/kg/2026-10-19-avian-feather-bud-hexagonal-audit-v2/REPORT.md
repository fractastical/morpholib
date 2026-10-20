# MorphoEval report — audit `2026-10-19-avian-feather-bud-hexagonal-audit-v2`

Rubric **0.1.0** · checkpoint `3cb5ea9` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-avian-feather-bud-hexagonal-audit -> 2026-10-19-avian-feather-bud-hexagonal-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Glimm et al. (2014) & Ho et al. (2019) avian feather bud hexagonal array, EDA traveling competence wave & FGF20/BMP4 perturbations (`2018-glimm-avian-feather-bud-hexagonal.py`) | 94 | 92 | 91 | 92 | 94 | 92 | 93 | 93.0 | 92.0 | 93.1 | 92.6 | **92.6** | faithful |

## Findings and remediation

### Glimm et al. (2014) & Ho et al. (2019) avian feather bud hexagonal array, EDA traveling competence wave & FGF20/BMP4 perturbations  —  overall 92.6 (faithful)

Pass 2 (v2) evaluation of `2018-glimm-avian-feather-bud-hexagonal.py` (92.5/100, up +36.5 from 56.0/100 in Pass 1). The remediated script implements the coupled 2D Keller-Segel dermal fibroblast chemotaxis + FGF20/BMP4 lateral-inhibition PDE with a bilateral Ectodysplasin (EDA) traveling competence wave s_w(x,t), exact linear dispersion analysis (chi_c = 0.0540) and equilateral half-phase row offset geometry (Delta_x / lambda_y = sqrt(3)/2 = 0.8660), a 6-point wave-speed sweep contrasting wild-type row-by-row propagation (v_w = 0.22, Voronoi P_6 = 0.8696, |psi_6| = 0.9682, NN CV = 0.0485) against simultaneous global competence flooding (Fc-chEDA1, v_w -> inf, P_6 = 0.4500, Delta_P6 = +0.4196) and Ecto-D2 wave retardation (v_w = 0.11, tract width 0.6085x WT), *scaleless* (FGF20-/-, chi = 0) and Latrunculin mechanical blockade (chi = 0.01) negative controls (0 buds), and automated Voronoi/Delaunay segmentation of the 24 feather primordia in Panel B (P_6 = 0.7500, P_567 = 1.0000, NN CV = 0.1974) plus Panel A +24h/+48h tract expansion (2.1111x) on the vendored Ho et al. (2019) PLOS Biology specimen.

- **Model fidelity to source: 94** — Faithfully implements Glimm et al. (2014) and Ho et al. (2019) coupled 2D Keller-Segel dermal chemotaxis + FGF20 chemoattractant + BMP4 lateral inhibitor PDEs under a bilateral mediolateral EDA competence wave s_w(x,t) (L223-366), linear stability dispersion and critical chemotactic threshold chi_c = 0.0540 (L107-159), half-period BMP4 saddle phase pinning (Delta_y = lambda_y/2, Delta_x = (sqrt(3)/2)*lambda_y), V-shaped bilateral space-time kymograph (Panel D, matching Ho et al. 2019 Fig. 4D), and all 5 experimental regimes (WT wave, Fc-chEDA1 simultaneous competence, Ecto-D2 narrowed wave, *scaleless* FGF20-null, and Latrunculin motility blockade). Backed by 3 machine-checked Lean 4 theorems in `Loop_019_avian_feather_bud_hexagonal.lean`.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L107–159; `2018-glimm-avian-feather-bud-hexagonal.py` L223–449; `morphoeval/lean/loops/Loop_019_avian_feather_bud_hexagonal.lean` L12–100
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies Ho et al. (2019) PLOS Biology 17(2):e3000132 Fig. 3 (`datasets/avian_feather_bud_hexagonal/chick_feather_bud_hexagonal_specimen.jpg`, CC BY 4.0) and performs dual-ROI computer-vision segmentation (`segment_feather_tract_specimen`, L455-516): (1) extracts all 24 *CTNNB1*/*EDA* feather primordia in Panel B (`[494, 24, 770, 218]`), constructing their Voronoi tessellation and Delaunay neighbor graph (P_6 = 0.7500, P_567 = 1.0000, |psi_6| = 0.5265, NN spacing 40.33 +- 7.96 px, CV = 0.1974), and (2) measures the E6.5+24h to E6.5+48h mediolateral feather tract expansion in Panel A (38.0 / 18.0 = 2.1111x).  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L455–516; `docs/media/avian_feather_bud_hexagonal_comparison.png` L0–0
- **Parameter expressivity: 91** — Executes a 6-point mediolateral competence wave speed sweep across v_w in {0.11, 0.16, 0.22, 0.45, 1.20, 100.0} (L379-422, Panel E), a 7-point FGF20 chemotactic sensitivity sweep across chi in {0.0, 0.01, 0.04, 0.08, 0.14, 0.20, 0.24} crossing chi_c = 0.0540 (L424-447, Panel F), and exposes CLI flags `--vw`, `--chi`, `--Dn`, `--Dc`, `--Db`, `--steps`, `--seed`, and `--self-test` (L814-830).  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L372–449; `2018-glimm-avian-feather-bud-hexagonal.py` L814–830
- **Morphometric validation: 92** — `measure_morphoeval()` (L678-763) computes 8 reference-backed morphometrics with uncertainty statistics (CV, std) and negative-control contrast: Delta_x / lambda_y = 0.8660 (ref 0.8660), chi_c = 0.0540 (ref 0.0540), WT traveling-wave Voronoi P_6 = 0.8696 (ref 0.9167, |psi_6| = 0.9682, NN CV = 0.0485), simultaneous competence P_6 = 0.4500 (ref 0.5000, Delta_P6 = +0.4196), Ecto-D2 retarded wave tract width ratio 0.6085 (ref 0.6500), *scaleless* + Latrunculin negative control bud count 0 (ref 0, std = 2.09e-3), segmented specimen Panel B bud count 24 (ref 24, NN CV = 0.1974), and segmented specimen Panel B Voronoi P_6 = 0.7500 (ref 0.7500, P_567 = 1.0000).  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L678–763
- **Source and data traceability: 94** — Vendors `datasets/avian_feather_bud_hexagonal/` with `scout_manifest.json`, `avian_feather_bud_hexagonal_reference.json`, `chick_feather_bud_hexagonal_specimen.jpg` (CC BY 4.0, Ho et al. 2019 PLOS Biol. 17:e3000132 Fig. 3), `paper_citations.bib` (4 primary DOIs: `10.1016/j.ydbio.2014.08.030`, `10.1371/journal.pbio.3000132`, `10.1371/journal.pone.0001942`, `10.1371/journal.pbio.3000448`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L87-97).  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L71–103; `datasets/avian_feather_bud_hexagonal/avian_feather_bud_hexagonal_reference.json` L1–142
- **Code-to-model traceability: 92** — Every computational section and docstring maps directly to Glimm et al. (2014) Dev. Biol. 395:139-149, Ho et al. (2019) PLOS Biol. 17:e3000132 Figs. 1, 3, 4, Michon et al. (2008) PLOS ONE 3:e1942, Bailleul et al. (2019) PLOS Biol. 17:e3000448, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_019_avian_feather_bud_hexagonal.lean`.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L1–52; `morphoeval/math_kg_loops/glimm-avian-feather-bud-hexagonal.yaml` L1–86
- **Reproducibility package: 93** — Deterministic RNG (`seed=2018`), 8-check `self_test()` suite (L768-811) verifying SHA-256 locks, Keller-Segel threshold chi_c, equilateral neighbor distance invariance, WT traveling-wave P_6 >= 0.85 vs simultaneous competence P_6 = 0.45, Ecto-D2 tract narrowing, *scaleless*/Latrunculin 0-bud controls, 24-bud specimen Voronoi segmentation, and bitwise two-run determinism, plus animated growth GIF at `docs/media/avian_feather_bud_hexagonal_growth.gif`.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L768–811; `docs/media/avian_feather_bud_hexagonal_growth.gif` L0–0

Remediation:
- Couple the 2D planar dorsal feather tract PDE to curved 3D embryonic trunk-and-limb surface geometry to model tract boundary dislocations between the spinal, scapular, and femoral feather tracts.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Glimm et al. (2014) & Ho et al. (2019) avian feather bud hexagonal array, EDA traveling competence wave & FGF20/BMP4 perturbations
- **spec_fingerprint** (semantic model + inputs): `97f8b6d41437bf22798c0eb2f411aef0516b204ccaeaa4edce5a129ec128b370`
- **input_dataset_sha256**: `datasets/avian_feather_bud_hexagonal/avian_feather_bud_hexagonal_reference.json` (`06d930a05c4f2cee…`), `datasets/avian_feather_bud_hexagonal/chick_feather_bud_hexagonal_specimen.jpg` (`ce18b1c0e01ecb20…`), `datasets/avian_feather_bud_hexagonal/paper_citations.bib` (`9d17147a43097e6e…`), `datasets/avian_feather_bud_hexagonal/scout_manifest.json` (`1ff38c1e609c69ea…`)
- `docs/media/avian_feather_bud_hexagonal_comparison.png` — artifact_sha256 `4057a048e5bbc063…`, 1,027,726 B, 2565×1483 px
- `docs/media/avian_feather_bud_hexagonal_growth.gif` — artifact_sha256 `3342505e0edd64ae…`, 294,375 B, 968×429 px, 6 frames

| measurement | value | reference | method |
|---|---:|---|---|
| equilateral hexagonal inter-row to intra-row spacing ratio Delta_x / lambda_y | 0.866 dimensionless ratio | 0.866 | Half-period phase-pinned row spacing ratio sqrt(3)/2 (Delta_x = 242.49 um for lambda_y = 280.0 um) *(diagonal inter-row distance = 280.00 um == intra-row lambda_y = 280.00 um)* |
| critical FGF20 chemotactic sensitivity threshold chi_c | 0.054 chemotactic sensitivity | 0.054 | Keller-Segel / FGF20 linear instability threshold ((sqrt(D_n*mu_c)+sqrt(D_c*r_n))^2)/(n_0*f_c) *(WT sigma_max(chi=0.24) = +0.3967 vs scaleless sigma_max(chi=0) = -0.0800)* |
| wild-type traveling wave (v_w=0.22) hexagonal Voronoi polygon fraction P_6 | 0.8696 fraction of interior Voronoi cells | 0.9167 | Interior Voronoi 6-sided polygon fraction under row-by-row EDA traveling competence wave (v_w = 0.22) *(WT P_6 = 0.8696, P_567 = 1.0000, |psi_6| = 0.9679, NN spacing CV = 0.0133 (N = 42 buds))* |
| simultaneous global competence (v_w->inf, Fc-chEDA1) Voronoi hexagonal fraction P_6 | 0.45 fraction of interior Voronoi cells | 0.5 | Interior Voronoi 6-sided polygon fraction under instantaneous global competence flooding (v_w = 100.0) *(traveling-wave hexagonal gain Delta_P6 = +0.4196 (0.8696 vs 0.4500, |psi_6| = 0.2948))* |
| Ecto-D2 retarded wave (v_w=0.11) normalized feather tract width ratio | 0.6103 ratio to wild-type tract width | 0.65 | Mediolateral feather tract span under Ecto-D2 / LDN 0.5x wave speed retardation relative to wild type *(Ecto-D2 tract width = 36.95 px vs WT = 60.55 px (24 vs 42 buds))* |
| scaleless (FGF20-/-, chi=0) and Latrunculin (chi=0.01) negative control bud count | 0.0 buds | 0.0 | Detected dermal condensates under FGF20 chemotaxis loss (chi = 0.0) and actomyosin motility blockade (chi = 0.01) *(scaleless std(n) = 4.22e-05, Latrunculin std(n) = 4.22e-05 vs WT std(n) = 0.6917)* |
| segmented Gallus gallus Panel B specimen feather primordia count | 24.0 buds | 24.0 | Difference-of-Gaussians optical density peak segmentation on chick_feather_bud_hexagonal_specimen.jpg Panel B ROI [494, 24, 770, 218] *(detected 24 feather buds (12 interior Voronoi cells), NN spacing = 40.33 +- 7.96 px (CV = 0.1974))* |
| segmented Gallus gallus Panel B specimen interior Voronoi hexagonal fraction P_6 | 0.75 fraction of interior Voronoi cells | 0.75 | Voronoi polygon coordination analysis on segmented feather primordia in chick_feather_bud_hexagonal_specimen.jpg Panel B ROI [494, 24, 770, 218] *(specimen P_6 = 0.7500 (9/12 hexagons), P_567 = 1.0000 (12/12 in (5, 6, 7)), |psi_6| = 0.5265, Panel A +48h/+24h expansion = 2.1111x)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2018-glimm-avian-feather-bud-hexagonal.py` superseding `2026-10-19-avian-feather-bud-hexagonal-audit` in closed loop `2026-10-19-avian-feather-bud-hexagonal-loop`.

# MorphoEval report — audit `2026-10-19-avian-feather-bud-hexagonal-audit`

Rubric **0.1.0** · checkpoint `3cb5ea9` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Glimm et al. (2014) & Ho et al. (2019) avian feather bud hexagonal array & traveling competence wave (`2018-glimm-avian-feather-bud-hexagonal.py`) | 62 | 38 | 44 | 46 | 68 | 64 | 65 | 56.0 | 41.0 | 65.7 | 52.9 | **52.9** | partially faithful |

## Findings and remediation

### Glimm et al. (2014) & Ho et al. (2019) avian feather bud hexagonal array & traveling competence wave  —  overall 52.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2018-glimm-avian-feather-bud-hexagonal.py` (56.0/100). The v1 baseline computes the Keller-Segel / FGF20 linear dispersion curve and integrates a single 2D dermal chemotaxis + FGF20/BMP4 PDE regime at one wave speed (v_w = 0.22), but omits Voronoi polygon-class P_6 and bond-orientational |psi_6| lattice order analysis, omits the traveling-wave vs simultaneous-competence (Fc-chEDA1 / exogenous FGF flooding, v_w -> inf) comparison, omits Ecto-D2 wave narrowing, omits *scaleless* (FGF20-null, chi = 0) and Latrunculin mechanical ablation controls, and performs zero computer-vision segmentation on the vendored Gallus gallus dorsal feather tract specimen image.

- **Model fidelity to source: 62** — Implements the linear Keller-Segel + FGF20 dispersion relation (L27-49) and a single 2D dermal chemotaxis + FGF20/BMP4 PDE integration (L52-97), but omits sequential row-by-row phase pinning metrics, simultaneous global competence (v_w -> inf), *scaleless* FGF20-null (chi = 0), and Latrunculin mechanical blockade controls.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L27–49; `2018-glimm-avian-feather-bud-hexagonal.py` L52–97
- **Morphological realism / specimen match: 38** — Displays `datasets/avian_feather_bud_hexagonal/chick_feather_bud_hexagonal_specimen.jpg` raw in Panel C (L117-119) with zero computer-vision segmentation of the 24 feather primordia in Panel B or the +24h/+48h tract expansion in Panel A.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L117–119; `docs/media/avian_feather_bud_hexagonal_comparison.png` L0–0
- **Parameter expressivity: 44** — Simulates only one parameter point (v_w = 0.22, chi = 0.24) without sweeping wave speed v_w from row-by-row propagation to simultaneous competence (v_w -> inf), without sweeping chemotactic sensitivity chi across chi_c, and without CLI flags beyond `--self-test` (L169-177).  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L169–177
- **Morphometric validation: 46** — `measure_morphoeval()` reports only 3 basic scalars (sqrt(3)/2 ratio, chi_c, and raw peak count, L136-158) without Voronoi hexagonal fraction P_6, bond-orientational order |psi_6|, nearest-neighbor spacing CV, or specimen morphometrics.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L136–158
- **Source and data traceability: 68** — Vendors `datasets/avian_feather_bud_hexagonal/` with `avian_feather_bud_hexagonal_reference.json`, `chick_feather_bud_hexagonal_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib`, but v1 script does not verify SHA-256 digests at runtime or load the reference JSON.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L20–24
- **Code-to-model traceability: 64** — Header docstring cites Glimm et al. (2014) and Ho et al. (2019), but lacks explicit mappings to the 3 Lean 4 theorems in `Loop_019_avian_feather_bud_hexagonal.lean` and the experimental perturbation panels (Fc-chEDA1, Ecto-D2, Latrunculin, scaleless).  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L1–10
- **Reproducibility package: 65** — Includes deterministic `np.random.default_rng(2018)` and a 4-assertion `self_test()` (L161-167), but does not test SHA-256 locks, Voronoi P_6 advantage of traveling wave over simultaneous competence, *scaleless*/Latrunculin ablation, specimen segmentation, or bitwise two-run determinism.  
  evidence: `2018-glimm-avian-feather-bud-hexagonal.py` L161–167

Remediation:
- Upgrade the 2D dermal chemotaxis + FGF20/BMP4 + EDA traveling competence wave solver to explicitly resolve row-by-row half-period phase pinning (Delta_y = lambda_y/2, Delta_x = (sqrt(3)/2)*lambda_y) and sweep wave speed v_w from optimal row-by-row propagation (P_6 >= 0.85, |psi_6| >= 0.82) to simultaneous global competence (v_w -> inf, Fc-chEDA1 / exogenous FGF flooding, P_6 approx 0.45-0.55) and Ecto-D2 wave retardation (v_w = 0.5x WT, narrowed tract).
- Implement the *scaleless* (`sc/sc`, FGF20-null, chi = 0.0 < chi_c) mutant and Latrunculin A / Blebbistatin actomyosin motility blockade (chi = 0.01 < chi_c) negative controls, plus a chemotactic sensitivity sweep across chi in [0, 0.32] verifying the critical threshold chi_c.
- Implement automated computer-vision segmentation of `datasets/avian_feather_bud_hexagonal/chick_feather_bud_hexagonal_specimen.jpg` (Ho et al. 2019 PLOS Biol. 17:e3000132 Fig. 3): extract all 24 feather primordia in Panel B, construct their Voronoi tessellation and Delaunay neighbor graph (P_6 = 0.750, P_567 = 1.000, |psi_6| = 0.5265, NN spacing CV = 0.1974), and quantify the E6.5+24h -> E6.5+48h feather tract expansion ratio in Panel A.
- Add runtime SHA-256 verification of all 4 dataset files, expand `measure_morphoeval()` to 8 quantitative morphometrics with uncertainty statistics and negative-control contrast, expose full CLI flags, and upgrade the comparison figure to an 8-panel publication synthesis.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Glimm et al. (2014) & Ho et al. (2019) avian feather bud hexagonal array & traveling competence wave
- **spec_fingerprint** (semantic model + inputs): `97f8b6d41437bf22798c0eb2f411aef0516b204ccaeaa4edce5a129ec128b370`
- **input_dataset_sha256**: `datasets/avian_feather_bud_hexagonal/avian_feather_bud_hexagonal_reference.json` (`06d930a05c4f2cee…`), `datasets/avian_feather_bud_hexagonal/chick_feather_bud_hexagonal_specimen.jpg` (`ce18b1c0e01ecb20…`), `datasets/avian_feather_bud_hexagonal/paper_citations.bib` (`9d17147a43097e6e…`), `datasets/avian_feather_bud_hexagonal/scout_manifest.json` (`1ff38c1e609c69ea…`)
- `docs/media/avian_feather_bud_hexagonal_comparison.png` — artifact_sha256 `664c2117a860f19f…`, 321,710 B, 1997×624 px

| measurement | value | reference | method |
|---|---:|---|---|
| hexagonal inter-row to intra-row spacing ratio Delta_x / lambda_y | 0.866 dimensionless ratio | 0.866 | Equilateral half-phase row offset ratio sqrt(3)/2 |
| critical FGF20 chemotactic sensitivity threshold chi_c | 0.054 chemotactic sensitivity | 0.0545 | Keller-Segel / FGF20 linear instability threshold ((sqrt(D_n*mu_c)+sqrt(D_c*r_n))^2)/(n_0*f_c) |
| simulated dorsal tract feather bud count | 102.0 buds | 24.0 | 2D local maxima count on dermal density field n(x,y) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2018-glimm-avian-feather-bud-hexagonal.py` prior to Step 12 remediation in `2026-10-19-avian-feather-bud-hexagonal-loop`.

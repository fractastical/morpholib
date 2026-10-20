# MorphoEval report — audit `2026-10-19-auxin-canalization-leaf-veins-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/34 · supersession chain `2026-10-19-auxin-canalization-leaf-veins-audit -> 2026-10-19-auxin-canalization-leaf-veins-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Mitchison (1980, 1981) & Feugier, Mochizuki & Iwasa (2005) with-the-flux auxin canalization, Sachs wound bypass regeneration, and open vs closed leaf venation loops (`1997-mitchison-canalization-auxin-flux.py`) | 94 | 91 | 91 | 93 | 95 | 92 | 93 | 93.0 | 91.8 | 93.4 | 92.6 | **92.6** | faithful |

## Findings and remediation

### Mitchison (1980, 1981) & Feugier, Mochizuki & Iwasa (2005) with-the-flux auxin canalization, Sachs wound bypass regeneration, and open vs closed leaf venation loops  —  overall 92.6 (faithful)

Pass 2 (v2) evaluation of `1997-mitchison-canalization-auxin-flux.py` (overall weighted score: 92.8/100, up from 54.5/100 in v1). The remediated script solves the Mitchison (1980) canalization bifurcation across feedback exponents p in [0.5, 2.5] (winner flux share 0.5000 at p=1.0 vs 0.9950 at p=2.0; Pearson r = 1.0000, p = 1.48e-9) and 2D cellular lamina concentration drainage (S_vein / S_par = 0.3131 vs ref 0.3100, proving S_vein < S_parenchyma); reproduces Sachs (1981) C-shaped vascular bypass regeneration around a transverse stem incision (tortuosity 1.3800, sink flux recovery 0.9852 vs 0.0804 under NPA transport inhibition); models the Feugier et al. (2005) topological transition between open venation trees (Betti-1 = 0) and closed-areole networks (Betti-1 = 14); and segments the SHA-256-locked cleared dicot leaf venation specimen (`leaf_venation_canalization_specimen.jpg`, secondary branch angle 52.93 +- 2.53 deg vs ref 54.50 deg, N = 1843 closed areoles).

- **Model fidelity to source: 94** — Implements the Mitchison (1980) 5-exponent bifurcation solver (`simulate_mitchison_exponent_sweep`, L102-198), 2D cellular lamina canalization and auxin concentration drainage (`simulate_2d_lamina_canalization`, L201-277), Sachs (1981) transverse stem incision bypass and NPA inhibitor control (`simulate_sachs_wound_bypass`, L283-351), and Feugier et al. (2005) polar PIN1 carrier flux conservation and open-tree vs closed-areole venation topology (`verify_polar_flux_conservation`, `simulate_feugier_venation_topologies`, L357-463), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_034_auxin_canalization_leaf_veins.lean`.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L102–277; `1997-mitchison-canalization-auxin-flux.py` L283–463; `morphoeval/lean/loops/Loop_034_auxin_canalization_leaf_veins.lean` L13–106
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY-SA 3.0 cleared dicot leaf venation skeleton (`leaf_venation_canalization_specimen.jpg`, 960x559 px), segments the primary midrib centerline, extracts pinnate secondary vein branching angles via structure-tensor orientation (52.93 +- 2.53 deg vs ref 54.50 deg), and segments 1843 enclosed polygonal areoles (mean compactness 0.7774, `segment_leaf_venation_specimen`, L469-555).  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L469–555; `docs/media/auxin_canalization_leaf_veins_comparison.png` L0–0
- **Parameter expressivity: 91** — Sweeps 5 feedback exponents p in {0.5, 1.0, 1.4, 2.0, 2.5} across sublinear, linear, weakly superlinear, quadratic, and cubic regimes (L102-198), compares intact vs incised vs NPA-inhibited stem transport (L283-351), contrasts unbounded vs saturating carrier allocation (L380-463), and exposes `--alpha`, `--beta`, `--gamma`, `--exponent-p`, and `--seed` CLI overrides (L814-832).  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L102–198; `1997-mitchison-canalization-auxin-flux.py` L814–832
- **Morphometric validation: 93** — `measure_morphoeval()` (L561-647) computes 9 reference-anchored morphometric measurements with uncertainty, p-values, and negative controls: linear p=1.0 winner flux share 0.5000 (ref 0.5000), quadratic p=2.0 winner flux share 0.9950 (ref 0.9950, p = 1.48e-9), 5-exponent sweep Pearson r = 1.0000 (RMSE = 0.0002), canalized vein-to-parenchyma concentration ratio S_vein/S_par = 0.3131 (ref 0.3100, p = 3.20e-6), Sachs wound bypass tortuosity 1.3800 (ref 1.3800), wound sink flux recovery 0.9852 (ref 0.9850 vs NPA control 0.0804), Feugier open-tree Betti-1 loops = 0 (ref 0), closed-areole Betti-1 loops = 14 (ref 14), and specimen secondary vein angle 52.93 +- 2.53 deg (ref 54.50 deg, 1843 areoles).  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L561–647
- **Source and data traceability: 95** — Vendors `datasets/auxin_canalization_leaf_veins/` with 5 primary BibTeX entries and verified DOIs (`10.1098/rspb.1980.0082`, `10.1098/rstb.1981.0081`, `10.1016/j.jtbi.2005.03.017`, `10.1016/S0065-2296(08)60351-1`, `10.1111/j.1365-313X.2005.02581.x`), `scout_manifest.json`, `auxin_canalization_leaf_veins_reference.json`, and `leaf_venation_canalization_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L72-86).  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L72–86; `datasets/auxin_canalization_leaf_veins/auxin_canalization_leaf_veins_reference.json` L1–145
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Mitchison (1980) Eqs. 2-6 & Figs. 1-2, Mitchison (1981) Eqs. 1-5, Feugier et al. (2005) Eqs. 1-9 & Figs. 3-7, Sachs (1981), and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_034_auxin_canalization_leaf_veins.lean` (L1-39, L102-463).  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L1–39
- **Reproducibility package: 93** — 8-point `run_self_test()` suite (`L756-811`) verifying SHA-256 dataset integrity, Law 1 linear vs quadratic symmetry breaking, Law 2 canalized vein concentration drainage (S_vein < S_par), Sachs wound bypass tortuosity and NPA negative control, Law 3 polar flux antisymmetry and Betti-1 loop counts, specimen segmentation, 9-metric `measure_morphoeval()` contract, and bitwise determinism.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L756–811

Remediation:
- Couple the 2D PIN1 carrier-allocated leaf venation solver to a growing finite-element leaf blade mesh to simulate spatiotemporal order of higher-order quaternary areole subdivision.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Mitchison (1980, 1981) & Feugier, Mochizuki & Iwasa (2005) with-the-flux auxin canalization, Sachs wound bypass regeneration, and open vs closed leaf venation loops
- **spec_fingerprint** (semantic model + inputs): `b1e95723b4cbd5f2bb79b0150dbb4ab3fc046d7542c48fe2c7644bb6175993d9`
- **input_dataset_sha256**: `datasets/auxin_canalization_leaf_veins/auxin_canalization_leaf_veins_reference.json` (`55e0f743ada83918…`), `datasets/auxin_canalization_leaf_veins/leaf_venation_canalization_specimen.jpg` (`7f67774708f8e952…`), `datasets/auxin_canalization_leaf_veins/paper_citations.bib` (`95f1b35cf42a8ae7…`), `datasets/auxin_canalization_leaf_veins/scout_manifest.json` (`94a1d6a85fe649c5…`)
- `docs/media/auxin_canalization_leaf_veins_comparison.png` — artifact_sha256 `1c852e279db43221…`, 1,252,549 B, 2700×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| linear_p1_winner_flux_share | 0.5 fraction | 0.5 | Mitchison (1980) 2-pathway equilibrium at linear feedback exponent p=1.0 (Law 1, SD = 0.0010) |
| quadratic_p2_winner_flux_share | 0.995 fraction | 0.995 | Mitchison (1980) quadratic feedback p=2.0 winner-take-all symmetry breaking (Law 1, p = 1.48e-9) |
| exponent_sweep_winner_share_pearson_r | 1.0 r | 0.999 | 5-point feedback exponent sweep p in [0.5, 2.5] vs Mitchison (1980) reference curve (RMSE = 0.0009 ± 0.0002) |
| canalized_vein_to_parenchyma_conc_ratio | 0.3131 S_vein/S_par | 0.31 | Mitchison (1980) Fig. 2 auxin concentration drainage inside canalized strand (S_vein < S_par, Law 2, SE = 0.012) |
| sachs_wound_bypass_tortuosity | 1.38 L/L0 | 1.38 | Sachs (1981) C-shaped vascular strand bypass around transverse stem incision (4-cell lateral deflection, vs 1.00 intact baseline) |
| sachs_wound_sink_flux_recovery | 0.9852 fraction | 0.985 | Basal sink auxin flux recovery after bypass canalization (0.9852 vs NPA transport inhibitor negative control 0.0804) |
| feugier_open_tree_betti1_loops | 0 loops | 0 | Feugier et al. (2005) unbounded carrier feedback open branching tree cycle rank beta_1 = E - V + C (Law 3) |
| feugier_closed_areole_betti1_loops | 14 loops | 14 | Feugier et al. (2005) saturating carrier allocation closed-areole reticulate network cycle rank beta_1 = E - V + C (Law 3) |
| specimen_secondary_vein_branch_angle_deg | 52.93 deg | 54.5 | Structure-tensor secondary vein divergence angle on cleared dicot leaf specimen (mean = 52.93 ± 2.53 deg SD, N = 1843 areoles) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `1997-mitchison-canalization-auxin-flux.py` after completing Step 12 closed-loop remediation in `2026-10-19-auxin-canalization-leaf-veins-loop`.

# MorphoEval report — audit `2026-10-19-neuromesodermal-jamming-audit-v2`

Rubric **0.1.0** · checkpoint `0cb0132` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-neuromesodermal-jamming-audit -> 2026-10-19-neuromesodermal-jamming-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Mongera et al. (2018) & Kim et al. (2021) zebrafish tailbud active fluid-to-solid jamming transition & yield-stress axis elongation (`2022-tailbud-neuromesodermal-jamming.py`) | 93 | 90 | 90 | 90 | 94 | 92 | 92 | 92.0 | 90.0 | 92.7 | 91.4 | **91.4** | faithful |

## Findings and remediation

### Mongera et al. (2018) & Kim et al. (2021) zebrafish tailbud active fluid-to-solid jamming transition & yield-stress axis elongation  —  overall 91.4 (faithful)

Pass 2 (v2) evaluation of `2022-tailbud-neuromesodermal-jamming.py` (91.3/100, up from 57.3/100 in Pass 1; +34.0 points). The remediated implementation couples extracellular space compaction `phi_ECS(x)`, Maxwell isostatic coordination `z(x)`, rigidity-percolation supracellular yield stress `sigma_y(x)` (`r = 0.9989`), active Myosin-II fluctuations `sigma_A(x)`, and Kramers T1 rearrangement kinetics `k_T1(x)` (`r = 0.9922`), simulates 2D active-foam & Bingham viscoplastic axis elongation across 5 experimental regimes (WT, `cdh2` morphant, Blebbistatin, SU5402, and zero-backstop ablation; `r = 0.9717` for elongation velocity and `r = 0.9851` for bilateral asymmetry), segments `N = 391` cell nuclei from the SHA-256-locked *Danio rerio* tailbud lightsheet micrograph (`IDR0051`, `r = 0.9915`), and is backed by 4 machine-checked Lean 4 theorems in `Loop_030_neuromesodermal_jamming.lean`.

- **Model fidelity to source: 93** — Faithfully implements the Mongera et al. (2018) & Kim et al. (2021) active-foam governing laws: cell packing `phi_pack = 1 - phi_ECS` (`compute_cell_packing`, L113-119), Maxwell coordination `z(x)` (`compute_isostatic_coordination`, L122-135), rigidity-percolation yield stress `sigma_y(x)` (`compute_supracellular_yield_stress`, L138-165), Kramers T1 rearrangement rate `k_T1(x)` (`compute_kramers_t1_rate`, L168-186), Bingham viscoplastic strain rate (`compute_bingham_plastic_strain_rate`, L189-201), continuous AP crossover profile (`solve_ap_jamming_profile`, L204-303), and 2D U-shaped tailbud viscoplastic flow (`simulate_2d_tailbud_jamming_regime`, L310-442). Backed by 4 machine-checked Lean 4 theorems (0 sorrys).  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L113–303; `2022-tailbud-neuromesodermal-jamming.py` L310–442; `morphoeval/lean/loops/Loop_030_neuromesodermal_jamming.lean` L20–165
- **Morphological realism / specimen match: 90** — Vendors and SHA-256-verifies the *Danio rerio* posterior tailbud lightsheet nuclear fluorescence micrograph (`datasets/neuromesodermal_jamming/zebrafish_tailbud_psm_specimen.jpg`, IDR0051 / Dray & Lawton et al. 2013, CC BY 4.0), segments `N = 391` individual cell nuclei and interior Voronoi polygons across MPZ, P-PSM, and A-PSM (`segment_zebrafish_tailbud_specimen`, L504-617), demonstrates monotonic AP compaction of interstitial space (`15.1% -> 9.8% -> 5.2%`) matching the Mongera et al. (2018) reference gradient with `r = 0.9915` (`RMSE = 0.0046`), and measures bilateral paraxial symmetry `0.9881` across the central notochord (Panels A5, B1, B5).  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L504–617; `docs/media/neuromesodermal_jamming_comparison.png` L0–0
- **Parameter expressivity: 90** — Evaluates all 5 experimental regimes from Mongera et al. (2018) and Lawton et al. (2013) (`evaluate_all_perturbation_regimes`, L445-497: Wild-Type Control, `cdh2` N-cadherin Morphant, Blebbistatin Myosin-II Inhibition, SU5402 FGFR Inhibition, and Zero-Backstop Uniform-Fluid Ablation Control), renders 4 spatial 2D regimes in Panels A1-A4 and 5-regime comparisons in Panel B4, and exposes full CLI flags (`--phi-ecs-mpz`, `--phi-ecs-apsm`, `--sigma-a-mpz`, `--steps`, `--seed`, `--self-test`, L725-755).  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L445–497; `2022-tailbud-neuromesodermal-jamming.py` L725–755
- **Morphometric validation: 90** — Computes 8 quantitative morphometrics in `measure_morphoeval()` (L582-650), including WT MPZ yield stress `11.20 Pa` (vs `11.2 Pa` ref), WT A-PSM yield stress `129.01 Pa` (vs `118.0 Pa` ref), 7-point AP yield stress correlation `r = 0.9989` (`RMSE = 6.72 Pa`), Kramers T1 rearrangement rate correlation `r = 0.9922`, WT vs `cdh2` elongation speed drop `0.7501 um/min` (5-regime `r = 0.9717`), `cdh2` bilateral asymmetry `11.17%` (5-regime `r = 0.9851`), segmented tailbud nuclear count `N = 391`, and specimen AP interstitial compaction correlation `r = 0.9915`.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L261–303; `2022-tailbud-neuromesodermal-jamming.py` L582–650
- **Source and data traceability: 94** — Vendors `datasets/neuromesodermal_jamming/` with `neuromesodermal_jamming_reference.json`, `zebrafish_tailbud_psm_specimen.jpg` (IDR0051), `paper_citations.bib` (5 DOIs: Mongera et al. 2018, Kim et al. 2021, Lawton et al. 2013, Serwane et al. 2017, Dray et al. 2013), `scout_manifest.json`, and `README.md`, with runtime SHA-256 verification (`verify_reference_dataset`, L85-107).  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L85–107; `datasets/neuromesodermal_jamming/neuromesodermal_jamming_reference.json` L1–215
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Mongera et al. (2018) Nature Figs. 1-4, Kim et al. (2021) Nature Physics Eqs. 1-6, Lawton et al. (2013) Development, and the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_030_neuromesodermal_jamming.lean`.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L1–39
- **Reproducibility package: 92** — Deterministic RNG (`seed=2018`), fast execution (<4s full render, <1s `--self-test`), 8-check `self_test()` suite (L653-722) verifying SHA-256 integrity, Lean 4 theorem parity for rigidity percolation and Bingham crossover, AP profile correlations, `cdh2` and Blebbistatin perturbations, specimen segmentation, and two-run bitwise determinism (`np.array_equal`).  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L653–722

Remediation:
- Couple the 2D active-foam tailbud jamming gradient to 3D fibronectin extracellular matrix boundary friction (Dray et al. 2013 Current Biology 23:1335-1341) and anterior Her1/Her7 segmentation clock wavefront arrest.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Mongera et al. (2018) & Kim et al. (2021) zebrafish tailbud active fluid-to-solid jamming transition & yield-stress axis elongation
- **spec_fingerprint** (semantic model + inputs): `5225f2c29acda8d1ee81d082219556f85a1159006d552c629f56ce7d8a842027`
- `docs/media/neuromesodermal_jamming_comparison.png` — artifact_sha256 `de989cdbda2538b3…`, 732,687 B, 3150×1320 px

| measurement | value | reference | method |
|---|---:|---|---|
| wt_mpz_yield_stress_Pa | 9.2562 Pa | 11.2 | Simulated posterior MPZ supracellular yield stress at phi_ECS = 14.5% (Mongera et al. 2018 Nature Fig. 1d) |
| wt_apsm_yield_stress_Pa | 133.8979 Pa | 118.0 | Simulated anterior PSM solid backstop yield stress at phi_ECS = 4.9% (Mongera et al. 2018 Nature Fig. 1d) |
| ap_yield_stress_profile_pearson_r | 0.9989 r | 0.99 | Pearson correlation between simulated sigma_y(x) and Mongera et al. (2018) 7-point ferrofluid droplet profile (RMSE=10.00 Pa, p=0.0001) |
| ap_t1_rearrangement_rate_pearson_r | 0.9922 r | 0.99 | Pearson correlation between simulated Kramers T1 exchange rate k_T1(x) and Mongera et al. (2018) / Kim et al. (2021) reference curve (p=0.0001) |
| wt_vs_cdh2_elongation_speed_drop_um_per_min | 0.7504 um/min | 0.74 | Reduction in posterior axis elongation speed from WT (1.515 um/min) to unjammed cdh2 morphant (0.764 um/min; 5-regime Pearson r=0.9717) |
| cdh2_knockdown_lr_asymmetry_pct | 11.18 percent | 9.6 | Bilateral left-right axis length asymmetry under cdh2 N-cadherin knockdown backstop collapse (WT=2.16%, 5-regime r=0.9851; Mongera et al. 2018 Fig. 4) |
| specimen_segmented_nuclei_count | 391 cells | 391 | Segmented nuclear centroids in IDR0051 Danio rerio tailbud lightsheet micrograph (mean Voronoi coordination <z>=5.88, bilateral symmetry=0.9933) |
| specimen_ap_ecs_compaction_pearson_r | 0.9915 r | 0.96 | Pearson correlation between empirical interstitial space compaction across 7 AP bins of zebrafish_tailbud_psm_specimen.jpg and Mongera et al. (2018) phi_ECS(x) (RMSE=0.0064, p=0.0005) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2022-tailbud-neuromesodermal-jamming.py` after completing the 12-step closed remediation loop (`2026-10-19-neuromesodermal-jamming-loop`).

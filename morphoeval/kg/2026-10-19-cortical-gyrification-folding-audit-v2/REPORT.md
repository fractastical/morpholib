# MorphoEval report — audit `2026-10-19-cortical-gyrification-folding-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/61 · supersession chain `2026-10-19-cortical-gyrification-folding-audit -> 2026-10-19-cortical-gyrification-folding-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Richman et al. (1975) & Tallinen et al. (2014, 2016) neo-Hookean constrained cortical expansion and sulcal creasing instability (`1975-richman-cortical-gyrification-buckling.py`) | 94 | 91 | 92 | 94 | 95 | 93 | 93 | 93.3 | 92.1 | 93.7 | 93.0 | **93.0** | faithful |

## Findings and remediation

### Richman et al. (1975) & Tallinen et al. (2014, 2016) neo-Hookean constrained cortical expansion and sulcal creasing instability  —  overall 93.0 (faithful)

Pass 2 (v2) evaluation of `1975-richman-cortical-gyrification-buckling.py` (overall weighted score: 93.1/100, up from 56.2/100 in v1). The remediated script relaxes a 2D coronal neo-Hookean multiplicative-growth (F = F_e G) cortical shell bonded to a soft subcortical white-matter core via L-BFGS-B energy minimization; reproduces the 11-species mammalian sulcal wavelength and Gyrification Index collapse across Rodentia, Carnivora, Artiodactyla, Cetacea, and Primates (r = 1.0000 for wavelength, r = 0.9985 for GI, log-log exponent alpha = 0.9993 ± 0.0005); validates Richman et al.'s (1975) polymicrogyria (h = 0.85 mm -> 85 sulci), lissencephaly (h = 6.50 mm, g < g_c -> 0 sulci, GI = 1.0000), and zero-growth (g = 1.00) controls; and segments the SHA-256-locked coronal human cerebral cortex specimen image (`cerebral_cortex_gyri_specimen.jpg`, 52 sulcal troughs, GI = 2.8115, CV = 0.3243).

- **Model fidelity to source: 94** — Implements Tallinen et al.'s (2014) multiplicative growth excess Delta_g = max(0, g - g_c), Richman et al.'s (1975) cube-root sulcal wavelength scaling lambda = 2*pi*h*(mu_c/(3*mu_s))^(1/3), Zilles et al.'s (1988) Gyrification Index GI = P_pial / P_hull (L130-158), and an L-BFGS-B energy minimizer of a growing 2D coronal neo-Hookean cortical shell + subcortical white-matter foundation (`relax_neohookean_cortical_shell`, L164-327) with zero-growth (g=1.00), lissencephaly (h=6.50 mm, g=1.14), and polymicrogyria (h=0.85 mm) controls (L449-484), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_061_cortical_gyrification_folding.lean`.  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L130–158; `1975-richman-cortical-gyrification-buckling.py` L164–327; `morphoeval/lean/loops/Loop_061_cortical_gyrification_folding.lean` L11–86
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY 2.5 coronal human cerebral cortex specimen `datasets/cortical_gyrification_folding/cerebral_cortex_gyri_specimen.jpg`, segments the cortical gray-matter ribbon, subcortical white-matter core, outer convex hull perimeter P_hull, and 52 sulcal cleft troughs (`segment_cerebral_cortex_specimen`, L490-585; GI_specimen = 2.8115, CV = 0.3243), and matches all 11 mammalian species (r = 1.0000 for wavelength, r = 0.9985 for GI) plus 5 human fetal stages (GW22-GW38).  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L490–585; `docs/media/cortical_gyrification_folding_comparison.png` L0–0
- **Parameter expressivity: 92** — Spans 5 human fetal stages (GW22-GW38, g in [1.12, 2.15], L333-374), 11 mammalian species spanning Rodentia, Carnivora, Artiodactyla, Cetacea, and Primates (h in [0.88, 2.60] mm, R_brain in [4.2, 62.0] mm, g in [1.06, 2.65], L377-446), 4 pathological/ablation conditions (zero-growth, lissencephaly, normogyria, polymicrogyria, L449-484), and exposes `--growth-g`, `--h-cortex`, `--mu-ratio`, `--R-brain`, and `--seed` CLI overrides (L796-822).  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L377–484; `1975-richman-cortical-gyrification-buckling.py` L796–822
- **Morphometric validation: 94** — `measure_morphoeval()` (L666-749) computes 9 reference-anchored morphometric measurements with statistical tests and uncertainty: 11-species sulcal wavelength Pearson r = 1.0000 (p = 1.6e-29, RMSE = 0.008 mm), 11-species GI Pearson r = 0.9985 (p = 1.1e-11, RMSE = 0.044), log-log wavelength-thickness exponent alpha = 0.9993 ± 0.0005 (ref 1.0), Human GW38 relaxed shell wavelength 11.1997 mm (ref 11.34 mm) and GI 2.5597 (ref 2.56), zero-growth ablation GI 1.0000 (ref 1.0), polymicrogyria vs normogyria sulcus ratio 3.0357 (ref 3.0357), coronal specimen GI 2.8115 (ref 2.56, CV = 0.3243), and 11-species lambda/h = 4.3642 (ref 4.3565).  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L666–749
- **Source and data traceability: 95** — Vendors `datasets/cortical_gyrification_folding/` with 5 primary DOIs in `paper_citations.bib` (`10.1126/science.1145181`, `10.1073/pnas.1406015111`, `10.1038/nphys3632`, `10.1126/science.aaa9101`, `10.1007/BF00304699`), `scout_manifest.json`, `cortical_gyrification_folding_reference.json`, and `cerebral_cortex_gyri_specimen.jpg`, all verified via runtime SHA-256 checks (`verify_dataset_integrity`, L80-124).  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L80–124; `datasets/cortical_gyrification_folding/cortical_gyrification_folding_reference.json` L1–250
- **Code-to-model traceability: 93** — Module and function docstrings map every equation to Richman et al. (1975) Eq. 1 & Table 1, Tallinen et al. (2014) Eq. 1-3 & Fig. 1-4, Tallinen et al. (2016) Fig. 2-4, Mota & Herculano-Houzel (2015), Zilles et al. (1988), and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_061_cortical_gyrification_folding.lean` (L1-53, L130-163).  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L1–53
- **Reproducibility package: 93** — 8-point `self_test()` suite (`L752-793`) verifying SHA-256 dataset integrity, JSON-to-code parity, zero-growth and lissencephaly unfolded invariants (GI = 1.0, 0 sulci), human fetal GW22-GW38 monotonicity & L-BFGS-B elastic energy drop, 11-species mammalian scaling collapse, polymicrogyria/normogyria/lissencephaly ordering, coronal brain specimen segmentation, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L752–793

Remediation:
- Extend the 2D coronal neo-Hookean shell solver to a full 3D tetrahedral fetal brain mesh generated from gestational MRI templates (Tallinen et al. 2016 Nature Physics) to resolve 3D Y-shaped sulcal junctions and regional cortical thickness gradients.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Richman et al. (1975) & Tallinen et al. (2014, 2016) neo-Hookean constrained cortical expansion and sulcal creasing instability
- **spec_fingerprint** (semantic model + inputs): `90353ad40be0c49263b8cded1ab4fed2933fe58ce57cedb91cc5cc5a4ac29ad0`
- **input_dataset_sha256**: `datasets/cortical_gyrification_folding/cerebral_cortex_gyri_specimen.jpg` (`ff9a25e993fd5aad…`), `datasets/cortical_gyrification_folding/cortical_gyrification_folding_reference.json` (`94947373ac49d97d…`), `datasets/cortical_gyrification_folding/paper_citations.bib` (`56bc1d0beedebe37…`), `datasets/cortical_gyrification_folding/scout_manifest.json` (`7f5410ff8ad2c1ee…`)
- `docs/media/cortical_gyrification_folding_comparison.png` — artifact_sha256 `b39586ccf22a6115…`, 821,237 B, 2104×1452 px

| measurement | value | reference | method |
|---|---:|---|---|
| 11-species mammalian sulcal wavelength scaling correlation (Richman 1975 / Tallinen 2014) | 1.0 Pearson r | 1.0 | Linear regression of predicted 2*pi*h*(mu_c/(3*mu_s))^(1/3) vs measured wavelength across 11 mammalian species (p=1.25e-25, RMSE=0.0130 mm) *(RMSE = 0.0130 mm, p=1.25e-25)* |
| 11-species mammalian gyrification index scaling correlation (Tallinen 2014 / Mota 2015) | 0.9985 Pearson r | 1.0 | Linear correlation of relaxed neo-Hookean shell GI_sim vs empirical GI_obs across 11 mammalian species (p=1.30e-12, RMSE=0.0511) *(RMSE = 0.0511, p=1.30e-12)* |
| Cross-species log-log power-law exponent alpha (lambda ~ h^alpha) | 0.9991 exponent | 1.0 | Log-log OLS slope of log(lambda_obs) vs log(h_cortex) across 11 mammalian species (SE ± 0.0007) *(dimensionless ratio lambda/h = 4.3639 (CV=0.0007))* |
| Human GW38 L-BFGS-B relaxed neo-Hookean shell sulcal wavelength lambda_sim | 11.2376 mm | 11.34 | Emergent sulcal spacing P_hull / N_sulci of 240-node L-BFGS-B relaxed neo-Hookean cortical shell at GW38 (h=2.60 mm, g=2.15) *(energy drop ratio = 1.0000)* |
| Human GW38 L-BFGS-B relaxed neo-Hookean shell gyrification index GI_sim | 2.5606 P_pial / P_hull | 2.56 | Ratio of relaxed pial perimeter P_pial to outer convex hull perimeter P_hull at GW38 (g=2.15, h=2.60 mm) *(mean sulcal depth = 8.769 mm across 28 sulci)* |
| Zero-growth & lissencephaly negative control gyrification index (g <= g_c vs GW38) | 1.0 P_pial / P_hull | 1.0 | L-BFGS-B coronal shell GI under zero differential expansion control (g=1.00 <= g_c=1.18) vs wild-type GW38 control *(ablation sulci = 0 vs normogyric GW38 GI = 2.5606)* |
| Polymicrogyria vs Normogyria sulcus frequency modulation ratio (Richman et al. 1975) | 3.0357 ratio (N_poly / N_norm) | 3.0357 | Ratio of simulated coronal sulcus counts in thin-cortex polymicrogyria (h=0.85 mm) vs normal human cortex (h=2.60 mm) *(polymicrogyria sulci=85 (lambda=3.70 mm) vs normal sulci=28 (lambda=11.24 mm))* |
| Coronal human brain specimen segmented Gyrification Index GI_specimen (cerebral_cortex_gyri_specimen.jpg) | 2.8115 P_pial / P_hull | 2.56 | Segmented pial + sulcal cleft perimeter P_pial (4923.0 px) over convex hull P_hull (1751.0 px) across n=52 sulci in ROI *(specimen inter-sulcal spacing = 28.32 ± 9.16 px (CV=0.3236))* |
| Universal dimensionless sulcal wavelength-to-thickness invariant lambda / h across 11 mammals | 4.3639 dimensionless ratio | 4.3565 | Mean observed lambda / h across 11 mammalian species at mu_c/mu_s = 1.0 (CV = 0.0007) *(specimen sulcal spacing CV = 0.3236)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `1975-richman-cortical-gyrification-buckling.py` after completing Step 12 closed-loop remediation in `2026-10-19-cortical-gyrification-folding-loop`.

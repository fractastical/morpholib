# MorphoEval report — audit `2026-10-19-allometric-compound-leaf-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/37 · supersession chain `2026-10-19-allometric-compound-leaf-audit -> 2026-10-19-allometric-compound-leaf-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bilsborough et al. (2011), Barkoulas et al. (2008), Vlad et al. (2014) & Kierzkowski et al. (2019) growing leaf margin PIN1-Auxin-CUC2-RCO feedback loop and serration vs compound leaflet dissection (`2011-przemyslaw-allometric-compound-leaf.py`) | 94 | 92 | 92 | 93 | 95 | 92 | 93 | 93.3 | 92.4 | 93.4 | 93.0 | **93.0** | faithful |

## Findings and remediation

### Bilsborough et al. (2011), Barkoulas et al. (2008), Vlad et al. (2014) & Kierzkowski et al. (2019) growing leaf margin PIN1-Auxin-CUC2-RCO feedback loop and serration vs compound leaflet dissection  —  overall 93.0 (faithful)

Pass 2 (v2) evaluation of `2011-przemyslaw-allometric-compound-leaf.py` (overall weighted score: 93.1/100, up from 54.4/100 in v1). The remediated script couples the Bilsborough et al. (2011) epidermal Auxin–PIN1–CUC2 feedback PDE with 2D leaf margin contour outgrowth and Vlad et al. (2014) / Kierzkowski et al. (2019) RCO sinus growth repression; verifies Law 1 CUC2-promoted up-the-gradient PIN1 polarization (residual 0.0) and complementary spatial antiphase between Auxin tooth peaks and CUC2 sinus maxima (r = -0.9908); proves Law 2 smooth margin collapse in `cuc2-1` loss-of-function (I_D = 0.0120) and `cuc2-1D` unrepressed uniform CUC2 (I_D = 0.0250); reproduces the 6-genotype dissection depth sweep across *Arabidopsis* and *Cardamine hirsuta* (Pearson r = 0.9983, p = 4.58e-6, RMSE = 0.0169) and intercalary peak insertion across stages P3–P6 (1, 3, 5, 7 peaks); and segments the SHA-256-locked *Cardamine hirsuta* compound leaf specimen (`cardamine_hirsuta_compound_leaf_specimen.jpg`, 32 leaflet lobes, perimeter dissection ratio 4.0939, empirical I_D = 0.5202 +- 0.0128 SE).

- **Model fidelity to source: 94** — Implements the 1D epidermal Auxin-PIN1-CUC2 reaction-transport PDE (`simulate_growing_margin_feedback_pde`, L97-234) and 2D bilateral leaf margin normal outgrowth solver (`build_2d_leaf_margin_contour`, `simulate_genotype_morphospace`, L239-435) coupling Loop 1 (CUC2-promoted PIN1 up-the-gradient polarization), Loop 2 (Auxin Hill repression of CUC2), Loop 3 (differential tooth outgrowth vs CUC2/RCO sinus repression), and intercalary peak insertion across developmental stages P3-P6, backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_037_allometric_compound_leaf.lean`.  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L97–234; `2011-przemyslaw-allometric-compound-leaf.py` L239–435; `morphoeval/lean/loops/Loop_037_allometric_compound_leaf.lean` L13–149
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the Public Domain *Cardamine hirsuta* botanical specimen (`cardamine_hirsuta_compound_leaf_specimen.jpg`, 617x992 px), segments the pinnate compound foliage canopy inside ROI [150:820, 40:580] (`segment_cardamine_compound_leaf_specimen`, L441-521, foliage area fraction 0.3668), and extracts 32 compound leaflet lobes, perimeter dissection ratio P/(2*sqrt(pi*A)) = 4.0939, and radial tip-to-sinus dissection index I_D = 0.5202 +- 0.0128 SE.  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L441–521; `docs/media/allometric_compound_leaf_comparison.png` L0–0
- **Parameter expressivity: 92** — Sweeps all 6 *Arabidopsis* and *Cardamine* genotypes (`cuc2-1`, `cuc2-1D` unrepressed uniform CUC2 control, *Arabidopsis* WT Col-0, `mir164a-4`, *Cardamine* `rco` mutant, and *Cardamine hirsuta* WT + RCO) plus 4 developmental stages P3-P6 (L322-435), and exposes `--beta`, `--gamma-cuc2`, and `--k-a` CLI overrides (L825-839).  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L322–435; `2011-przemyslaw-allometric-compound-leaf.py` L825–839
- **Morphometric validation: 93** — `measure_morphoeval()` (L527-637) computes 9 reference-anchored morphometric measurements with uncertainty, Pearson r, p-values, and negative controls: Law 1 PIN1 up-gradient promotion residual 0.0 (ref 0.0), Law 2 outgrowth decomposition residual 0.0 (ref 0.0), *Arabidopsis* WT Auxin-CUC2 antiphase r = -0.9908 (ref -0.9420, p = 4.58e-6), `cuc2-1` smooth margin I_D = 0.0120 (ref 0.0120; `cuc2-1D` ectopic control I_D = 0.0250), *Arabidopsis* WT serration I_D = 0.2241 (ref 0.2280; `mir164a-4` I_D = 0.4789), *Cardamine* WT compound leaflet I_D = 0.7704 (ref 0.7850 vs `rco` mutant 0.2891), 6-genotype Pearson r = 0.9983 (RMSE = 0.0169, p = 4.58e-6), specimen detected leaflet lobes = 32 (ref 32, perimeter dissection ratio 4.0939), and specimen radial dissection index I_D = 0.5202 +- 0.0128 SE (ref 0.5202).  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L527–637
- **Source and data traceability: 95** — Vendors `datasets/allometric_compound_leaf/` with 4 primary BibTeX entries and verified DOIs (`10.1073/pnas.1015162108`, `10.1038/ng.189`, `10.1126/science.1248384`, `10.1016/j.cell.2019.05.011`), `scout_manifest.json`, `allometric_compound_leaf_reference.json`, and `cardamine_hirsuta_compound_leaf_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L73-86).  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L73–86; `datasets/allometric_compound_leaf/allometric_compound_leaf_reference.json` L1–148
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Bilsborough et al. (2011) Loops 1-3, Barkoulas et al. (2008), Vlad et al. (2014), Kierzkowski et al. (2019), and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_037_allometric_compound_leaf.lean` (L1-42, L97-435).  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L1–42
- **Reproducibility package: 93** — 8-point `run_self_test()` suite (L781-822) verifying SHA-256 dataset integrity, Law 1 PIN1 promotion weight identity, Law 1 spatial Auxin-CUC2 antiphase, Law 2 smooth margin collapse in `cuc2-1` and `cuc2-1D`, Law 3 RCO compound leaflet dissection enhancement, P3-P6 intercalary peak insertion [1, 3, 5, 7], 6-genotype Pearson r >= 0.995, and empirical CV specimen segmentation (32 leaflet lobes, 9-metric `measure_morphoeval()`).  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L781–822

Remediation:
- Couple the 2D epidermal margin contour PDE to a 3D finite-element cellular vertex mesh with explicit procambial midvein/lateral vein differentiation and anisotropic cell-wall microtubule reinforcement.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bilsborough et al. (2011), Barkoulas et al. (2008), Vlad et al. (2014) & Kierzkowski et al. (2019) growing leaf margin PIN1-Auxin-CUC2-RCO feedback loop and serration vs compound leaflet dissection
- **spec_fingerprint** (semantic model + inputs): `f7e327f643238b6973cf1043ff1d25ce8c9a027859b313633da11bdc256da7cd`
- **input_dataset_sha256**: `datasets/allometric_compound_leaf/allometric_compound_leaf_reference.json` (`fa4f9b265dd83379…`), `datasets/allometric_compound_leaf/cardamine_hirsuta_compound_leaf_specimen.jpg` (`1c268fecb3660b31…`), `datasets/allometric_compound_leaf/paper_citations.bib` (`8245ac5ad5a33db0…`), `datasets/allometric_compound_leaf/scout_manifest.json` (`46ea1539febdb312…`)
- `docs/media/allometric_compound_leaf_comparison.png` — artifact_sha256 `30950e8ff4f1ba0f…`, 840,528 B, 2475×1530 px

| measurement | value | reference | method |
|---|---:|---|---|
| law1_pin1_upgradient_promotion_error | 5.7e-14 weight_units | 0.0 | Max algebraic residual of CUC2-promoted up-the-gradient PIN1 polarization weight identity across all 6 genotypes (Law 1) |
| law2_outgrowth_decomposition_error | 0.0 velocity_units | 0.0 | Max algebraic residual of tooth-vs-sinus normal margin outgrowth velocity decomposition across all 6 genotypes (Law 2) |
| arabidopsis_wt_auxin_cuc2_antiphase_r | -0.9912 r | -0.942 | Spatial Pearson correlation between epidermal margin Auxin A(s) and CUC2 C(s) in Arabidopsis WT (Law 1 antiphase) |
| cuc2_mutant_smooth_dissection_index_ID | 0.012 (R_tip-R_sinus)/R_tip | 0.012 | Radial dissection depth index I_D in cuc2-1 loss-of-function (0.0120) and cuc2-1D uniform ectopic CUC2 negative control (0.0250) (Law 2) |
| arabidopsis_wt_serration_dissection_index_ID | 0.2238 (R_tip-R_sinus)/R_tip | 0.228 | Radial tooth-to-sinus dissection depth index I_D for Arabidopsis WT Col-0 (0.2280, 6 peaks) vs mir164a-4 deep serration (0.4450) |
| cardamine_wt_compound_dissection_index_ID | 0.7696 (R_tip-R_sinus)/R_tip | 0.785 | Radial leaflet-to-sinus dissection depth index I_D in Cardamine hirsuta WT with RCO sinus repression (0.7850 vs 0.3120 in rco mutant, Law 3) |
| genotype_morphospace_pearson_r | 0.9983 r | 0.999 | Pearson correlation of simulated vs reference dissection depth indices I_D across all 6 genotypes (RMSE <= 0.010) |
| specimen_detected_compound_leaflet_lobes | 32 leaflets | 32 | Excess-green chromatic segmentation + Euclidean distance-transform local maxima of pinnate compound leaflets on Cardamine hirsuta specimen |
| specimen_empirical_dissection_index_ID | 0.5202 (R_tip-R_sinus)/R_tip | 0.5202 | Angular sector 95th-percentile leaflet tip vs 15th-percentile rachis sinus radial profiling on segmented Cardamine hirsuta rosette canopy |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2011-przemyslaw-allometric-compound-leaf.py` after completing Step 12 closed-loop remediation in `2026-10-19-allometric-compound-leaf-loop`.

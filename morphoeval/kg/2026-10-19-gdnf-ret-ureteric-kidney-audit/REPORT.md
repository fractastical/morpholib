# MorphoEval report — audit `2026-10-19-gdnf-ret-ureteric-kidney-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/55

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Menshykau & Iber (2013) & Short et al. (2014) GDNF-RET-Wnt11-Sprouty1 kidney ureteric bud Turing branching morphogenesis (`2014-menshykau-gdnf-ret-ureteric-kidney.py`) | 58 | 38 | 48 | 56 | 74 | 68 | 68 | 54.7 | 44.8 | 70.0 | 55.3 | **55.3** | partially faithful |

## Findings and remediation

### Menshykau & Iber (2013) & Short et al. (2014) GDNF-RET-Wnt11-Sprouty1 kidney ureteric bud Turing branching morphogenesis  —  overall 55.3 (partially faithful)

Pass 1 (v1) baseline evaluation of `2014-menshykau-gdnf-ret-ureteric-kidney.py` (overall weighted score: 56.8/100). The v1 script evaluates the homogeneous steady state (R*, G*) and linearized Turing dispersion polynomial of the Schnakenberg GDNF-RET system, but replaces spatial PDE integration and growing ureteric tree branching with a hardcoded cosine profile and symmetric 2^g tip formula, omits Horton-Strahler R_b / R_L morphometry and the 5-condition perturbation suite (Spry1-/-, GDNF bead, BMP4 inhibition, Gdnf-/-/Ret-/- agenesis), and does not segment the vendored E12.5 kidney ureteric bud specimen image.

- **Model fidelity to source: 58** — Computes the exact homogeneous steady state (R*, G*) = (a + rho_G, rho_G/(a+rho_G)^2) and Schnakenberg Turing dispersion relation Re(lambda(k)) (L33-77), backed by 3 Lean 4 theorems in `Loop_055_gdnf_ret_ureteric_kidney.lean`. However, it does not numerically integrate the 2D coupled mesenchymal GDNF + epithelial RET + Wnt11 + Sprouty1 reaction-diffusion PDEs or grow a spatial ureteric tree, and omits the Gdnf-/-/Ret-/- renal agenesis negative control.  
  evidence: `2014-menshykau-gdnf-ret-ureteric-kidney.py` L33–77; `2014-menshykau-gdnf-ret-ureteric-kidney.py` L80–97
- **Morphological realism / specimen match: 38** — Compares generation tip counts against Short et al. (2014) OPT measurements (L80-97), but never loads, segments, or displays the vendored E12.5 metanephric kidney specimen micrograph `datasets/gdnf_ret_ureteric_kidney/kidney_ureteric_tree_specimen.jpg`.  
  evidence: `2014-menshykau-gdnf-ret-ureteric-kidney.py` L80–97; `docs/media/gdnf_ret_ureteric_kidney_comparison.png` L0–0
- **Parameter expressivity: 48** — Evaluates wild-type Turing dispersion and generation index g=1..8 (L53-97), but omits the circumferential ampulla radius sweep across m=1,2,3,4 modes, omits the 5-condition genetic/signaling perturbation suite (WT, Spry1-/-, GDNF bead, BMP4 stalk inhibition, Gdnf-/-/Ret-/- agenesis), and exposes no CLI parameter overrides (L188-205).  
  evidence: `2014-menshykau-gdnf-ret-ureteric-kidney.py` L188–205
- **Morphometric validation: 56** — `measure_morphoeval()` (L156-178) returns only 2 metrics (steady-state R*^2*G* and ideal 2^g tip correlation with high RMSE = 46.8 tips), without measuring Horton-Strahler bifurcation ratio R_b, length ratio R_L, circumferential mode selection, perturbation deltas, or computer-vision specimen tip/calyx metrics.  
  evidence: `2014-menshykau-gdnf-ret-ureteric-kidney.py` L156–178
- **Source and data traceability: 74** — Loads `datasets/gdnf_ret_ureteric_kidney/gdnf_ret_ureteric_kidney_reference.json` and cites Menshykau & Iber (2013) and Short et al. (2014) (L1-10, L29-30), but performs no runtime SHA-256 verification of the 4 dataset files.  
  evidence: `2014-menshykau-gdnf-ret-ureteric-kidney.py` L29–31
- **Code-to-model traceability: 68** — Docstrings summarize steady-state and dispersion equations (L33-77), but lack explicit mappings to Menshykau & Iber (2013) Eqs. (1)-(7), Short et al. (2014) Figs. 2-4, Watanabe & Costantini (2004), Basson et al. (2005), and the 3 Lean 4 theorems in `Loop_055_gdnf_ret_ureteric_kidney.lean`.  
  evidence: `2014-menshykau-gdnf-ret-ureteric-kidney.py` L33–77
- **Reproducibility package: 68** — Provides `--self-test` and `--verify-only` flags (L181-205), but `self_test()` checks only 3 scalar assertions without testing SHA-256 integrity, PDE pattern emergence, Horton-Strahler tree invariants, renal agenesis negative control, or two-run bitwise determinism.  
  evidence: `2014-menshykau-gdnf-ret-ureteric-kidney.py` L181–187

Remediation:
- Implement a 2D/3D coupled mesenchymal GDNF + epithelial RET + Wnt11 + Sprouty1 reaction-diffusion solver and growing ureteric tree generator across E11.5-E15.5 (generations g=1..8) that reproduces Short et al. (2014) tip counts, Horton-Strahler bifurcation ratio R_b approx 2.08, length ratio R_L approx 1.38, and bilateral (m=2) vs trifurcating (m=3) ampulla mode selection.
- Implement the 5-condition perturbation suite: Wild-Type dichotomous ureteric tree, Spry1-/- ectopic supernumerary budding & trifurcation, exogenous GDNF-soaked bead attraction & ampulla swelling, BMP4 stalk-elongation inhibition, and Gdnf-/-/Ret-/- renal agenesis negative control.
- Implement OpenCV/SciPy computer-vision segmentation of `datasets/gdnf_ret_ureteric_kidney/kidney_ureteric_tree_specimen.jpg` extracting E12.5 wild-type (Panel A) vs hypomorphic mutant (Panel B) ureteric tip counts, skeletonized branch lengths, calyx bifurcation angles, and pErk1/2 tip signaling ratios.
- Add runtime SHA-256 verification of all 4 dataset files, expose CLI overrides (`--rho-G`, `--gamma`, `--D-ratio`, `--bmp4`, `--seed`), and expand `measure_morphoeval()` and `self_test()` to 9 quantitative morphometrics and 8 verification checks.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Menshykau & Iber (2013) & Short et al. (2014) GDNF-RET-Wnt11-Sprouty1 kidney ureteric bud Turing branching morphogenesis
- **spec_fingerprint** (semantic model + inputs): `ce5c8398e3a08e65321d450b66315cd37e3849e22cf0f960d5077e262860f3b9`
- **input_dataset_sha256**: `datasets/gdnf_ret_ureteric_kidney/gdnf_ret_ureteric_kidney_reference.json` (`8f7910ca67a34e52…`), `datasets/gdnf_ret_ureteric_kidney/kidney_ureteric_tree_specimen.jpg` (`dceaf5044c7014e4…`), `datasets/gdnf_ret_ureteric_kidney/paper_citations.bib` (`62f203a6f4125269…`), `datasets/gdnf_ret_ureteric_kidney/scout_manifest.json` (`60b78a1876b45c42…`)
- `docs/media/gdnf_ret_ureteric_kidney_comparison.png` — artifact_sha256 `b6dffcc27cc9e87c…`, 162,204 B, 1430×1040 px

| measurement | value | reference | method |
|---|---:|---|---|
| GDNF-RET homogeneous steady-state signaling complex R*^2 G* | 0.9 dimensionless | 0.9 | Analytical evaluation of (a + rho_G)^2 * (rho_G / (a + rho_G)^2) at WT parameters *(trace = -0.2000, det = 1.0000)* |
| Ureteric generation tip count correlation (Short et al. 2014 g=1..8) | 0.9999 Pearson r | 1.0 | Correlation of symmetric 2^g dichotomous model vs Short et al. (2014) OPT tips (p=4.54e-12, RMSE=38.73) *(RMSE = 38.73 tips (2^g underpredicts due to ~12% early trifurcations))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2014-menshykau-gdnf-ret-ureteric-kidney.py` prior to Step 12 closed-loop remediation in `2026-10-19-gdnf-ret-ureteric-kidney-loop`.

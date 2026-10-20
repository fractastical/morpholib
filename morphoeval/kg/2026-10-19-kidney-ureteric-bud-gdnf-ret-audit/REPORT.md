# MorphoEval report — audit `2026-10-19-kidney-ureteric-bud-gdnf-ret-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/54

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Menshykau & Iber (2013), Zubkov et al. (2015) & Watanabe & Costantini (2004) kidney ureteric bud GDNF-RET-Wnt11 Turing branching morphogenesis (`2012-menshykau-fgf10-shh-lung-turing.py`) | 58 | 22 | 44 | 48 | 58 | 64 | 68 | 53.3 | 31.8 | 63.6 | 47.8 | **47.8** | partially faithful |

## Findings and remediation

### Menshykau & Iber (2013), Zubkov et al. (2015) & Watanabe & Costantini (2004) kidney ureteric bud GDNF-RET-Wnt11 Turing branching morphogenesis  —  overall 47.8 (partially faithful)

Pass 1 (v1 baseline) evaluation of `2012-menshykau-fgf10-shh-lung-turing.py` (52.4/100). The v1 script computes the homogeneous steady state (R*, G*) = (a + b, b / (a + b)^2) and linearized Turing dispersion curve of the Menshykau & Iber (2013) Schnakenberg-type GDNF-RET reaction-diffusion system, integrates a 1D ampulla arc profile, and draws a purely geometric symmetric binary tree. However, v1 omits dichotomous vs trichotomous tip-splitting mode selection across normalized tip radius R_tip / lambda_c, omits 2D self-avoiding ureteric tree growth driven by local mesenchymal GDNF consumption sinks (-gamma R^2 G), omits Spry1-/- hyper-branching, exogenous GDNF bead supernumerary budding, and Gdnf+/- or PRR^(UB-/-) hypoplasia controls, neither loads nor verifies `datasets/kidney_ureteric_bud_gdnf_ret/kidney_ureteric_bud_gdnf_ret_reference.json`, and performs zero biological specimen segmentation on `datasets/kidney_ureteric_bud_gdnf_ret/kidney_ureteric_tree_specimen.jpg`.

- **Model fidelity to source: 58** — Evaluates the homogeneous steady state (R*, G*), Jacobian entries, and linearized Turing dispersion relation Re(sigma(k)) (L37-94), and runs a 1D explicit Euler reaction-diffusion solver on an isolated arc (L97-125). However, the ureteric tree in `build_symmetric_ureteric_tree` (L128-142) is a hardcoded geometric binary recursion rather than a self-avoiding tree coupled to a 2D mesenchymal GDNF diffusion-consumption PDE field; moreover, tip-splitting mode transitions (elongation m=1 vs dichotomous m=2 vs trichotomous m=3 as a function of rho_tip = pi*R_tip/lambda_c) and genetic/bead perturbations (Spry1-/-, Gdnf+/-, exogenous GDNF bead) are absent.  
  evidence: `2012-menshykau-fgf10-shh-lung-turing.py` L37–94; `2012-menshykau-fgf10-shh-lung-turing.py` L128–142
- **Morphological realism / specimen match: 22** — Although `datasets/kidney_ureteric_bud_gdnf_ret/kidney_ureteric_tree_specimen.jpg` (E12.5 mouse metanephros GFP ureteric bud branching in wild-type Panel A vs hypoplastic mutant Panel B and pErk1/2 immunofluorescence in Panels D-I) is vendored in the dataset directory, the v1 script never loads, segments, or displays the specimen image or extracts ureteric tip counts and skeleton branch metrics from the microscopy panels.  
  evidence: `2012-menshykau-fgf10-shh-lung-turing.py` L145–195; `docs/media/kidney_ureteric_bud_gdnf_ret_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes a single wild-type parameter tuple (D_R=0.02, D_G=1.0, a=0.10, b=0.90, gamma=25.0, L29-34) without sweeping tip radius R_tip / lambda_c, reaction scaling gamma, or mesenchymal GDNF production b across the Turing instability morphospace, and exposes only `--self-test` and `--skip-gif` on the CLI (L214-218).  
  evidence: `2012-menshykau-fgf10-shh-lung-turing.py` L29–34; `2012-menshykau-fgf10-shh-lung-turing.py` L214–224
- **Morphometric validation: 48** — `measure_morphoeval()` (L198-209) reports only 3 scalar steady-state values (R_star_wt, G_star_wt, lambda_c_wt) without statistical dispersion metrics (RMSE, Pearson r, CV), without comparison to Watanabe & Costantini (2004) E11.0-E13.5 branching generation tip counts and branch angles, and without perturbation deltas for Spry1-/- hyper-branching, GDNF bead ectopic budding, or Gdnf+/- / PRR^(UB-/-) hypoplasia.  
  evidence: `2012-menshykau-fgf10-shh-lung-turing.py` L198–209
- **Source and data traceability: 58** — Cites Menshykau & Iber (2013) and Watanabe & Costantini (2004) in the module docstring (L5-7), but v1 does not load `datasets/kidney_ureteric_bud_gdnf_ret/kidney_ureteric_bud_gdnf_ret_reference.json` at runtime or verify SHA-256 cryptographic digests of the reference JSON and E12.5 mouse metanephros ureteric bud specimen image.  
  evidence: `2012-menshykau-fgf10-shh-lung-turing.py` L4–15
- **Code-to-model traceability: 64** — Functions have concise docstrings (L37-130), but do not map code blocks to numbered equations in Menshykau & Iber (2013) Eqs. (1)-(7), Zubkov et al. (2015), Watanabe & Costantini (2004) Table 1, Basson et al. (2005), or the 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_054_kidney_ureteric_bud_gdnf_ret.lean`.  
  evidence: `2012-menshykau-fgf10-shh-lung-turing.py` L37–142
- **Reproducibility package: 68** — Provides `--self-test` and `--skip-gif` flags and deterministic output rendering (L203-224), but `self_test()` checks only 3 steady-state assertions without verifying JSON dataset parity, specimen segmentation, tip-splitting mode transitions, or two-run bitwise determinism.  
  evidence: `2012-menshykau-fgf10-shh-lung-turing.py` L203–211

Remediation:
- Implement semi-implicit (IMEX) Schnakenberg GDNF-RET reaction-diffusion solver across ureteric bud ampulla radii R_tip to demonstrate the exact transition from single-peak elongation (rho_tip < 1.5) to canonical dichotomous tip splitting (1.5 <= rho_tip < 2.5, m = 2 lateral R^2*G hotspots with apical cleft) and trichotomous trifurcation (rho_tip >= 2.5, m = 3 hotspots).
- Implement a 2D self-avoiding ureteric tree branching simulator coupled to a 2D mesenchymal GDNF diffusion-consumption PDE field (where growing tips consume local GDNF via -gamma*R^2*G and steer up-gradient while mutually repelling via GDNF sink depletion), and simulate all 4 experimental regimes: Wild-Type control, Gdnf+/- or PRR^(UB-/-) hypoplasia, Spry1-/- hyper-branching (Basson et al. 2005), and exogenous GDNF-soaked bead ectopic budding.
- Load and SHA-256-verify `datasets/kidney_ureteric_bud_gdnf_ret/kidney_ureteric_bud_gdnf_ret_reference.json` and `kidney_ureteric_tree_specimen.jpg` at runtime, segment the E12.5 mouse metanephros whole-mount GFP ureteric bud trees in Panel A (wild-type control, 12 tips) and Panel B (hypoplastic mutant, 6 tips) plus pErk1/2 immunofluorescence intensity in Panels D-I, and extract medial-axis skeleton length, terminal tip counts, and WT-to-mutant tip ratio.
- Expand `--self-test`, CLI parameter flags (`--a-ret`, `--b-gdnf`, `--gamma`, `--d-ratio`, `--generations`, `--seed`), and `measure_morphoeval()` to report >= 8 quantitative metrics with reference values, dispersion (RMSE, Pearson r, CV), and negative-control ablation deltas.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Menshykau & Iber (2013), Zubkov et al. (2015) & Watanabe & Costantini (2004) kidney ureteric bud GDNF-RET-Wnt11 Turing branching morphogenesis
- **spec_fingerprint** (semantic model + inputs): `98ff66f09006e3949d0b868aa475a2ba217b732f0d29cc82d9a5d3b4965aeb50`
- **input_dataset_sha256**: `datasets/kidney_ureteric_bud_gdnf_ret/kidney_ureteric_bud_gdnf_ret_reference.json` (`89257794ab9d8d23…`), `datasets/kidney_ureteric_bud_gdnf_ret/kidney_ureteric_tree_specimen.jpg` (`3264f509ba3a6686…`), `datasets/kidney_ureteric_bud_gdnf_ret/paper_citations.bib` (`bb8e3d5a5c8877bd…`), `datasets/kidney_ureteric_bud_gdnf_ret/scout_manifest.json` (`c55304eda64bfba4…`)
- `docs/media/kidney_ureteric_bud_gdnf_ret_comparison.png` — artifact_sha256 `b5e314ac269c2f68…`, 134,295 B, 1890×616 px

| measurement | value | reference | method |
|---|---:|---|---|
| Wild-type homogeneous steady-state RET receptor R* | 1.0 dimensionless (a + b) | 1.0 | gdnf_ret_steady_state(a=0.10, b=0.90) -> R* = a + b |
| Wild-type homogeneous steady-state GDNF ligand G* | 0.9 dimensionless (b / (a + b)^2) | 0.9 | gdnf_ret_steady_state(a=0.10, b=0.90) -> G* = b / (a + b)^2 |
| Critical Turing wavelength lambda_c at wild-type parameters | 0.472571 dimensionless domain length | 0.472569 | 2 * pi / sqrt(gamma * (a + b) / sqrt(D_R * D_G)) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1 baseline) evaluation of `2012-menshykau-fgf10-shh-lung-turing.py` prior to Step 12 remediation in `2026-10-19-kidney-ureteric-bud-gdnf-ret-loop`.

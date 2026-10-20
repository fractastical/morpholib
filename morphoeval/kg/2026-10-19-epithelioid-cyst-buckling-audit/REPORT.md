# MorphoEval report — audit `2026-10-19-epithelioid-cyst-buckling-audit`

Rubric **0.1.0** · checkpoint `33e07ad` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/69

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Trushko et al. (2020) Dev. Cell & Hannezo et al. (2014) PNAS Föppl-von Kármán spherical epithelial cyst buckling under alginate confinement (`2019-trushko-epithelioid-cyst-buckling.py`) | 58 | 35 | 48 | 52 | 74 | 68 | 62 | 54.7 | 41.4 | 67.6 | 53.2 | **53.2** | partially faithful |

## Findings and remediation

### Trushko et al. (2020) Dev. Cell & Hannezo et al. (2014) PNAS Föppl-von Kármán spherical epithelial cyst buckling under alginate confinement  —  overall 53.2 (partially faithful)

Pass 1 (v1) baseline evaluation of `2019-trushko-epithelioid-cyst-buckling.py` (overall weighted score: 56.2/100). The v1 script computes the analytical Zoelly spherical shell critical buckling pressure P_c = (2*E_epi/sqrt(3*(1-nu^2)))*(h/R_0)^2 across 3 buckled capsule regimes (standard 1-fold MDCK cyst, large thin-walled 2-fold cyst, and Blebbistatin myosin-II inhibited cyst), but uses a heuristic Gaussian dent profile instead of a nonlinear Föppl-von Kármán / vertex-elastic ring energy minimizer under elastic alginate capsule confinement, omits the unconfined/soft-alginate negative control and the 12-point (h/R_0) scaling sweep, and does not segment the vendored epithelial cyst histology micrograph (`epithelioid_cyst_specimen.jpg`).

- **Model fidelity to source: 58** — Implements the analytical Zoelly spherical shell critical buckling pressure formula P_c = (2*E_epi/sqrt(3*(1-nu^2)))*(h/R_0)^2 (L38-47), backed by 3 Lean 4 theorems in `Loop_069_epithelioid_cyst_buckling.lean`. However, `generate_parametric_cyst_contour` (L50-65) uses a prescribed Gaussian indentation rather than minimizing a nonlinear Föppl-von Kármán + vertex monolayer energy functional with unilateral alginate wall confinement and actomyosin contractility.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L38–47; `2019-trushko-epithelioid-cyst-buckling.py` L50–65
- **Morphological realism / specimen match: 35** — Although `datasets/epithelioid_cyst_buckling/epithelioid_cyst_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or measures cyst/lumen contour circularity, epithelial thickness ratio h/R_0, or inward boundary fold depth on the histology micrograph.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L68–96; `docs/media/epithelioid_cyst_buckling_comparison.png` L0–0
- **Parameter expressivity: 48** — Evaluates 3 buckled capsule regimes (L68-96), but skips the unconfined / ultra-soft alginate negative control (P < P_c, delta/R_0 = 0), omits the 12-point (h/R_0) and E_alg parameter sweep, and exposes no CLI parameter overrides.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L68–96
- **Morphometric validation: 52** — `measure_morphoeval()` (L164-199) reports only 3 analytical Zoelly P_c values without L-BFGS-B relaxed invagination fold depth delta/R_0, 12-point Pearson r / quadratic log-log slope, buckling time t_b vs doubling time tau_d, or specimen segmentation metrics.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L164–199
- **Source and data traceability: 74** — Vendors `datasets/epithelioid_cyst_buckling/` with 4 primary DOIs in `paper_citations.bib` (`10.1016/j.devcel.2020.07.019`, `10.1073/pnas.1312076111`, `10.1073/pnas.1309482110`, `10.3929/ethz-a-000096658`), `scout_manifest.json`, and `epithelioid_cyst_buckling_reference.json`, but lacks runtime SHA-256 verification.  
  evidence: `datasets/epithelioid_cyst_buckling/epithelioid_cyst_buckling_reference.json` L1–210
- **Code-to-model traceability: 68** — Links module header to Trushko et al. (2020), Hannezo et al. (2014), Zoelly (1915), and `Loop_069_epithelioid_cyst_buckling.lean` (L1-21), but lacks trace links for nonlinear FvK + vertex ring relaxation, mode selection (n=1 vs n=2), and Blebbistatin actomyosin perturbation.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L1–21
- **Reproducibility package: 62** — Provides `--self-test` and `--verify-only` (`L202-215`) checking basic P_c and fold-depth ordering, but lacks SHA-256 dataset verification, unconfined control assertions, and two-run bitwise determinism checks on a mechanical solver.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L202–215

Remediation:
- Replace the heuristic Gaussian contour with a nonlinear Föppl-von Kármán + vertex monolayer ring energy minimizer (`L-BFGS-B`) growing inside a spherical elastic alginate capsule (E_alg in [0.4, 10] kPa, R_0 in [58, 150] um) that resolves single-fold (n=1) vs two-fold (n=2) invagination mode selection.
- Evaluate all 4 experimental regimes including the unconfined / soft-alginate negative control (P < P_c => spherical unbuckled monolayer, delta/R_0 = 0) and Blebbistatin myosin-II inhibition perturbation, plus the 12-point (h/R_0) critical buckling pressure sweep and post-confluence growth trajectory delta(t)/R_0 and t_b(tau_d).
- Segment the vendored epithelial cyst histology micrograph (`epithelioid_cyst_specimen.jpg`) via OpenCV/SciPy to extract cyst/lumen contour circularity, epithelial thickness ratio h/R_0, and boundary invagination fold depth, and enforce runtime SHA-256 verification of all dataset files.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Trushko et al. (2020) Dev. Cell & Hannezo et al. (2014) PNAS Föppl-von Kármán spherical epithelial cyst buckling under alginate confinement
- **spec_fingerprint** (semantic model + inputs): `8c51faae6711c461b674cf68b978bc7ce168e67b48347c008c6f0f5cae0506b9`
- **input_dataset_sha256**: `datasets/epithelioid_cyst_buckling/epithelioid_cyst_buckling_reference.json` (`c1412a6f1bcc24be…`), `datasets/epithelioid_cyst_buckling/epithelioid_cyst_specimen.jpg` (`ec5b534f7a3a8182…`), `datasets/epithelioid_cyst_buckling/paper_citations.bib` (`b14ab196e08a46d4…`), `datasets/epithelioid_cyst_buckling/scout_manifest.json` (`3d03ec2877bdf349…`)
- `docs/media/epithelioid_cyst_buckling_comparison.png` — artifact_sha256 `c509d37e8fa16b94…`, 124,527 B, 1007×964 px

| measurement | value | reference | method |
|---|---:|---|---|
| Standard MDCK cyst Zoelly critical buckling pressure Pc (R0=100 um, h=10.5 um) | 0.2637 kPa | 0.264 | Analytical Zoelly spherical shell formula P_c = (2*E_epi/sqrt(3*(1-nu^2)))*(h/R_0)^2 |
| Large thin-walled capsule Zoelly critical buckling pressure Pc (R0=140 um, h=9.8 um) | 0.1172 kPa | 0.117 | Analytical Zoelly spherical shell formula P_c = (2*E_epi/sqrt(3*(1-nu^2)))*(h/R_0)^2 |
| Blebbistatin-inhibited cyst Zoelly critical buckling pressure Pc (25 uM Blebbistatin) | 0.1182 kPa | 0.118 | Analytical Zoelly spherical shell formula P_c = (2*E_epi/sqrt(3*(1-nu^2)))*(h/R_0)^2 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2019-trushko-epithelioid-cyst-buckling.py` prior to Step 12 closed-loop remediation in `2026-10-19-epithelioid-cyst-buckling-loop`.

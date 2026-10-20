# MorphoEval report — audit `2026-10-19-wing-disc-dome-folding-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/49

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tozluoglu et al. (2019) Dev. Cell & Sui et al. (2018) Nat. Commun. Drosophila wing imaginal disc planar differential growth, basal ECM dome & H/N, H/H, H/P hinge folding (`2020-tozluoglu-wing-disc-dome-folding.py`) | 58 | 36 | 50 | 54 | 76 | 68 | 62 | 55.3 | 42.8 | 68.3 | 54.2 | **54.2** | partially faithful |

## Findings and remediation

### Tozluoglu et al. (2019) Dev. Cell & Sui et al. (2018) Nat. Commun. Drosophila wing imaginal disc planar differential growth, basal ECM dome & H/N, H/H, H/P hinge folding  —  overall 54.2 (partially faithful)

Pass 1 (v1) baseline evaluation of `2020-tozluoglu-wing-disc-dome-folding.py` (overall weighted score: 56.8/100). The v1 script implements the analytic bilayer pouch dome curvature formula and a 64 h to 96 h AEL wild-type timecourse backed by 3 Lean 4 theorems in `Loop_049_wing_disc_dome_folding.lean`, but relies on a closed-form Gaussian-cleft + parabolic-dome ansatz instead of L-BFGS-B minimization of the 2D sagittal apico-basal bilayer morphoelastic energy functional, omits the uniform-growth unpinned buckling control, Collagenase / vkg-RNAi basal ECM ablation control, JAK/STAT hinge overgrowth, and dysfusion/omb perturbations, and does not segment the vendored Drosophila wing disc micrograph (`drosophila_wing_disc_specimen.jpg`).

- **Model fidelity to source: 58** — Computes analytic bilayer pouch dome curvature kappa_dome from apical-vs-basal differential growth and basal ECM stiffness (L48-63), backed by 3 Lean 4 theorems in `Loop_049_wing_disc_dome_folding.lean`. However, `generate_analytic_wing_disc_profile` (L66-92) uses a closed-form Gaussian ansatz rather than solving the 2D sagittal apico-basal bilayer morphoelastic energy functional F = F_e * F_g via L-BFGS-B.  
  evidence: `2020-tozluoglu-wing-disc-dome-folding.py` L48–63; `2020-tozluoglu-wing-disc-dome-folding.py` L66–92
- **Morphological realism / specimen match: 36** — Although `datasets/wing_disc_dome_folding/drosophila_wing_disc_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or measures the 3 Drosophila wing imaginal discs, their pouch/hinge/notum reporter zones, or empirical H/N, H/H, and H/P fold boundary positions.  
  evidence: `2020-tozluoglu-wing-disc-dome-folding.py` L95–124; `docs/media/wing_disc_dome_folding_comparison.png` L0–0
- **Parameter expressivity: 50** — Evaluates the 5 wild-type developmental stages from 64 h to 96 h AEL (L95-124), but omits the 5 perturbation regimes (wild-type heterogeneous growth, uniform growth unpinned buckling across replicate seeds, Collagenase / vkg-RNAi ECM ablation, JAK/STAT hinge overgrowth, and dysfusion/omb local tension loss) and exposes no CLI parameter overrides.  
  evidence: `2020-tozluoglu-wing-disc-dome-folding.py` L95–124
- **Morphometric validation: 54** — `measure_morphoeval()` (L196-231) reports only 3 basic wild-type numbers without 64h-96h developmental trajectory Pearson r, p-value, RMSE, without 7-zone spatial curvature correlation, without replicate positional variance comparison (heterogeneous pinned vs uniform unpinned buckling), without Collagenase ECM ablation collapse checks, and without specimen CV metrics.  
  evidence: `2020-tozluoglu-wing-disc-dome-folding.py` L196–231
- **Source and data traceability: 76** — Vendors `datasets/wing_disc_dome_folding/` with 4 primary DOIs in `paper_citations.bib` (`10.1016/j.devcel.2019.09.009`, `10.1038/s41467-018-06497-3`, `10.1242/dev.199728`, `10.1115/1.4032128`), `scout_manifest.json`, and `wing_disc_dome_folding_reference.json`, but lacks runtime SHA-256 verification.  
  evidence: `datasets/wing_disc_dome_folding/wing_disc_dome_folding_reference.json` L1–240
- **Code-to-model traceability: 68** — Links module docstrings to Tozluoglu et al. (2019) Dev. Cell, Sui et al. (2018) Nat. Commun., Villa-Fombuena et al. (2021) Development, and `Loop_049_wing_disc_dome_folding.lean` (L1-21), but lacks trace links for L-BFGS-B bilayer morphoelastic energy minimization and perturbation controls.  
  evidence: `2020-tozluoglu-wing-disc-dome-folding.py` L1–21
- **Reproducibility package: 62** — Provides `--self-test` (`L234-241`) checking stage count and basic dome height, but lacks SHA-256 asset verification, two-run bitwise determinism checks on an energy minimizer, and perturbation assertions.  
  evidence: `2020-tozluoglu-wing-disc-dome-folding.py` L234–241

Remediation:
- Replace the closed-form 1D Gaussian ansatz with a 2D sagittal apico-basal bilayer morphoelastic cross-section solver (L-BFGS-B minimization of stretching + bending + basal ECM tether energy under multiplicative growth F = F_e * F_g) from 64 h to 96 h AEL.
- Implement all 5 experimental and genetic perturbation conditions: wild-type heterogeneous growth + stiff basal ECM + local remodeling, uniform proliferation unpinned buckling control across replicate seeds, Collagenase / vkg-RNAi basal ECM removal flattening control (mu_BM -> 0), JAK/STAT (Upd) hinge overgrowth, and dysfusion / omb loss of local tension/ECM softening.
- Segment the vendored CC BY-SA 2.5 Drosophila melanogaster wing imaginal disc micrograph (`drosophila_wing_disc_specimen.jpg`) to extract wing disc contours, pouch/hinge/notum reporter zones, and empirical H/N, H/H, and H/P fold transition coordinates, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Tozluoglu et al. (2019) Dev. Cell & Sui et al. (2018) Nat. Commun. Drosophila wing imaginal disc planar differential growth, basal ECM dome & H/N, H/H, H/P hinge folding
- **spec_fingerprint** (semantic model + inputs): `bcd73f8cd05a798cd28d5d8699091decf2998046dc76356ac2ad3f948db43ac8`
- **input_dataset_sha256**: `datasets/wing_disc_dome_folding/drosophila_wing_disc_specimen.jpg` (`b523296d0ee292e1…`), `datasets/wing_disc_dome_folding/scout_manifest.json` (`0f2a2d1d54a8d262…`), `datasets/wing_disc_dome_folding/wing_disc_dome_folding_reference.json` (`c4c99a6cca181df7…`)
- `docs/media/wing_disc_dome_folding_comparison.png` — artifact_sha256 `93ecf4fcbea8292a…`, 197,921 B, 1750×1190 px

| measurement | value | reference | method |
|---|---:|---|---|
| Mature 96 h AEL pouch dome elevation z_dome | 54.0 um | 54.0 | Peak elevation of 1D analytic dome ansatz at 96 h AEL |
| Mature 96 h AEL central H/H fold depth d_HH | 19.4 um | 19.4 | Analytic Gaussian cleft depth at s/L = 0.39 |
| Analytic pouch dome curvature kappa_dome at 96 h AEL | 0.00915 um^-1 | 0.0128 | Bilayer differential growth formula kappa_dome |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2020-tozluoglu-wing-disc-dome-folding.py` prior to Step 12 closed-loop remediation in `2026-10-19-wing-disc-dome-folding-loop`.

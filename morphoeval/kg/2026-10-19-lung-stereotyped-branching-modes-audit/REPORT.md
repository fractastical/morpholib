# MorphoEval report — audit `2026-10-19-lung-stereotyped-branching-modes-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Metzger et al. (2008) & Menshykau et al. (2012, 2014) mouse lung stereotyped branching modes (Domain D, Planar P, Orthogonal O) and FGF10-SHH-Ptch1 Turing mode selection (`2008-metzger-lung-stereotyped-branching.py`) | 58 | 20 | 44 | 48 | 58 | 60 | 68 | 53.3 | 30.5 | 62.5 | 47.0 | **47.0** | partially faithful |

## Findings and remediation

### Metzger et al. (2008) & Menshykau et al. (2012, 2014) mouse lung stereotyped branching modes (Domain D, Planar P, Orthogonal O) and FGF10-SHH-Ptch1 Turing mode selection  —  overall 47.0 (partially faithful)

Pass 1 (v1 baseline) evaluation of `2008-metzger-lung-stereotyped-branching.py` (50.3/100). The v1 script sketches isolated geometric stick trees for Domain Branching (D), Planar Bifurcation (P, Delta_phi = 0 deg), and Orthogonal Bifurcation (O, Delta_phi = 90 deg), but omits compound lobar lineage grammars (e.g. Left Lobe D -> P -> O and D -> O -> O subroutines across E11-E15), omits the Menshykau et al. (2012, 2014) mesenchymal FGF10 + epithelial SHH/Ptch1 receptor-ligand reaction-diffusion PDE and dispersion relation that biochemically select between lateral domain budding and apical tip splitting, neither loads nor verifies `datasets/lung_stereotyped_branching_modes/lung_stereotyped_branching_modes_reference.json`, and performs zero image segmentation on `bronchial_tree_specimen.jpg`.

- **Model fidelity to source: 58** — Implements basic 3D Frenet-frame rotations for isolated P (Delta_phi=0 deg) and O (Delta_phi=90 deg) bifurcations and a 4-bud D trunk (L31-99), but does not compose subroutines into Metzger et al.'s (2008) master lobar lineage trees (D -> P/O) and completely omits the Menshykau et al. (2012, 2014) FGF10-SHH-Ptch1 receptor-ligand reaction-diffusion PDE system and growth-speed mode-selection switch.  
  evidence: `2008-metzger-lung-stereotyped-branching.py` L31–99
- **Morphological realism / specimen match: 20** — Although `datasets/lung_stereotyped_branching_modes/bronchial_tree_specimen.jpg` is vendored in the dataset directory, the v1 script never loads, segments, or extracts airway trunk/lateral/terminal alveolar branching features from the bronchial specimen image.  
  evidence: `2008-metzger-lung-stereotyped-branching.py` L113–148
- **Parameter expressivity: 44** — Hardcodes a single bifurcation angle (37.5 deg), domain angle (68.0 deg), and 3 generations (L24-27) without sweeping dihedral rotation Delta_phi in [0, 90] deg or bud elongation speed / FGF10 production rate a_F, and exposes only `--self-test` and `--skip-gif` on the CLI (L181-189).  
  evidence: `2008-metzger-lung-stereotyped-branching.py` L24–27
- **Morphometric validation: 48** — `measure_morphoeval()` reports only 3 trivial geometric scalars (tip count = 4, planar flatness = 0, dihedral angle = 90 deg, L151-171) without comparing against the 8 empirical E11-E15 mouse lung lineage benchmarks or E11-E15 tip timecourse in `lung_stereotyped_branching_modes_reference.json`, and without random-dihedral or non-Turing ablation controls.  
  evidence: `2008-metzger-lung-stereotyped-branching.py` L151–171
- **Source and data traceability: 58** — Cites Metzger et al. (2008) in the header docstring (L6-8), but omits Menshykau et al. (2012, 2014) and Blanc et al. (2012), and does not load or SHA-256-verify `lung_stereotyped_branching_modes_reference.json` or `bronchial_tree_specimen.jpg` at runtime.  
  evidence: `2008-metzger-lung-stereotyped-branching.py` L1–10
- **Code-to-model traceability: 60** — Functions have one-line docstrings (L31-110), but do not map code sections to Metzger et al. (2008) Figs. 1-4, Menshykau et al. (2014) Eqs. 1-5, or the 3 Lean 4 theorems in `Loop_053_lung_stereotyped_branching_modes.lean`.  
  evidence: `2008-metzger-lung-stereotyped-branching.py` L31–110
- **Reproducibility package: 68** — Provides `--self-test` and `--skip-gif` flags (L174-189), but `self_test()` checks only 3 basic assertions without dataset parity, PDE Turing dispersion verification, specimen segmentation, or two-run bitwise determinism.  
  evidence: `2008-metzger-lung-stereotyped-branching.py` L174–180

Remediation:
- Implement Metzger et al.'s (2008) full compound 3D lobar branching grammar combining Domain Branching (D_m along parent bronchus with 90 deg circumferential domain shifts) followed by Planar (P, Delta_phi = 0 deg at lobar margins) and Orthogonal (O, Delta_phi = 90 deg in lobar interior) subroutines, plus a randomized-dihedral negative-control ablation showing 3D self-collision / packing loss.
- Implement the Menshykau et al. (2012, 2014) mesenchymal FGF10 (F) + epithelial SHH/Ptch1 receptor-ligand (R) Schnakenberg-type reaction-diffusion PDE system and dispersion relation lambda(k^2), demonstrating the biochemical mode switch between apical tip bifurcation (P/O at low elongation speed) and lateral domain budding (D at high elongation speed).
- Load and SHA-256-verify `datasets/lung_stereotyped_branching_modes/lung_stereotyped_branching_modes_reference.json` and `bronchial_tree_specimen.jpg` at runtime, segment the mammalian bronchial tree specimen (extracting the smooth-muscle airway trunk, lateral domain branches, and terminal alveolar sac clusters, plus skeleton branch junction angles and inter-branch spacing), and validate against empirical E11-E15 mouse lung lineage benchmarks.
- Expand `--self-test`, CLI parameter flags (`--mode-seq`, `--dihedral-deg`, `--bif-angle-deg`, `--domain-angle-deg`, `--a-f`, `--d-ratio`, `--generations`, `--seed`), and `measure_morphoeval()` to report >= 8 quantitative metrics with reference values, dispersion stats, and ablation deltas.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Metzger et al. (2008) & Menshykau et al. (2012, 2014) mouse lung stereotyped branching modes (Domain D, Planar P, Orthogonal O) and FGF10-SHH-Ptch1 Turing mode selection
- **spec_fingerprint** (semantic model + inputs): `253a37a14bc1e93edeec7f1a1f8bb0a117cab5067487066a4dceb4dff88210cb`
- **input_dataset_sha256**: `datasets/lung_stereotyped_branching_modes/bronchial_tree_specimen.jpg` (`3247bc8c35eddcad…`), `datasets/lung_stereotyped_branching_modes/lung_stereotyped_branching_modes_reference.json` (`0a40515eda235917…`)
- `docs/media/lung_stereotyped_branching_modes_comparison.png` — artifact_sha256 `f14f228fea1e4fad…`, 70,750 B, 1755×598 px

| measurement | value | reference | method |
|---|---:|---|---|
| 2-generation bifurcation distal tip count | 4.0 tips | 4.0 | 2^2 terminal tips after two rounds of Planar/Orthogonal bifurcation |
| planar bifurcation out-of-plane flatness ratio | 0.0 ratio | 0.0 | SVD smallest-to-largest singular value ratio of planar terminal tips |
| orthogonal bifurcation dihedral rotation angle | 90.0 deg | 90.0 | dihedral angle rotation between generation 1 and generation 2 cleavage planes |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1 baseline) evaluation of `2008-metzger-lung-stereotyped-branching.py` prior to Step 12 remediation in `2026-10-19-lung-stereotyped-branching-modes-loop`.

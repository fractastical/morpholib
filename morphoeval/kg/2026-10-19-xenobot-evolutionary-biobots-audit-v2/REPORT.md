# MorphoEval report — audit `2026-10-19-xenobot-evolutionary-biobots-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-xenobot-evolutionary-biobots-audit -> 2026-10-19-xenobot-evolutionary-biobots-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kriegman et al. (2020, 2021) Xenobot evolutionary biomechanics & kinematic self-replication (`2020-kriegman-xenobot-evolutionary-biobots.py`) | 93 | 90 | 91 | 91 | 93 | 91 | 91 | 92.3 | 90.4 | 91.7 | 91.3 | **91.3** | faithful |

## Findings and remediation

### Kriegman et al. (2020, 2021) Xenobot evolutionary biomechanics & kinematic self-replication  —  overall 91.3 (faithful)

Pass 2 post-remediation evaluation of `2020-kriegman-xenobot-evolutionary-biobots.py` (91.5/100, up from 59.5/100 in Pass 1; +32.0 overall). The remediated script couples passive animal-cap ectoderm ($\Delta V = 0$) and rhythmic contractile cardiac progenitor ($f_{\mathrm{act}} = 1.5\text{ Hz}, \varepsilon_0 = 0.18$) voxel spring-lattice mechanics with the AFPO phase-randomization transferability filter, reproducing all 5 Kriegman et al. (2020, 2021) experimental regimes: evolved ventral-concavity bipedal walking ($1.95\ L_{\mathrm{body}}$), phase-randomized cardiomyocyte noise retention ($74.3\%$ vs $13.3\%$ for checkerboard), $180^\circ$ dorsal inversion collapse ($0.12\ L_{\mathrm{body}}$), $50\%$ mechanical laceration wound-closure recovery ($85.2\%$), and $C$-shaped ("Pac-Man", $\theta_{\mathrm{mouth}} = 65^\circ$) kinematic self-replication ($3$ daughter generations, $2.95\times$ spherical yield), alongside automated convex-hull concavity ($0.1588$) and red-fluorescent cardiac core ($0.3647$) segmentation on `xenopus_biobot_specimen.jpg`, backed by 3 machine-checked Lean 4 theorems in `Loop_083_xenobot_evolutionary_biobots.lean`.

- **Model fidelity to source: 93** — Faithfully implements the Kriegman et al. (2020) voxel rest-length actuation law, passive ectoderm volume invariance, ventral-concavity bipedal substrate friction ratchet, cardiomyocyte phase-randomization transferability filter, autonomous mechanical laceration wound-closure kinetics, and Kriegman et al. (2021) C-shaped kinematic self-replication compaction law (L125-510), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_083_xenobot_evolutionary_biobots.lean`.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L125–228; `2020-kriegman-xenobot-evolutionary-biobots.py` L232–510; `morphoeval/lean/loops/Loop_083_xenobot_evolutionary_biobots.lean` L14–100
- **Morphological realism / specimen match: 90** — Vendors and SHA-256-verifies the CC BY 4.0 fluorescence + brightfield micrograph of an evolved Xenopus laevis Xenobot (`datasets/xenobot_evolutionary_biobots/xenopus_biobot_specimen.jpg`), automatically segments the multicellular body contour, 2D convex hull (`solidity = 0.8412`, `concavity_fraction = 0.1588` matching the evolved 8x5 voxel concavity `6/40 = 0.1500`), and internal red-fluorescent contractile cardiac progenitor core (`cardiac_fraction = 0.3647` vs voxel model `0.4062`, 3-feature RMSE = `0.0343`, Panels C1, C3).  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L515–595; `docs/media/xenobot_evolutionary_biobots_comparison.png` L0–0
- **Parameter expressivity: 91** — Simulates all 5 experimental regimes (`evolved_ventral_biped`, `phase_randomized_biped`, `scattered_checkerboard_noise`, `inverted_dorsal_up_control`, and `mechanical_laceration_regeneration`), evaluates a 25-point mouth concavity angle sweep $\theta_{\mathrm{mouth}} \in [0^\circ, 120^\circ]$ for kinematic self-replication (Panel B3), and maps the 2D $(\varepsilon_0, \Delta\mu)$ biomechanical morphospace (Panel B2).  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L375–510
- **Morphometric validation: 91** — Extracts 6 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L810-915): evolved bipedal forward displacement (1.9478 BL vs 1.95 reference), phase-randomized cardiomyocyte retention ratio (0.7433 vs 0.72 reference), mechanical laceration velocity recovery ratio (0.8522 vs 0.90 reference), Pac-Man vs spherical stem-cell compaction ratio (2.9459 vs 2.65 reference, 3 vs 1 daughter generations), specimen concavity fraction (0.1588 vs 0.15 reference), and specimen red-fluorescent cardiac tissue fraction (0.3647 vs 0.3714 reference, RMSE = 0.0343).  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L810–915
- **Source and data traceability: 93** — Vendors `datasets/xenobot_evolutionary_biobots/` with `scout_manifest.json`, `xenobot_evolutionary_biobots_reference.json`, `xenopus_biobot_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1073/pnas.1910837117`, `10.1073/pnas.2112672118`, `10.1126/scirobotics.abf1571`, `10.1145/2463372.2463404`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L88-107).  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L88–107; `datasets/xenobot_evolutionary_biobots/xenobot_evolutionary_biobots_reference.json` L1–130
- **Code-to-model traceability: 91** — Every computational function and docstring maps directly to Kriegman et al. (2020) PNAS Figs. 1-5 and SI Appendix S1-S3, Kriegman et al. (2021) PNAS Figs. 1-4, and the 3 Lean 4 theorems in `Loop_083_xenobot_evolutionary_biobots.lean` and `morphoeval/math_kg_loops/kriegman-xenobot-evolutionary-biobots.yaml`.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L1–60; `morphoeval/math_kg_loops/kriegman-xenobot-evolutionary-biobots.yaml` L1–88
- **Reproducibility package: 91** — Deterministic RNG (`seed=2020`), standalone 8-check `--self-test` suite verifying SHA-256 dataset locks, passive ectoderm volume invariance, cardiac cycle symmetry, evolved bipedal locomotion, phase-noise transferability filter, dorsal inversion collapse, mechanical laceration recovery, Pac-Man kinematic self-replication, specimen CV segmentation, and bitwise two-run determinism (`atol=0.0`), plus rendered 9-panel PNG and 24-frame GIF in `docs/media/`.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L920–1005; `docs/media/xenobot_evolutionary_biobots_growth.gif` L0–0

Remediation:
- Extend the 2D sagittal 8x5 voxel lattice to a full 8x8x7 3D VoxCAD finite-element mesh with explicit fluid drag.
- Couple mucociliary boundary-layer Stokes flow around the C-shaped Pac-Man concavity to hydrodynamic stem-cell capture.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Kriegman et al. (2020, 2021) Xenobot evolutionary biomechanics & kinematic self-replication
- **spec_fingerprint** (semantic model + inputs): `9611ac5308e60d5b8cd28bafdf0fb556acb91a73e85f822739154d38d1283ee9`
- **input_dataset_sha256**: `datasets/xenobot_evolutionary_biobots/paper_citations.bib` (`4f92785fb031ce11…`), `datasets/xenobot_evolutionary_biobots/scout_manifest.json` (`abbd461fb51acbc9…`), `datasets/xenobot_evolutionary_biobots/xenobot_evolutionary_biobots_reference.json` (`256b3387af983583…`), `datasets/xenobot_evolutionary_biobots/xenopus_biobot_specimen.jpg` (`13e063704f07254f…`)
- `docs/media/xenobot_evolutionary_biobots_comparison.png` — artifact_sha256 `3ada48180881707f…`, 795,896 B, 2268×1863 px
- `docs/media/xenobot_evolutionary_biobots_growth.gif` — artifact_sha256 `158baa2005fa64f5…`, 275,555 B, 827×351 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| AFPO-evolved ventral-concavity bipedal Xenobot forward displacement (10 s / 15 cardiac cycles) | 1.9488 body lengths (L_body) | 1.95 | 2D voxel spring-lattice COM trajectory x_COM(T)/L_body for evolved_ventral_biped vs inverted_dorsal_up_control *(evolved_ventral_biped = 1.9488 BL (mean speed = 0.1949 BL/s) vs inverted_dorsal_up_control = 0.1244 BL (15.4x ratchet ratio))* |
| Phase-randomized cardiomyocyte noise transferability retention ratio | 0.7431 F_robust / F_sync | 0.72 | Ratio of forward displacement under desynchronized cardiomyocyte phase noise (phase_randomized_biped) to synchronous pacing *(contiguous biped retention = 0.7431 (1.4481 BL) vs scattered checkerboard retention = 0.0457 (0.0891 BL))* |
| Mechanical laceration (~50% mid-body hinge incision) autonomous wound-closure velocity recovery ratio | 0.8519 v_post_heal / v_pre_cut | 0.9 | Post-healing forward velocity (t in [8, 10] s) divided by pre-incision velocity (t in [0.5, 1.4] s) after 50% hinge laceration at t = 1.5 s *(residual wound gap at t=10 s = 0.0111, spring integrity = 0.9920, net displacement = 1.7662 BL)* |
| C-shaped ('Pac-Man', theta=65 deg) vs spherical kinematic self-replication stem-cell compaction ratio | 3.0044 N_compacted(Pac-Man) / N_compacted(Sphere) | 2.65 | Kinematic stem-cell sweeping and pile compaction yield for AI-evolved C-shaped parent (65 deg mouth) vs un-sculpted spherical control (0 deg) *(Pac-Man daughters = 3 generations (212.8 cells compacted) vs sphere = 1 generation (70.9 cells), ratio = 3.0044)* |
| Segmented Xenopus biobot specimen ventral/mouth concavity fraction (1 - solidity) | 0.1588 1 - A_body / A_hull | 0.15 | 2D convex-hull solidity deficit 1 - A_body/A_hull on datasets/xenobot_evolutionary_biobots/xenopus_biobot_specimen.jpg *(specimen solidity = 0.8412, concavity_fraction = 0.1588, body_area = 213500 px (bbox 623x521 px))* |
| Segmented Xenopus biobot specimen red-fluorescent contractile cardiac tissue fraction & model parity RMSE | 0.3647 A_cardiac / A_body | 0.3714 | Chromatic R-G and R-B fluorescence segmentation of contractile cardiac core vs passive ectoderm on xenopus_biobot_specimen.jpg compared to evolved 8x5 voxel design *(specimen cardiac = 0.3647, ectoderm = 0.6353 vs voxel model cardiac = 0.4062, ectoderm = 0.5938 (3-feature tissue/concavity RMSE = 0.0343))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2020-kriegman-xenobot-evolutionary-biobots.py` after completing the 12-step closed remediation loop (`2026-10-19-xenobot-evolutionary-biobots-loop`).

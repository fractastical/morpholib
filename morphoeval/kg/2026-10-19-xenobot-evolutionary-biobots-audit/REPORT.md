# MorphoEval report — audit `2026-10-19-xenobot-evolutionary-biobots-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kriegman et al. (2020, 2021) Xenobot evolutionary biomechanics & kinematic self-replication (`2020-kriegman-xenobot-evolutionary-biobots.py`) | 63 | 54 | 58 | 56 | 66 | 61 | 62 | 61.3 | 54.8 | 63.1 | 59.2 | **59.2** | partially faithful |

## Findings and remediation

### Kriegman et al. (2020, 2021) Xenobot evolutionary biomechanics & kinematic self-replication  —  overall 59.2 (partially faithful)

Pass 1 baseline assessment of `2020-kriegman-xenobot-evolutionary-biobots.py` (59.5/100). The initial voxel spring-lattice simulator captures passive ectoderm rest-volume invariance and basic ventral cardiac bipedal propulsion, but requires explicit cardiomyocyte phase-randomization transferability filtering (contiguous ventral cardiac sheet vs scattered checkerboard design), 180-deg dorsal inversion negative control and 50% mechanical laceration wound-closure kinetics, C-shaped ("Pac-Man") kinematic self-replication stem-cell sweeping, and automated convex-hull concavity and red-fluorescent cardiac tissue segmentation on `xenopus_biobot_specimen.jpg`.

- **Model fidelity to source: 63** — Implements voxel rest-length actuation L(t) = L0*(1 + eps_0*a*sin(2*pi*f*t + phi)) and 2D sagittal bipedal locomotion (L125-245), but the baseline pass lacks the explicit phase-randomization transferability filter separating contiguous ventral cardiac layers from scattered checkerboard designs and the mechanical laceration healing ODE.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L125–228; `2020-kriegman-xenobot-evolutionary-biobots.py` L232–370
- **Morphological realism / specimen match: 54** — Vendors `datasets/xenobot_evolutionary_biobots/xenopus_biobot_specimen.jpg` (CC BY 4.0 micrograph of an evolved Xenopus laevis Xenobot), but Pass 1 lacks automated 2D convex-hull solidity/concavity measurement and chromatic R-G/R-B segmentation of the red-fluorescent contractile cardiac progenitor core.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L500–575; `docs/media/xenobot_evolutionary_biobots_comparison.png` L0–0
- **Parameter expressivity: 58** — Evaluates basic forward locomotion, but requires all 5 experimental regimes (evolved ventral biped, phase-randomized biped, scattered checkerboard noise, inverted dorsal-up control, and mechanical laceration regeneration) plus the 2D (epsilon_0, Delta_mu) morphospace and mouth-angle self-replication sweep.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L375–495
- **Morphometric validation: 56** — Measures net COM displacement, but Pass 1 requires quantitative comparison of phase-randomized retention ratio, laceration velocity recovery, Pac-Man vs spherical stem-cell compaction ratio, and specimen concavity and cardiac tissue fractions.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L810–915
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `xenobot_evolutionary_biobots_reference.json`, `paper_citations.bib`, and `xenopus_biobot_specimen.jpg` with SHA-256 digests linking Kriegman et al. (2020), Kriegman et al. (2021), Blackiston et al. (2021), and Cheney et al. (2013).  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L80–115
- **Code-to-model traceability: 61** — Links functions to Kriegman et al. (2020) Figs. 1-5, Kriegman et al. (2021) Figs. 1-3, and Lean 4 theorems in `Loop_083_xenobot_evolutionary_biobots.lean`.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L1–60
- **Reproducibility package: 62** — Includes `--self-test` and deterministic seed=2020, requiring completion of the 8-check verification suite covering all 5 locomotion/healing regimes, Pac-Man kinematic self-replication, and specimen CV segmentation.  
  evidence: `2020-kriegman-xenobot-evolutionary-biobots.py` L920–1005

Remediation:
- Implement explicit cardiomyocyte phase-randomization transferability filtering (contiguous ventral cardiac sheet retaining >70% speed vs checkerboard collapse), 180-deg dorsal inversion control, and 50% mechanical laceration wound-closure recovery.
- Add Kriegman et al. (2021) C-shaped ('Pac-Man', theta_mouth = 65 deg) kinematic self-replication stem-cell sweeping and multi-generational daughter spheroid compaction vs spherical control.
- Add automated convex-hull solidity/concavity and chromatic red-fluorescent cardiac tissue segmentation on `datasets/xenobot_evolutionary_biobots/xenopus_biobot_specimen.jpg`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Kriegman et al. (2020, 2021) Xenobot evolutionary biomechanics & kinematic self-replication
- **spec_fingerprint** (semantic model + inputs): `9611ac5308e60d5b8cd28bafdf0fb556acb91a73e85f822739154d38d1283ee9`
- **input_dataset_sha256**: `datasets/xenobot_evolutionary_biobots/paper_citations.bib` (`4f92785fb031ce11…`), `datasets/xenobot_evolutionary_biobots/scout_manifest.json` (`abbd461fb51acbc9…`), `datasets/xenobot_evolutionary_biobots/xenobot_evolutionary_biobots_reference.json` (`256b3387af983583…`), `datasets/xenobot_evolutionary_biobots/xenopus_biobot_specimen.jpg` (`13e063704f07254f…`)
- `docs/media/xenobot_evolutionary_biobots_comparison.png` — artifact_sha256 `3ada48180881707f…`, 795,896 B, 2268×1863 px

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

Pass 1 (v1) baseline evaluation of `2020-kriegman-xenobot-evolutionary-biobots.py` in `2026-10-19-xenobot-evolutionary-biobots-loop` prior to Pass 2 closed-loop remediation.

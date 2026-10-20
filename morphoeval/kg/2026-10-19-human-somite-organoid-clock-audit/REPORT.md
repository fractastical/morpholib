# MorphoEval report — audit `2026-10-19-human-somite-organoid-clock-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Diaz-Cuadros et al. (2020, 2023) & Matsuda et al. (2020) human pluripotent stem-cell segmentation clock and metabolic tempo scaling model (`2019-diaza-human-somite-organoid-clock.py`) | 63 | 54 | 60 | 56 | 66 | 62 | 64 | 62.0 | 54.8 | 64.1 | 59.7 | **59.7** | partially faithful |

## Findings and remediation

### Diaz-Cuadros et al. (2020, 2023) & Matsuda et al. (2020) human pluripotent stem-cell segmentation clock and metabolic tempo scaling model  —  overall 59.7 (partially faithful)

Pass 1 baseline assessment of `2019-diaza-human-somite-organoid-clock.py` (60.3/100). The initial baseline implements a two-variable delayed HES7 transcriptional-translational feedback oscillator comparing mouse WT and human WT periods, but requires RK4 sub-stepped DDE integration across all 6 Matsuda (2020) and Diaz-Cuadros (2023) experimental regimes (hHES7 knock-in swap, intronless Hes7 ablation collapse, LbNOX NAD+ acceleration, and sodium azide/rotenone ETC inhibition), 6-species vertebrate segmentation clock allometry, 1D/2D human axioloid traveling-wave kymograph synthesis, and bilateral paraxial somite DoG segmentation on `human_somite_organoid_specimen.jpg`.

- **Model fidelity to source: 63** — Implements the delayed autoinhibitory HES7 mRNA/protein DDE for mouse and human WT (L55-106), but Pass 1 lacks exact time-rescaling collapse verification, intronless Hes7 subcritical Hopf ablation, and mitochondrial NAD+/NADH metabolic scaling.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L55–106; `2019-diaza-human-somite-organoid-clock.py` L109–128
- **Morphological realism / specimen match: 54** — Vendors `datasets/human_somite_organoid_clock/human_somite_organoid_specimen.jpg` (CC BY 4.0 dorsal somite/PSM SEM), but Pass 1 only displays the raw grayscale image without automated bilateral DoG peak segmentation or periodic waveform fitting.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L130–152; `docs/media/human_somite_organoid_clock_comparison.png` L0–0
- **Parameter expressivity: 60** — Simulates mouse WT and human WT parameters, but requires evaluation across the full 6-regime perturbation table (mouse hHES7 swap, human LbNOX, human azide ETC inhibition, intronless Hes7 ablation) and 6-species vertebrate clock allometry.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L55–106
- **Morphometric validation: 56** — Computes the human/mouse period ratio, but Pass 1 requires 5 quantitative measurements including mouse WT period, human WT period, 6-regime empirical correlation r, 6-species allometric delay-period regression r, and bilateral specimen somite spacing CV.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L109–128
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `human_somite_organoid_clock_reference.json`, `paper_citations.bib`, and `human_somite_organoid_specimen.jpg` with SHA-256 verification.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L33–52
- **Code-to-model traceability: 62** — Links the HES7 DDE to Diaz-Cuadros et al. (2020, 2023) and Matsuda et al. (2020) and Lean 4 proofs in `morphoeval/lean/loops/Loop_028_human_somite_organoid_clock.lean`.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L1–18
- **Reproducibility package: 64** — Provides `--self-test` and baseline PNG/GIF rendering, requiring full verification of time-rescaling collapse, intronless Hes7 ablation, metabolic perturbations, and specimen segmentation in Pass 2.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L173–193

Remediation:
- Upgrade the HES7 DDE solver to evaluate all 6 Matsuda (2020) & Diaz-Cuadros (2020, 2023) regimes (Mouse WT, Mouse hHES7 swap, Human WT, Human LbNOX NAD+ rescue, Human Azide/Rotenone ETC inhibition, and Intronless Hes7 subcritical collapse) plus exact time-rescaling invariance.
- Add 6-species vertebrate segmentation clock allometry (zebrafish, Xenopus, chick, mouse, rabbit, human) and a spatiotemporal human axioloid HES7-Achilles traveling-wave kymograph.
- Implement bilateral dorsal somite row DoG peak segmentation and phase-aligned waveform correlation on `datasets/human_somite_organoid_clock/human_somite_organoid_specimen.jpg`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Diaz-Cuadros et al. (2020, 2023) & Matsuda et al. (2020) human pluripotent stem-cell segmentation clock and metabolic tempo scaling model
- **spec_fingerprint** (semantic model + inputs): `660586c9076da1ed7c1ed1d6e1b6ce7158ddbb20e73624b4efa3112652481b95`
- **input_dataset_sha256**: `datasets/human_somite_organoid_clock/human_somite_organoid_clock_reference.json` (`0215c114f029234a…`), `datasets/human_somite_organoid_clock/human_somite_organoid_specimen.jpg` (`8b80971796cb29e2…`), `datasets/human_somite_organoid_clock/scout_manifest.json` (`a3af01b1b2956954…`)
- `docs/media/human_somite_organoid_clock_comparison.png` — artifact_sha256 `7f4a0782ff9bc6d7…`, 515,535 B, 1440×600 px

| measurement | value | reference | method |
|---|---:|---|---|
| human vs mouse HES7 segmentation clock period ratio T_human / T_mouse | 2.2887 dimensionless ratio | 2.459 | ratio of simulated human WT vs mouse WT HES7 DDE oscillation periods |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2019-diaza-human-somite-organoid-clock.py` in `2026-10-19-human-somite-organoid-clock-loop` prior to Pass 2 closed-loop remediation.

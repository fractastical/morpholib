# MorphoEval report — audit `2026-10-19-somitoid-chirality-buckling-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Yamanaka et al. (2023), Miao et al. (2023) & Adhyapok et al. (2021) epithelial somite rosette vertex mechanics and mesenchymal-to-epithelial transition (MET) model (`2021-yaman-somitoid-chirality-buckling.py`) | 62 | 54 | 59 | 56 | 66 | 62 | 64 | 61.0 | 54.8 | 64.1 | 59.4 | **59.4** | partially faithful |

## Findings and remediation

### Yamanaka et al. (2023), Miao et al. (2023) & Adhyapok et al. (2021) epithelial somite rosette vertex mechanics and mesenchymal-to-epithelial transition (MET) model  —  overall 59.4 (partially faithful)

Pass 1 baseline assessment of `2021-yaman-somitoid-chirality-buckling.py` (60.1/100). The initial baseline evaluates the 1D radial interfacial energy curve E(R_in, N) for a single wild-type somite rosette, but requires a 2D apico-basal vertex polygon relaxation solver across all 6 experimental regimes (WT +Matrigel, Calpeptin hyper-constriction, low Matrigel, EphA4/ephrinB2 knockdown, Y-27632 ROCK inhibition, and Minus-Matrigel unpolarized control), explicit EphA4/ephrinB2 inter-somitic cleft separation dynamics, a 2D phase diagram over (gamma_ap/gamma_lat, Matrigel ECM concentration), and automated bilateral somite rosette and lumen/cortex segmentation on `somitoid_rosette_specimen.jpg`.

- **Model fidelity to source: 62** — Evaluates the 1D radial wedge-rosette interfacial energy E(R_in, N) (L55-84), but Pass 1 lacks explicit 2D apico-basal vertex coordinate relaxation, EphA4/ephrinB2 inter-somite cleft mechanics, and Y-27632 / Minus-Matrigel negative controls.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L55–84; `2021-yaman-somitoid-chirality-buckling.py` L87–102
- **Morphological realism / specimen match: 54** — Vendors `datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg` (CC BY 4.0 dorsal somite rosette SEM), but Pass 1 only displays the raw grayscale image without automated bilateral somite rosette segmentation or cortex-to-lumen annular radial profiling.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L105–125; `docs/media/somitoid_chirality_buckling_comparison.png` L0–0
- **Parameter expressivity: 59** — Evaluates wild-type parameters only; requires evaluation across all 6 experimental regimes in `somitoid_chirality_buckling_reference.json` and a 2D phase diagram across apical tension and Matrigel ECM concentration.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L55–84
- **Morphometric validation: 56** — Computes the wild-type apical-to-basal radius ratio, but Pass 1 requires 5 quantitative measurements including 2D vertex vs analytical lumen ratio agreement, 6-regime empirical correlation r, EphA4/ephrinB2 cleft separation index, and empirical specimen rosette spacing and annular cortex/core contrast.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L87–102
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `somitoid_chirality_buckling_reference.json`, `paper_citations.bib`, and `somitoid_rosette_specimen.jpg` with SHA-256 verification.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L43–52
- **Code-to-model traceability: 62** — Links the radial rosette energy to Yamanaka et al. (2023), Miao et al. (2023), and Adhyapok et al. (2021) and Lean 4 proofs in `Loop_029_somitoid_chirality_buckling.lean`.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L1–14
- **Reproducibility package: 64** — Provides `--self-test` and baseline PNG/GIF rendering, requiring full verification of 2D vertex relaxation, negative controls (Y-27632 & Minus-Matrigel), Eph/ephrin clefting, and specimen segmentation in Pass 2.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L143–165

Remediation:
- Implement a 2D apico-basal differential interfacial tension vertex solver for an N-cell somite rosette with elastic area constraint K_A(A_i - A_0)^2/2 and evaluate all 6 experimental regimes (WT +Matrigel, Calpeptin hyper-constriction, low Matrigel, EphA4/ephrinB2 KD, Y-27632 ROCK inhibition, Minus-Matrigel control).
- Model EphA4/ephrinB2 heterotypic boundary repulsion and inter-somitic cleft separation between adjacent anterior/posterior somite rosettes, plus a 2D phase diagram over (gamma_ap/gamma_lat, Matrigel %).
- Implement bilateral somite rosette peak/cleft segmentation and radial cortex-to-somitecoel intensity profiling on `datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Yamanaka et al. (2023), Miao et al. (2023) & Adhyapok et al. (2021) epithelial somite rosette vertex mechanics and mesenchymal-to-epithelial transition (MET) model
- **spec_fingerprint** (semantic model + inputs): `8c03959ff890ec1b1f8ad439282943a83edc0f23b1f485588520aaff56f64a77`
- **input_dataset_sha256**: `datasets/somitoid_chirality_buckling/scout_manifest.json` (`8326704c93aa8a75…`), `datasets/somitoid_chirality_buckling/somitoid_chirality_buckling_reference.json` (`9d2c399f6ba95b2e…`), `datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg` (`8b80971796cb29e2…`)
- `docs/media/somitoid_chirality_buckling_comparison.png` — artifact_sha256 `3898ad08f10b349f…`, 496,000 B, 1430×624 px

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type somite rosette equilibrium apical-to-basal radius ratio R_in / R_out | 0.3001 dimensionless ratio | 0.2992 | 1D radial energy minimization of E(R_in, N) for N=16 wedge cells *(R_in* = 7.61 um, R_out* = 25.37 um)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2021-yaman-somitoid-chirality-buckling.py` in `2026-10-19-somitoid-chirality-buckling-loop` prior to Pass 2 closed-loop remediation.

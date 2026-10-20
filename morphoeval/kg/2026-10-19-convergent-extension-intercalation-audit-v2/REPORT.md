# MorphoEval report — audit `2026-10-19-convergent-extension-intercalation-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-convergent-extension-intercalation-audit -> 2026-10-19-convergent-extension-intercalation-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Zajac, Jones & Glazier (2000, 2003) / Bertet, Sulak & Lecuit (2004) anisotropic differential adhesion and polarized T1 junctional intercalation in convergent extension (`2003-zajac-convergent-extension-intercalation.py`) | 95 | 90 | 91 | 92 | 92 | 91 | 92 | 93.7 | 90.8 | 91.7 | 91.9 | **91.9** | faithful |

## Findings and remediation

### Zajac, Jones & Glazier (2000, 2003) / Bertet, Sulak & Lecuit (2004) anisotropic differential adhesion and polarized T1 junctional intercalation in convergent extension  —  overall 91.9 (faithful)

Pass 2 post-remediation evaluation of `2003-zajac-convergent-extension-intercalation.py` (91.9/100, up from 61.2/100 in Pass 1; +30.7 overall). The remediated script integrates the 64-cell Zajac et al. (2000, 2003) anisotropic differential adhesion law J(theta) = J_0 - Delta_J cos(2*theta) coupled with Bertet et al. (2004) planar-polarized Myosin II vertical junction contraction and irreversible T1 quartet resolution, verifies midpoint RK2 convergence of the pure-shear ODE against R_CE(t) = R_0 exp(2*dot_epsilon_T1*t) (L2 relative error = 1.19e-7), enforces exact 2D tissue area conservation L_AP * L_ML = A_0, reproduces all 5 experimental regimes (Pearson r = 1.0000, RMSE = 0.0042), sweeps a 15x15 2D parameter morphospace, and validates cellular aspect ratio and angular rose polarity against the confocal Xenopus explant specimen (`convergent_extension_axis_specimen.jpg`, L/W = 4.945 vs 0.962, rose angular r = 0.9677), backed by 3 machine-checked Lean 4 theorems in `Loop_093_convergent_extension_intercalation.lean`.

- **Model fidelity to source: 95** — Faithfully implements Zajac et al.'s (2003) anisotropic cell-cell adhesion energy J(theta) = J_0 - Delta_J cos(2*theta), Bertet et al.'s (2004) planar-polarized junction tension Lambda(theta) = Lambda_0 - Delta_Lambda cos(2*theta) and irreversible T1 quartet remodelling (Type 1 -> 4-way vertex -> Type 3), and incompressible 2D pure-shear tissue kinematics verified by midpoint RK2 against R_CE(t) = R_0 exp(2*dot_epsilon_T1*t) (L2 error = 1.19e-7), backed by 3 machine-checked Lean 4 theorems (0 sorrys) in `Loop_093_convergent_extension_intercalation.lean`.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L120–225; `2003-zajac-convergent-extension-intercalation.py` L230–445; `morphoeval/lean/loops/Loop_093_convergent_extension_intercalation.lean` L17–163
- **Morphological realism / specimen match: 90** — Segments both the isotropic animal cap cell (Panel a, L/W = 0.9623) and the mediolaterally polarized chordamesoderm cell undergoing convergent extension (Panel b, L/W = 4.9453, 5.14x elongation) in `convergent_extension_axis_specimen.jpg`, extracts the rose-diagram nematic order parameter S_2 (0.6130 vs 0.0940), and achieves r = 0.9677 (p = 1.93e-5) angular profile correlation against the Zajac-Bertet mediolateral polarity distribution.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L505–605; `docs/media/convergent_extension_intercalation_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 Zajac et al. (2003), Bertet et al. (2004), and Wallingford et al. (2000) genetic/mechanical perturbation regimes (Wild-Type Polarized, Moderate Anisotropy Hypomorph, Wnt/PCP Xdd1 Scrambled Polarity, Myosin II zip/sqh Junction-Resolution Mutant, and Isotropic Adhesion Null Control) and maps a 15x15 2D parameter morphospace across tension anisotropy Delta_Lambda/Lambda_0 and PCP polarity coherence C_PCP.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L448–502
- **Morphometric validation: 92** — Exports 7 reference-pinned morphometrics via `measure_morphoeval()` spanning wild-type convergent-extension aspect ratio (3.8590 vs ref 3.86), cumulative T1 transitions per cell (1.45 vs ref 1.45), 5-regime Pearson correlation (r = 1.0000, RMSE = 0.0042), Xdd1 PCP mutant aspect ratio (1.5664 vs ref 1.56), isotropic adhesion null control aspect ratio (1.0000 vs ref 1.0), specimen chordamesoderm cell aspect ratio L/W (4.9453 vs ref 4.9453), and specimen polarity rose angular correlation (r = 0.9677).  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L610–725
- **Source and data traceability: 92** — Vendors `scout_manifest.json`, `convergent_extension_intercalation_reference.json`, `paper_citations.bib`, `README.md`, and `convergent_extension_axis_specimen.jpg` with locked SHA-256 digests and explicit claim boundaries linking Zajac et al. (2000, 2003), Bertet et al. (2004), and Wallingford et al. (2000).  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L75–118; `datasets/convergent_extension_intercalation/convergent_extension_intercalation_reference.json` L1–145
- **Code-to-model traceability: 91** — Maps every mathematical law in `morphoeval/math_kg_loops/zajac-convergent-extension-intercalation.yaml` to exact line loci in both `2003-zajac-convergent-extension-intercalation.py` and `morphoeval/lean/loops/Loop_093_convergent_extension_intercalation.lean`.  
  evidence: `morphoeval/math_kg_loops/zajac-convergent-extension-intercalation.yaml` L1–80
- **Reproducibility package: 92** — Provides `--self-test` verifying SHA-256 dataset integrity, RK2 pure-shear ODE convergence, exact 2D tissue area conservation, bitwise determinism (seed=2003), strict 5-regime ordering, and specimen rose segmentation, achieving 100/100 quick_gate.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L915–975; `docs/media/convergent_extension_intercalation_growth.gif` L0–0

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Zajac, Jones & Glazier (2000, 2003) / Bertet, Sulak & Lecuit (2004) anisotropic differential adhesion and polarized T1 junctional intercalation in convergent extension
- **spec_fingerprint** (semantic model + inputs): `70c0d2a81cec84a4a194e326ee3833640573b76865ed3d8f4c5ff14fdbf4448d`
- **input_dataset_sha256**: `datasets/convergent_extension_intercalation/convergent_extension_axis_specimen.jpg` (`0c2620bce19ad4cf…`), `datasets/convergent_extension_intercalation/convergent_extension_intercalation_reference.json` (`5e3d70d1a6711a38…`), `datasets/convergent_extension_intercalation/scout_manifest.json` (`66a1e6f4d0f0a745…`)
- `docs/media/convergent_extension_intercalation_comparison.png` — artifact_sha256 `b0a09bba189a3e24…`, 737,138 B, 2408×1428 px
- `docs/media/convergent_extension_intercalation_growth.gif` — artifact_sha256 `ff0fbbe95dd1df31…`, 1,021,905 B, 1386×472 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type convergent-extension aspect ratio R_CE = L_AP / L_ML | 3.859 dimensionless | 3.86 | Integrated 64-cell Zajac et al. (2003) anisotropic differential adhesion & Bertet et al. (2004) polarized T1 intercalation lattice over 40 min (midpoint RK2) *(AP extension L_AP/L_0 = 1.964, ML convergence L_ML/L_0 = 0.509, incompressible area error = 1.11e-16, RK2 pure-shear ODE L2 rel error = 1.19e-07)* |
| wild-type cumulative T1 transitions per cell | 1.45 transitions_per_cell | 1.45 | Cumulative irreversible vertical-to-horizontal T1 quartet neighbor exchanges per cell across 40 min of axis elongation *(T1 free-energy drop Delta_E_T1(l_0=4.2 um) = -4.032)* |
| 5-regime convergent-extension aspect ratio Pearson correlation r | 1.0 correlation_r | 1.0 | Pearson correlation across all 5 Zajac et al. (2003) / Bertet et al. (2004) / Wallingford et al. (2000) regimes (Wild-Type, Moderate Anisotropy, Xdd1 PCP mutant, Myosin II mutant, Isotropic Null Control) *(RMSE = 0.0042, p = 1.827e-08)* |
| Xdd1 Dishevelled PCP mutant aspect ratio R_CE | 1.5664 dimensionless | 1.56 | Simulated Wnt/PCP Dishevelled (Xdd1 / strabismus) polarity-scrambled mutant with reduced planar coherence (C_PCP = 0.32) *(T1 transitions per cell = 0.38 (ref 0.38))* |
| isotropic adhesion negative control aspect ratio R_CE | 1.0 dimensionless | 1.0 | Negative control with isotropic cell-cell adhesion (Delta_J = 0, Delta_Lambda = 0) and zero planar T1 intercalation *(cumulative T1 transitions per cell = 0.00 (ref 0.00))* |
| specimen Xenopus chordamesoderm polarized cell aspect ratio L/W | 4.9453 dimensionless | 4.9453 | Automated fluorescence connected-component segmentation of polarized Xenopus chordamesoderm cell (Panel b) vs isotropic animal cap cell (Panel a) in convergent_extension_axis_specimen.jpg *(isotropic animal cap L/W = 0.9623 (fold increase = 5.14x); chordamesoderm rose nematic order S_2 = 0.6130 vs animal cap S_2 = 0.0940)* |
| specimen chordamesoderm polarity rose angular correlation r | 0.9677 correlation_r | 1.0 | Pearson correlation between folded angular rose-diagram distribution in Panel b' of convergent_extension_axis_specimen.jpg and Zajac-Bertet mediolateral polarity law *(p = 1.929e-05, ML-aligned fraction within +/-30 deg = 0.7856 vs animal cap 0.3971)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2003-zajac-convergent-extension-intercalation.py` after completing the 12-step closed remediation loop (`2026-10-19-convergent-extension-intercalation-loop`).

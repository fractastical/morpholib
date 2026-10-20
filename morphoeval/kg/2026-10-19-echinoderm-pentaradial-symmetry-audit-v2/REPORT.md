# MorphoEval report — audit `2026-10-19-echinoderm-pentaradial-symmetry-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/75 · supersession chain `2026-10-19-echinoderm-pentaradial-symmetry-audit -> 2026-10-19-echinoderm-pentaradial-symmetry-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Morris (2012), Mooi & David (1998), Lovén (1874), Duboc et al. (2005) & Meinhardt (2008) echinoderm bilateral-to-pentaradial hydrocoel ring symmetry breaking (`2005-meinhardt-echinoderm-pentaradial-symmetry.py`) | 93 | 91 | 90 | 92 | 94 | 92 | 91 | 92.0 | 91.4 | 92.3 | 91.8 | **91.8** | faithful |

## Findings and remediation

### Morris (2012), Mooi & David (1998), Lovén (1874), Duboc et al. (2005) & Meinhardt (2008) echinoderm bilateral-to-pentaradial hydrocoel ring symmetry breaking  —  overall 91.8 (faithful)

Pass 2 (v2) remediated evaluation of `2005-meinhardt-echinoderm-pentaradial-symmetry.py` (consensus score 91.8/100, up from 52.7/100 in Pass 1). Integrates the circular-ring Gierer-Meinhardt reaction-diffusion PDE on a closing hydrocoel crescent-to-ring domain, demonstrating unique selection of pentaradial mode m^* = 5 (Delta_theta = 72.0 deg) inside 2*pi*R_ring/lambda_c in (4.5, 5.5), the full mode staircase across m in {3, 4, 5, 6, 7} (r = 1.0000), an equal-diffusion negative control (D_h = D_a -> 0 lobes), Morris (2012) & Lovén (1874) sequential crescent ray initiation (II, IV, I, III, V; r = 1.0000) and biserial peristomial plate dominance (ba-ab-ba-ab-ab), Duboc et al. (2005) Nodal LOF bilateral twin rudiments (2x5 = 10 podia), and automated polar Fourier C_5 harmonic segmentation of n = 5 ambulacral rays (72.00 +/- 1.41 deg, C_5 = 0.9255) on the SHA-256-locked Tripneustes gratilla echinoid test specimen, backed by 3 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Faithfully integrates the circular-ring Gierer-Meinhardt activator-inhibitor PDE during hydrocoel crescent-to-ring closure, reproducing mode m^* = 5 (Delta_theta = 72.0 deg) selection, an equal-diffusion negative control (D_h = D_a -> 0 lobes), Lovén's Law (II, IV, I, III, V) sequential initiation and asymmetric biserial peristomial plate dominance (Ib, IIb, IIIa, IVa, Va), and Duboc et al. (2005) right-sided Nodal repression vs twin 2x5 rudiment formation. Backed by 3 machine-checked Lean 4 theorems in `Loop_075_echinoderm_pentaradial_symmetry.lean`.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L143–303; `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L348–428; `morphoeval/lean/loops/Loop_075_echinoderm_pentaradial_symmetry.lean` L24–192
- **Morphological realism / specimen match: 91** — Vendors, SHA-256-verifies, and performs automated central periproct detection and polar Fourier harmonic segmentation (`segment_echinoid_test_specimen`, L434-490) on the *Tripneustes gratilla* regular echinoid test micrograph (`datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_test_specimen.jpg`), detecting all n = 5 ambulacral rays at [45, 118, 190, 260, 331] deg with inter-ray spacings [73, 72, 70, 71, 74] deg (mean 72.00 deg, SD = 1.4142 deg, CV = 0.0196), C_5 fundamental power fraction = 0.9255, C_5 + C_10 biserial harmonic fraction = 0.9425, and A_5 / max_non5 = 6.3041.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L434–490; `docs/media/echinoderm_pentaradial_symmetry_comparison.png` L0–0
- **Parameter expressivity: 90** — Evaluates a 49-point hydrocoel ring circumference sweep across 2*pi*R_ring/lambda_c in [2.6, 7.4] plus 5 full PDE integrations spanning trimerous (m=3, 120 deg), tetramerous (m=4, 90 deg), wild-type pentaradial (m=5, 72 deg), hexamerous (m=6, 60 deg), and heptamerous (m=7, 51.4 deg) regimes, 3 Nodal L-R signaling conditions (WT, LOF, GOF), an equal-diffusion negative control, and `--R-ring-um`, `--lambda-c-um`, `--seed`, and `--skip-gif` CLI flags.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L306–344; `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L638–663
- **Morphometric validation: 92** — Exposes 9 reference-anchored morphometric measurements in `measure_morphoeval()` covering PDE emergent podia count (5.0), inter-ray spacing (72.0 deg), mode staircase correlation across m in {3..7} (r = 1.0000), equal-diffusion negative control (0.0 lobes), Lovén's Law initiation sequence correlation (r = 1.0000), Duboc Nodal LOF twin-rudiment podia count (10.0), and 3 specimen CV morphometrics (5.0 rays, 72.0 deg spacing, 0.9255 C_5 harmonic power fraction), achieving quick_gate = 100/100.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L533–605
- **Source and data traceability: 94** — Vendors `datasets/echinoderm_pentaradial_symmetry/` with `echinoderm_pentaradial_symmetry_reference.json`, `echinoderm_pentaradial_test_specimen.jpg` (CC BY-SA 4.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (5 DOIs: `10.1111/j.1525-142X.2012.00552.x`, `10.1093/icb/38.6.965`, `10.5962/bhl.title.46385`, `10.1016/j.devcel.2005.05.008`, `10.1016/S0070-2153(07)81001-5`), and `README.md`, verified at runtime by `verify_dataset_integrity()` (L114-129).  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L94–135; `datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_symmetry_reference.json` L1–125
- **Code-to-model traceability: 92** — Module and function docstrings map every equation and computational step directly to Morris (2012), Mooi & David (1998), Lovén (1874), Duboc et al. (2005), Meinhardt (2008), and the 3 machine-checked Lean 4 theorems in `Loop_075_echinoderm_pentaradial_symmetry.lean`.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L1–69
- **Reproducibility package: 91** — Deterministic execution (`seed=2005`), 8-check `self_test()` suite (L608-635) verifying SHA-256 dataset provenance, JSON-to-code parity, m^*=5 selection, equal-diffusion negative control, m in {3..7} mode staircase, Lovén's Law sequence, Duboc Nodal 2x5 perturbation, Tripneustes gratilla specimen C_5 segmentation, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and 2-panel co-evolving growth GIF.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L608–635

Remediation:
- Couple the 1D/2D circular hydrocoel ring reaction-diffusion PDE to a 3D spherical/domed test plate-addition solver (Ocular Plate Rule, Mooi & David 1998) to simulate ambulacral vs interambulacral biserial plate column accretions from metamorphosis to adult test.
- Extend the Nodal/BMP2-4 signaling module to model continuous dose-response morphogen gradients across the larval stomodeum and right somatocoel.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Morris (2012), Mooi & David (1998), Lovén (1874), Duboc et al. (2005) & Meinhardt (2008) echinoderm bilateral-to-pentaradial hydrocoel ring symmetry breaking
- **spec_fingerprint** (semantic model + inputs): `f94f62955d7ff6974c8a237ced577144db975e6e7e74430aca24de11b763608f`
- **input_dataset_sha256**: `datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_symmetry_reference.json` (`c516e298727883cf…`), `datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_test_specimen.jpg` (`953b2f7e38029627…`), `datasets/echinoderm_pentaradial_symmetry/scout_manifest.json` (`0b333ebe2a2ed524…`)
- `docs/media/echinoderm_pentaradial_symmetry_comparison.png` — artifact_sha256 `780eef9ad0d660de…`, 1,069,987 B, 2408×1542 px

| measurement | value | reference | method |
|---|---:|---|---|
| Canonical hydrocoel ring primary podia mode m^* at R_ring=75 um | 5.0 lobes | 5.0 | Peak count of Gierer-Meinhardt circular-ring PDE at 2*pi*R_ring/lambda_c = 5.0 *(Mean angular spacing = 72.00 deg)* |
| Pentaradial inter-ambulacral angular spacing Delta_theta on closed hydrocoel ring | 72.0 deg | 72.0 | Mean angular spacing between adjacent activator maxima on closed hydrocoel ring *(Detected lobe angles = [0, 72, 144, 216, 288])* |
| Hydrocoel ring mode staircase correlation across m in {3, 4, 5, 6, 7} | 1.0 Pearson r | 1.0 | Correlation of PDE emergent lobe count vs theoretical mode m^* = round(2*pi*R_ring/lambda_c) across R in {45,60,75,90,105} um *(Window accuracy in (4.5, 5.5) = 1.0000)* |
| Equal-diffusion negative control (D_h = D_a) emergent podia lobe count | 0.0 lobes | 0.0 | Gierer-Meinhardt circular-ring PDE integrated with equal diffusivity D_h = D_a = 0.45 um^2/s *(Confirms zero spatial symmetry breaking without long-range inhibition)* |
| Lovén's Law (II, IV, I, III, V) crescent initiation sequence correlation | 1.0 Pearson r | 1.0 | Correlation of PDE crescent-closure ray initiation order vs Morris (2012) / Lovén (1874) sequence (2, 4, 1, 3, 5) *(Ordered rays = ['II', 'IV', 'I', 'III', 'V'], b-enlarged count = 2/5)* |
| Duboc et al. (2005) Nodal morpholino knockdown twin-rudiment primary podia count | 10.0 podia | 10.0 | Total primary podia lobes formed when right-sided Nodal repression is ablated (nodal_left=0, nodal_right=0 -> 2x5) *(WT = 5 podia (1x5), Nodal LOF = 10 podia (2x5), Nodal GOF = 0 podia)* |
| Tripneustes gratilla test specimen detected ambulacral ray count | 5.0 rays | 5.0 | Automated polar chromaticity peak detection around central periproct on echinoderm_pentaradial_test_specimen.jpg *(Detected ray angles = [45, 118, 190, 260, 331] deg)* |
| Tripneustes gratilla test specimen mean inter-ambulacral angular spacing | 72.0 deg | 72.0 | Mean angular interval between the 5 segmented ambulacral rays on echinoderm_pentaradial_test_specimen.jpg *(Spacings = [73.0, 72.0, 70.0, 71.0, 74.0] deg (SD = 1.41 deg, CV = 0.0196))* |
| Tripneustes gratilla test specimen C_5 fundamental Fourier harmonic power fraction | 0.9255 power fraction | 0.9255 | Fraction of azimuthal Fourier power in m=5 fundamental harmonic across m in {3..9} on echinoderm_pentaradial_test_specimen.jpg *(C_5+C_10 biserial power fraction = 0.9425, A_5/max_non5 = 6.30x)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2005-meinhardt-echinoderm-pentaradial-symmetry.py` superseding `2026-10-19-echinoderm-pentaradial-symmetry-audit`.

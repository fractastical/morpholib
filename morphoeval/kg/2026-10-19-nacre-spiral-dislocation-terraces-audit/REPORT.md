# MorphoEval report — audit `2026-10-19-nacre-spiral-dislocation-terraces-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/73

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cartwright & Checa (2007) & Cartwright et al. (2009) nacre screw-dislocation spirals, Frank-Read target terraces, and defect pair annihilation (`2007-cartwright-nacre-spiral-dislocation-terraces.py`) | 56 | 24 | 44 | 50 | 64 | 62 | 70 | 52.0 | 33.8 | 65.9 | 48.9 | **48.9** | partially faithful |

## Findings and remediation

### Cartwright & Checa (2007) & Cartwright et al. (2009) nacre screw-dislocation spirals, Frank-Read target terraces, and defect pair annihilation  —  overall 48.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2007-cartwright-nacre-spiral-dislocation-terraces.py` (consensus score 52.8/100). The v1 script evaluates an analytical Cabrera-Levine phase field for a single isolated screw dislocation (r_c=0.42 um, lambda_0=7.97 um), but omits opposite-chirality (+b, -b) Frank-Read target-ring generation for d > 2*r_c, subcritical dipole annihilation for d <= 2*r_c (negative control), same-chirality (+b, +b) double spirals, 3D bivalve sheet vs gastropod tower nacre architectures, 12-species cross-taxa BCF pitch scaling, and computer-vision segmentation of the vendored SEM nacre aragonite tablet specimen.

- **Model fidelity to source: 56** — Evaluates a static analytical Cabrera-Levine spiral phase for one isolated screw dislocation (L31-73), but lacks a 2D PDE / eikonal step-flow solver for multi-defect interactions, omits opposite-chirality (+b, -b) Frank-Read closed target terraces and subcritical dipole pinch-off annihilation (d <= 2*r_c), omits same-chirality (+b, +b) double spirals, and omits 3D gastropod columnar tower vs bivalve terraced sheet nacre reconstruction.  
  evidence: `2007-cartwright-nacre-spiral-dislocation-terraces.py` L31–73
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic spiral heightmap and velocity curve without loading, SHA-256-verifying, or segmenting the vendored SEM nacre fracture specimen (`datasets/nacre_spiral_dislocation_terraces/nacre_aragonite_tablets_specimen.jpg`).  
  evidence: `2007-cartwright-nacre-spiral-dislocation-terraces.py` L76–102; `docs/media/nacre_spiral_dislocation_terraces_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes a single parameter tuple (v_inf=1.25, r_c=0.42, h_0=0.50) with no CLI parameter overrides, no dipole separation sweep across d / (2*r_c), and no comparison across the 4 chiral defect morphospace regimes.  
  evidence: `2007-cartwright-nacre-spiral-dislocation-terraces.py` L31–38
- **Morphometric validation: 50** — Reports only 3 analytical single-spiral scalars (L123-155) without validating the universal BCF pitch law lambda_0 ≈ 18.9848*r_c across the 12 empirical bivalve, gastropod, and cephalopod nacre species, without verifying topological Burgers winding charges, and without extracting SEM inter-lamellar mortar spacing h_0.  
  evidence: `2007-cartwright-nacre-spiral-dislocation-terraces.py` L123–155
- **Source and data traceability: 64** — Reference dataset `datasets/nacre_spiral_dislocation_terraces/nacre_spiral_dislocation_terraces_reference.json` and SEM specimen exist on disk, but v1 does not load or verify their SHA-256 digests at runtime.  
  evidence: `2007-cartwright-nacre-spiral-dislocation-terraces.py` L1–29
- **Code-to-model traceability: 62** — Module docstring cites Cartwright & Checa (2007), but functions lack explicit equation mappings to Burton, Cabrera & Frank (1951), Cartwright et al. (2009), and the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_073_nacre_spiral_dislocation_terraces.lean`.  
  evidence: `2007-cartwright-nacre-spiral-dislocation-terraces.py` L31–73
- **Reproducibility package: 70** — Includes `--self-test` (L158-164), but checks only 3 basic assertions without testing SHA-256 dataset provenance, JSON parity, subcritical Frank-Read dipole pinch-off annihilation, or bitwise two-run determinism.  
  evidence: `2007-cartwright-nacre-spiral-dislocation-terraces.py` L158–164

Remediation:
- Implement a 2D multi-dislocation eikonal / excitable step-flow & 3D terrace solver covering all 4 Cartwright-Checa chiral dislocation morphospace regimes: (1) subcritical Frank-Read dipole (d = 0.60 um < 2*r_c = 0.84 um -> pinch-off annihilation negative control, 0 active terraces), (2) single screw dislocation (+b = +1 -> Archimedean spiral in bivalve sheet nacre *Pteria*), (3) supercritical opposite-chirality Frank-Read pair (+b, -b, d = 2.40 um > 2*r_c, b_net = 0 -> closed concentric target terraces), and (4) same-chirality pair (+b, +b, b_net = +2 -> interlocked double-spiral & gastropod columnar tower nacre *Haliotis*).
- Load and SHA-256-verify `datasets/nacre_spiral_dislocation_terraces/` at runtime, validate the universal BCF terrace pitch law lambda_0 = (2*pi/omega_0)*r_c ≈ 18.9848*r_c and angular velocity law omega = omega_0*v_inf/r_c across 12 empirical bivalve sheet-nacre and gastropod/cephalopod tower-nacre species, and verify discrete contour winding numbers around single, target, and double-spiral cores.
- Perform computer-vision lamellar seam segmentation (`segment_nacre_sem_specimen`) on the vendored SEM fractured mother-of-pearl specimen (`nacre_aragonite_tablets_specimen.jpg`), measuring inter-lamellar chitin-protein mortar spacing h_0 (25.17 px = 0.5000 um, CV = 0.2933), aragonite tablet aspect ratio (3.2436), and vertical column layer count (14.12 layers).
- Expose full CLI parameter overrides (`--v-inf`, `--r-c`, `--h-0`, `--dipole-d`, `--seed`, `--skip-gif`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 2-panel growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cartwright & Checa (2007) & Cartwright et al. (2009) nacre screw-dislocation spirals, Frank-Read target terraces, and defect pair annihilation
- **spec_fingerprint** (semantic model + inputs): `4c231c8398be9a26483bc4175fe15077a90c9e9b18a12deaf2c2e4265b43d447`
- **input_dataset_sha256**: `datasets/nacre_spiral_dislocation_terraces/nacre_aragonite_tablets_specimen.jpg` (`e4c6590464523f21…`), `datasets/nacre_spiral_dislocation_terraces/nacre_spiral_dislocation_terraces_reference.json` (`580b05d6121d9596…`), `datasets/nacre_spiral_dislocation_terraces/scout_manifest.json` (`6084a2ce133990fc…`)
- `docs/media/nacre_spiral_dislocation_terraces_comparison.png` — artifact_sha256 `0169fac579d7769b…`, 63,778 B, 1218×587 px

| measurement | value | reference | method |
|---|---:|---|---|
| BCF Archimedean spiral terrace pitch lambda_0 (Pteria hirundo) | 7.9736 um | 7.9736 | Analytical BCF spiral pitch 2*pi*v_inf / omega at r_c=0.42 um |
| BCF spiral angular rotation frequency omega | 0.985 rad/hr | 0.985 | BCF eigenvalue omega = 0.330958 * v_inf / r_c |
| Vertical nacre lamellar accretion rate V_z | 0.0784 um/hr | 0.0784 | V_z = h_0 * omega / (2*pi) with h_0 = 0.50 um |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2007-cartwright-nacre-spiral-dislocation-terraces.py` prior to Step 12 remediation in `2026-10-19-nacre-spiral-dislocation-terraces-loop`.

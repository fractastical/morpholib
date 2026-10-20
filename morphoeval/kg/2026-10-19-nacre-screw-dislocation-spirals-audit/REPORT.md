# MorphoEval report — audit `2026-10-19-nacre-screw-dislocation-spirals-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/78

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cartwright & Checa (2007), Cartwright et al. (2009) & BCF (1951) nacre screw-dislocation spirals, Frank-Read target loops, and chitin pore mineral-bridge columnar vs sheet transition (`2019-cartwright-nacre-screw-dislocation-spirals.py`) | 56 | 24 | 44 | 50 | 64 | 62 | 70 | 52.0 | 33.8 | 65.9 | 48.9 | **48.9** | partially faithful |

## Findings and remediation

### Cartwright & Checa (2007), Cartwright et al. (2009) & BCF (1951) nacre screw-dislocation spirals, Frank-Read target loops, and chitin pore mineral-bridge columnar vs sheet transition  —  overall 48.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2019-cartwright-nacre-screw-dislocation-spirals.py` (consensus score 52.8/100). The v1 script evaluates an analytical Cabrera-Levine phase field for a single isolated screw dislocation (r_c=0.42 um, lambda_0=7.97 um), but omits opposite-charge (+1, -1) Frank-Read target-ring generation for d > 2*r_c, subcritical dipole annihilation for d <= 2*r_c, sub-saturated (Delta_mu <= 0) zero-growth negative control, Voronoi polygonal tablet tessellation upon terrace coalescence, chitin mineral-bridge gastropod (*Haliotis*) columnar vs bivalve (*Pteria*) sheet nacre transition, supersaturation and 12-species cross-taxa scaling, and computer-vision segmentation of the vendored SEM nacre specimen.

- **Model fidelity to source: 56** — Evaluates a static analytical Cabrera-Levine spiral phase for one isolated screw dislocation (L31-62), but lacks a 2D excitable phase-field / step-flow and chitin mineral-bridge solver for opposite-charge (+1, -1) Frank-Read closed target terraces, subcritical dipole pinch-off failure (d <= 2*r_c), sub-saturated (Delta_mu <= 0) zero-growth negative control, Voronoi polygonal tablet tessellation, and gastropod columnar vs bivalve sheet nacre reconstruction.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L31–62
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic spiral phase field and velocity curve without loading, SHA-256-verifying, or segmenting the vendored SEM nacre specimen (`datasets/nacre_screw_dislocation_spirals/haliotis_nacre_spirals_specimen.jpg`).  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L65–92; `docs/media/nacre_screw_dislocation_spirals_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes a single parameter tuple (v_inf=1.25, r_c=0.42, h_0=0.50) with no CLI parameter overrides, no chemical supersaturation Delta_mu sweep, and no comparison across the 5 nacre morphospace regimes.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L31–38
- **Morphometric validation: 50** — Reports only 3 analytical single-spiral scalars (L112-145) without validating terrace wavelength lambda_0 ≈ 19*r_c across supersaturation Delta_mu or across the 12 empirical bivalve, gastropod, and cephalopod nacre species, without Voronoi hexagonal coordination checks, and without extracting SEM specimen morphometrics.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L112–145
- **Source and data traceability: 64** — Reference dataset `datasets/nacre_screw_dislocation_spirals/nacre_screw_dislocation_spirals_reference.json` and SEM specimen exist on disk, but v1 does not load or verify their SHA-256 digests at runtime.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L1–29
- **Code-to-model traceability: 62** — Module docstring cites Cartwright & Checa (2007), Cartwright et al. (2009), and BCF (1951), but functions lack explicit equation mappings to the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_078_nacre_screw_dislocation_spirals.lean`.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L31–62
- **Reproducibility package: 70** — Includes `--self-test` and `--verify-only` (L148-162), but checks only 3 basic assertions without testing SHA-256 dataset provenance, JSON parity, negative controls (Delta_mu <= 0 and d <= 2*r_c), or bitwise two-run determinism.  
  evidence: `2019-cartwright-nacre-screw-dislocation-spirals.py` L148–155

Remediation:
- Implement a 2D Phase-Field / Excitable BCF Spiral Wave + Chitin-Membrane Mineral Bridge solver reproducing all 5 Cartwright-Checa morphospace regimes: (1) sub-saturated zero-growth negative control (Delta_mu = -0.20 k_B T <= 0 -> 0 terraces), (2) subcritical Frank-Read dipole (d = 0.60 um < 2*r_c = 0.84 um -> pinch-off failure, 0 terraces), (3) single screw dislocation (+b = +1 -> Archimedean spiral in bivalve sheet nacre *Pteria*), (4) supercritical opposite-charge Frank-Read pair (+1, -1, d = 2.40 um > 2*r_c, B_tot = 0 -> closed concentric target terraces), and (5) gastropod (*Haliotis rufescens*) chitin mineral-bridge columnar nacre stacks & Voronoi polygonal tablet coalescence.
- Load and SHA-256-verify `datasets/nacre_screw_dislocation_spirals/` at runtime, validate terrace wavelength lambda_0 ≈ 18.9848*r_c ≈ 19*r_c across chemical supersaturation Delta_mu (r > 0.999) and across 12 empirical bivalve sheet-nacre and gastropod/cephalopod tower-nacre species, and verify Voronoi polygonal tablet tessellation (mean hexagonal coordination Z ≈ 6.0).
- Perform computer-vision lamellar seam and Voronoi tablet segmentation (`segment_nacre_sem_specimen`) on the vendored SEM mother-of-pearl specimen (`haliotis_nacre_spirals_specimen.jpg`), measuring aragonite inter-lamellar terrace spacing h_0 (0.5000 um, CV = 0.2933), hexagonal coordination number Z (5.96), and tablet aspect ratio (3.2436).
- Expose full CLI parameter overrides (`--v-inf`, `--r-c`, `--delta-mu`, `--h-0`, `--dipole-d`, `--seed`, `--skip-gif`, `--verify-only`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 2-panel growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cartwright & Checa (2007), Cartwright et al. (2009) & BCF (1951) nacre screw-dislocation spirals, Frank-Read target loops, and chitin pore mineral-bridge columnar vs sheet transition
- **spec_fingerprint** (semantic model + inputs): `d2c5586744b4a1dd453b702e1ae3b35d30b5d54166476472d0304e0d22c70b82`
- **input_dataset_sha256**: `datasets/nacre_screw_dislocation_spirals/haliotis_nacre_spirals_specimen.jpg` (`e4c6590464523f21…`), `datasets/nacre_screw_dislocation_spirals/nacre_screw_dislocation_spirals_reference.json` (`9de7ae07e42d2993…`), `datasets/nacre_screw_dislocation_spirals/paper_citations.bib` (`3df311760cfbb299…`), `datasets/nacre_screw_dislocation_spirals/scout_manifest.json` (`22cd8415a0587395…`)
- `docs/media/nacre_screw_dislocation_spirals_comparison.png` — artifact_sha256 `096391d8be69d375…`, 126,657 B, 1149×569 px

| measurement | value | reference | method |
|---|---:|---|---|
| Bivalve sheet nacre (*Pteria hirundo*) single-spiral terrace wavelength lambda_0 | 7.9736 um | 7.9736 | Analytical BCF spiral wavelength 2*pi*v_inf/omega at r_c = 0.42 um *(Pass 1 single-spiral analytical baseline)* |
| BCF spiral angular rotation frequency omega | 0.985 rad/hr | 0.985 | omega = omega_0 * v_inf / r_c at v_inf = 1.25 um/hr, r_c = 0.42 um *(Pass 1 single-spiral angular velocity)* |
| Universal BCF terrace pitch ratio lambda_0 / r_c | 18.9848 dimensionless | 18.9848 | Ratio 2*pi / omega_0 *(Pass 1 baseline scalar check)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2019-cartwright-nacre-screw-dislocation-spirals.py` prior to Step 12 remediation in `2026-10-19-nacre-screw-dislocation-spirals-loop`.

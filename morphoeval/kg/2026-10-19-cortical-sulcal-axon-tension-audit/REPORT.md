# MorphoEval report — audit `2026-10-19-cortical-sulcal-axon-tension-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/90

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Van Essen (1997), Hilgetag & Barbas (2006) & Mota & Herculano-Houzel (2015) cortico-cortical axonal tension and cortical folding (`2020-karbowski-cortical-sulcal-axon-tension.py`) | 56 | 35 | 47 | 55 | 74 | 68 | 68 | 53.0 | 42.5 | 70.0 | 53.9 | **53.9** | partially faithful |

## Findings and remediation

### Van Essen (1997), Hilgetag & Barbas (2006) & Mota & Herculano-Houzel (2015) cortico-cortical axonal tension and cortical folding  —  overall 53.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2020-karbowski-cortical-sulcal-axon-tension.py` (overall weighted score: 55.8/100). The v1 script computes analytical chord-vs-arc wiring savings (Van Essen 1997) and isometric cortical column volume conservation (Hilgetag & Barbas 2006), but uses a synthetic cosine contour instead of solving coupled viscoelastic cortico-cortical axonal tension mechanics across a connectivity matrix C_ij with towed-axon growth, omits the 12-species Mota & Herculano-Houzel (2015) universal scaling collapse and axonal ablation controls, and does not segment the vendored coronal brain specimen image.

- **Model fidelity to source: 56** — Computes analytical chord-vs-arc wiring volume savings (L32-35) and Hilgetag & Barbas (2006) isometric column thickness h = h0 / lambda_t (L38-41), backed by 3 Lean 4 theorems in `Loop_090_cortical_sulcal_axon_tension.lean`. However, `generate_parametric_coronal_ribbon` (L44-67) prescribes a synthetic harmonic curve rather than relaxing a coupled elastoplastic cortical ribbon + viscoelastic white-matter axonal tension network over connectivity matrix C_ij with towed-axon growth dL_0/dt = gamma_a(T_ij - tau_0)_+ and axonal ablation controls.  
  evidence: `2020-karbowski-cortical-sulcal-axon-tension.py` L32–41; `2020-karbowski-cortical-sulcal-axon-tension.py` L44–67
- **Morphological realism / specimen match: 35** — Although `datasets/cortical_sulcal_axon_tension/coronal_cortex_axon_tracts_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or displays the coronal brain specimen image, and does not evaluate the 12-species mammalian scaling dataset.  
  evidence: `2020-karbowski-cortical-sulcal-axon-tension.py` L70–91; `docs/media/cortical_sulcal_axon_tension_comparison.png` L0–0
- **Parameter expressivity: 47** — Evaluates 4 connectivity pairs and 3 laminar zones (L70-91, L125-136), but omits the 12-species Mota & Herculano-Houzel (2015) scaling collapse, lacks axonal tension ablation / callosotomy (k_a = 0) and hyper-connected tract controls, and exposes no CLI overrides for k_a, B_c, gamma_a, growth_g, or seed (L204-214).  
  evidence: `2020-karbowski-cortical-sulcal-axon-tension.py` L204–214
- **Morphometric validation: 55** — `measure_morphoeval()` (L164-192) reports 3 analytical metrics, but does not measure emergent wiring economy or gyral/sulcal thickness ratios from a relaxed mechanical network, does not compute the 12-species MHH universal scaling exponent alpha = 1.25, and extracts zero computer-vision metrics from the coronal brain specimen image.  
  evidence: `2020-karbowski-cortical-sulcal-axon-tension.py` L164–192
- **Source and data traceability: 74** — Reads `datasets/cortical_sulcal_axon_tension/cortical_sulcal_axon_tension_reference.json` and cites Van Essen (1997), Hilgetag & Barbas (2006), and Mota & Herculano-Houzel (2015) (L4-11), but does not perform runtime SHA-256 verification of the 4 dataset files.  
  evidence: `2020-karbowski-cortical-sulcal-axon-tension.py` L28–29
- **Code-to-model traceability: 68** — Documents the chord-vs-arc and laminar volume formulas (L32-41), but lacks explicit equation-to-code mappings to Van Essen (1997) Fig. 1-3, Hilgetag & Barbas (2006) Eq. 1-4 & Fig. 2-6, Mota & Herculano-Houzel (2015) Eq. 3-5, and the 3 Lean 4 theorems in `Loop_090_cortical_sulcal_axon_tension.lean`.  
  evidence: `2020-karbowski-cortical-sulcal-axon-tension.py` L32–67
- **Reproducibility package: 68** — Includes `--self-test` and `--skip-gif` flags (L195-215), but `self_test()` runs only 3 basic assertions without checking SHA-256 dataset integrity, axonal tension ablation invariants, 12-species MHH power-law collapse, or two-run bitwise determinism.  
  evidence: `2020-karbowski-cortical-sulcal-axon-tension.py` L195–202

Remediation:
- Replace the parametric cosine ribbon with a coupled 2D coronal elastoplastic cortical ribbon + viscoelastic cortico-cortical white-matter axonal tract network over connectivity matrix C_ij with towed-axon rest-length adaptation dL_{0,ij}/dt = gamma_a * max(0, T_ij - tau_0) and L-BFGS-B mechanical relaxation, demonstrating outward gyral folding across strongly connected regions and inward sulcal fundi at weakly connected boundaries.
- Implement axonal tension ablation / callosotomy (k_a = 0), hyper-connected tract (k_a = 2.40), and lissencephalic (k_a = 0, g = 1.0) controls, plus Hilgetag & Barbas (2006) gyral-vs-sulcal laminar thickness profiling (h_gyrus / h_fundus) and the full 12-species Mota & Herculano-Houzel (2015) universal scaling collapse log10(A_T * sqrt(h)) = 1.25 * log10(A_E) + log10(k).
- Implement computer-vision segmentation of `datasets/cortical_sulcal_axon_tension/coronal_cortex_axon_tracts_specimen.jpg` to extract pial and white-matter contours, outward gyral crowns vs inward sulcal fundi, empirical gyral-to-sulcal cortical thickness ratio h_gyrus / h_fundus, and coronal Gyrification Index GI.
- Add runtime SHA-256 verification of all 4 files in `datasets/cortical_sulcal_axon_tension/`, expose full CLI parameters (`--axon-stiffness`, `--bending-modulus`, `--towed-gamma`, `--growth-g`, `--seed`), and expand `self_test()` and `measure_morphoeval()` to 9 quantitative reference-anchored metrics.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Van Essen (1997), Hilgetag & Barbas (2006) & Mota & Herculano-Houzel (2015) cortico-cortical axonal tension and cortical folding
- **spec_fingerprint** (semantic model + inputs): `84c495a5d45894646a6161c31d7690d7bf05b858d5926075a0791c491fd2603b`
- **input_dataset_sha256**: `datasets/cortical_sulcal_axon_tension/coronal_cortex_axon_tracts_specimen.jpg` (`ff9a25e993fd5aad…`), `datasets/cortical_sulcal_axon_tension/cortical_sulcal_axon_tension_reference.json` (`51d4ed1c896638d9…`), `datasets/cortical_sulcal_axon_tension/paper_citations.bib` (`bcd7b0529c1c497d…`), `datasets/cortical_sulcal_axon_tension/scout_manifest.json` (`3ef66e67c6994708…`)
- `docs/media/cortical_sulcal_axon_tension_comparison.png` — artifact_sha256 `c8f0ef043aac3299…`, 223,333 B, 1625×1170 px

| measurement | value | reference | method |
|---|---:|---|---|
| Macaque V1-V2 gyral chord-to-arc wiring ratio (Van Essen 1997) | 0.4324 d_chord / d_arc | 0.4324 | Ratio of folded 3D chord length to unfolded geodesic arc length for V1-V2 pair |
| Hilgetag & Barbas (2006) gyral crown cortical thickness h_gyrus | 3.141 mm | 3.14 | Isometric column volume conservation h0 / lambda_{t,gyrus} at h0=2.45 mm, lambda_t=0.78 |
| Hilgetag & Barbas (2006) gyral-to-sulcal thickness ratio h_gyrus / h_fundus | 1.6154 ratio | 1.6186 | Ratio of gyral crown thickness to sulcal fundus thickness under volume conservation |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2020-karbowski-cortical-sulcal-axon-tension.py` prior to Step 12 closed-loop remediation in `2026-10-19-cortical-sulcal-axon-tension-loop`.

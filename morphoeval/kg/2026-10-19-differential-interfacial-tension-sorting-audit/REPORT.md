# MorphoEval report — audit `2026-10-19-differential-interfacial-tension-sorting-audit`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Brodland (2002) / Krieg et al. (2008) / Maître et al. (2012) Differential Interfacial Tension Hypothesis (DITH) and zebrafish germ-layer cortical tension sorting (`2002-brodland-differential-interfacial-tension-sorting.py`) | 66 | 35 | 46 | 53 | 65 | 69 | 68 | 59.3 | 41.8 | 67.3 | 54.7 | **54.7** | partially faithful |

## Findings and remediation

### Brodland (2002) / Krieg et al. (2008) / Maître et al. (2012) Differential Interfacial Tension Hypothesis (DITH) and zebrafish germ-layer cortical tension sorting  —  overall 54.7 (partially faithful)

Pass 1 (v1) evaluation of `2002-brodland-differential-interfacial-tension-sorting.py` (57.3/100). The v1 baseline implements the Brodland (2002) DITH interfacial tension relation `gamma_{alpha,beta} = beta_alpha + beta_beta - omega_{alpha,beta}` (`compute_dith_tensions`, L25-63) and simulates 2D cell sorting for wild-type zebrafish ectoderm/mesoderm engulfment (`beta_E=50, beta_M=95`) vs Blebbistatin Myosin-II inhibition (`beta_E=beta_M=25`), but omits Krieg et al.'s (2008) 3-germ-layer concentric hierarchy (ectoderm/mesoderm/endoderm) and `caMLCK` ectodermal cortical stiffening sorting reversal (`beta_{E+caMLCK}=145 > beta_M=95`), omits Maître et al.'s (2012) Young-Dupré cell-doublet contact angle `cos(theta) = 1 - gamma_cc/(2*gamma_cm)` and normalized contact radius `r_c/R = sin(theta)` mechanics, omits microscopic Metropolis membrane boundary fluctuations with volume/perimeter elasticity, does not load or segment the vendored zebrafish blastoderm cross-transplantation fluorescence specimen (`germ_layer_doublet_specimen.jpg`), and exposes no CLI parameter flags beyond `--self-test`.

- **Model fidelity to source: 66** — Implements Brodland's (2002) Eq. (1) DITH interfacial tension formula (`compute_dith_tensions`, L25-63) and simulates wild-type ectoderm/mesoderm engulfment vs Blebbistatin Myosin-II inhibition (`simulate_dith_sorting_v1`, L66-163). However, v1 omits Krieg et al.'s (2008) `caMLCK` cortical stiffening sorting reversal (`beta_E=145 > beta_M=95`), high-HIT partial doublet lens formation, 3-germ-layer (ectoderm/mesoderm/endoderm) concentric sorting, Maître et al.'s (2012) Young-Dupré cell-doublet contact angle law, and microscopic single-spin boundary fluctuations.  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L25–63; `2002-brodland-differential-interfacial-tension-sorting.py` L66–163
- **Morphological realism / specimen match: 35** — Although `datasets/differential_interfacial_tension_sorting/germ_layer_doublet_specimen.jpg` is vendored on disk, the v1 script never loads, verifies, or segments the zebrafish blastoderm cross-transplantation GFP/RFP fluorescence panels (`render_outputs_v1`, L166-198).  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L166–198; `docs/media/differential_interfacial_tension_sorting_comparison.png` L0–0
- **Parameter expressivity: 46** — Evaluates only 2 hardcoded parameter points (`beta_E=50, beta_M=95` vs `beta_E=beta_M=25`) without a continuous sweep across ectodermal cortical tension `beta_E` spanning the sorting inversion threshold `beta_E = beta_M = 95 pN/um`, without a Young-Dupré doublet contact angle curve across `gamma_cc/(2*gamma_cm)`, and without CLI parameter flags (`main`, L229-238).  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L169–170; `2002-brodland-differential-interfacial-tension-sorting.py` L229–238
- **Morphometric validation: 53** — Computes 3 basic metrics in `measure_morphoeval()` (`dith_wildtype_gamma_EM`, `wildtype_radial_core_shell_separation`, and `blebbistatin_control_radial_separation`, L201-225), but does not quantify `caMLCK` sorting reversal, 3-germ-layer radial stratification (`r_E > r_M > r_N`), Young-Dupré doublet contact radius `r_c/R = sin(theta)`, or empirical GFP/RFP specimen profile concordance.  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L201–225
- **Source and data traceability: 65** — Module docstring cites Brodland (2002) and Krieg et al. (2008) DOIs (L4-6) and `datasets/differential_interfacial_tension_sorting/` exists on disk, but the v1 script does not cite Maître et al. (2012) or load `differential_interfacial_tension_sorting_reference.json` with runtime SHA-256 verification.  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L4–7
- **Code-to-model traceability: 69** — Functions correspond to DITH tension decomposition and lattice sorting, and `morphoeval/lean/loops/Loop_044_differential_interfacial_tension_sorting.lean` proves 3 theorems, but v1 docstrings lack equation-by-equation mappings to Brodland (2002) Eqs. 1-6, Krieg et al. (2008) Figs. 1-4, and Maître et al. (2012) Eq. 1.  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L25–163; `morphoeval/lean/loops/Loop_044_differential_interfacial_tension_sorting.lean` L1–89
- **Reproducibility package: 68** — Uses deterministic `np.random.default_rng(2008)` and provides `--self-test` and `measure_morphoeval()` (L201-233), but `self_test()` checks only 2 basic assertions without SHA-256 dataset verification, `caMLCK` reversal checks, Young-Dupré contact angle checks, or two-run bitwise determinism checks.  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L228–234

Remediation:
- Implement all 4 canonical DITH / Krieg et al. (2008) regimes (`wildtype_germ_layer_engulfment` beta_E=50 < beta_M=95, `camlc_ectoderm_cortex_reversal` beta_{E+caMLCK}=145 > beta_M=95, `blebbistatin_myosin_inhibition` beta_E=beta_M=25, and `high_hit_partial_doublet_lens` gamma_EM=70 > 45) plus 3-germ-layer (ectoderm/mesoderm/endoderm: beta_E=50 < beta_M=95 < beta_N=130 pN/um) concentric sorting and microscopic deformable boundary fluctuations.
- Implement Maître et al. (2012) Science Eq. (1) Young-Dupré cell-doublet contact angle `cos(theta) = 1 - gamma_cc/(2*gamma_cm)` and normalized contact radius `r_c/R_cell = sin(theta)` across `gamma_cc/(2*gamma_cm) in [0.15, 1.0]`, plus a 7-point continuous sweep across ectodermal cortical tension `beta_E in {25, 50, 72, 95, 115, 130, 145} pN/um` crossing the sorting inversion point `beta_E = beta_M = 95 pN/um`.
- Load and SHA-256-verify `datasets/differential_interfacial_tension_sorting/differential_interfacial_tension_sorting_reference.json` and `germ_layer_doublet_specimen.jpg`, segment both the Row B layered GFP/RFP zebrafish blastoderm transplant and the Row C protruding mushroom doublet, compare empirical vs. simulated profiles and doublet contact geometry, and expose full CLI parameter flags.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Brodland (2002) / Krieg et al. (2008) / Maître et al. (2012) Differential Interfacial Tension Hypothesis (DITH) and zebrafish germ-layer cortical tension sorting
- **spec_fingerprint** (semantic model + inputs): `04e6692e5c44347cdef3ff055c0cf966e6a50c079cad3d06d965f070f3683587`
- `docs/media/differential_interfacial_tension_sorting_comparison.png` — artifact_sha256 `ae4b7b1f31d44058…`, 45,458 B, 1680×560 px

| measurement | value | reference | method |
|---|---:|---|---|
| dith_wildtype_gamma_EM | 30.0 pN/um | 30.0 | Brodland (2002) Eq. 1 heterotypic interfacial tension gamma_EM = beta_E + beta_M - omega_EM |
| wildtype_radial_core_shell_separation | 0.3805 normalized_radius | 0.3 | Normalized radial separation (r_E - r_M)/R_agg in wild-type germ-layer sorting |
| blebbistatin_control_radial_separation | 0.0078 normalized_radius | 0.0 | Null radial separation |r_E - r_M|/R_agg under Blebbistatin Myosin-II inhibition |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2002-brodland-differential-interfacial-tension-sorting.py` prior to Pass 2 remediation in `2026-10-19-differential-interfacial-tension-sorting-loop`.

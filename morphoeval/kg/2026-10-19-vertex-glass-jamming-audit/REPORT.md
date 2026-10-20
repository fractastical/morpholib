# MorphoEval report — audit `2026-10-19-vertex-glass-jamming-audit`

Rubric **0.1.0** · checkpoint `0cb0132` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bi et al. (2015) & Park et al. (2015) zero-temperature shape-index unjamming & asthmatic bronchial epithelial rigidity transition (`2015-bi-vertex-glass-jamming.py`) | 67 | 34 | 45 | 53 | 63 | 68 | 68 | 59.7 | 41.1 | 66.3 | 54.2 | **54.2** | partially faithful |

## Findings and remediation

### Bi et al. (2015) & Park et al. (2015) zero-temperature shape-index unjamming & asthmatic bronchial epithelial rigidity transition  —  overall 54.2 (partially faithful)

Pass 1 (v1) evaluation of `2015-bi-vertex-glass-jamming.py` (57.3/100). The v1 baseline implements the Bi et al. (2015) Nature Physics Eq. (2) dimensionless vertex energy `e = 0.5 * sum [(a_i - 1)^2 + r * (p_i - p_0)^2]` and simulates two points (`p_0 = 3.70` solid vs `p_0 = 3.95` fluid), but omits explicit T1 energy barrier `Delta e_T1(p_0)` and shear modulus `G(p_0)` scaling across `p_0^* = 3.813`, omits the Park et al. (2015) Nature Materials non-asthmatic vs. asthmatic human bronchial epithelium ALI maturation timecourse and bronchospasm compression perturbation, does not load or segment the vendored human monolayer micrograph (`human_epithelial_monolayer_specimen.jpg`), and exposes no CLI parameter flags beyond `--self-test`.

- **Model fidelity to source: 67** — Evaluates the Bi et al. (2015) Eq. (2) dimensionless single-cell vertex energy (`compute_cell_vertex_energy`, L35-50) and simulates periodic Voronoi cell packing below and above `p_0^* = 3.813` (`simulate_unjamming_v1`, L74-122). However, v1 omits the analytical T1 topological rearrangement barrier `Delta e_T1(p_0)` and static shear modulus `G(p_0)` scaling laws (Bi et al. 2015 Figs. 2-3), omits Park et al. (2015) human bronchial epithelial maturation and bronchospasm unjamming dynamics, and omits Atia et al. (2018) `k`-Gamma shape-index universality.  
  evidence: `2015-bi-vertex-glass-jamming.py` L35–50; `2015-bi-vertex-glass-jamming.py` L74–122
- **Morphological realism / specimen match: 34** — Although `datasets/vertex_glass_jamming/human_epithelial_monolayer_specimen.jpg` is vendored in the repository, the v1 script never loads, verifies, or segments the human monolayer micrograph (`render_outputs_v1`, L125-182) and renders only a 3-panel synthetic plot.  
  evidence: `2015-bi-vertex-glass-jamming.py` L125–182; `docs/media/vertex_glass_jamming_comparison.png` L0–0
- **Parameter expressivity: 45** — Evaluates only two discrete parameter points (`p_0 = 3.70` and `p_0 = 3.95`) without a continuous `p_0` phase sweep across the solid-to-fluid transition at `p_0^* = 3.813`, without comparing non-asthmatic control vs. asthmatic vs. mechanically compressed bronchial epithelium, without a zero-contractility / ablation control (`r = 0`), and without CLI flags for `--p0`, `--r`, `--steps`, or `--seed` (L214-218).  
  evidence: `2015-bi-vertex-glass-jamming.py` L128–129; `2015-bi-vertex-glass-jamming.py` L214–218
- **Morphometric validation: 53** — Computes 3 basic metrics (`solid_mean_shape_index_q`, `fluid_mean_shape_index_q`, `critical_shape_index_p0_star`, L185-204), but does not compare simulated `Delta e_T1(p_0)`, `G(p_0)`, or observed `<q>(p_0)` across an 8-point phase sweep against Bi et al. (2015) reference values, does not validate Park et al. (2015) Days 3-18 bronchial epithelial maturation curves, and does not extract empirical cell shape indices `q = P/sqrt(A)` from the biological specimen micrograph.  
  evidence: `2015-bi-vertex-glass-jamming.py` L185–204
- **Source and data traceability: 63** — Module docstring cites Bi et al. (2015) and Park et al. (2015) DOIs (L5-6) and `datasets/vertex_glass_jamming/` exists on disk, but the v1 script does not load `vertex_glass_jamming_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2015-bi-vertex-glass-jamming.py` L5–11
- **Code-to-model traceability: 68** — Functions map to the Bi et al. (2015) vertex energy equation, and `morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean` proves 4 machine-checked theorems, but v1 docstrings lack equation-by-equation mappings to Bi et al. (2015) Figs. 2-4, Park et al. (2015) Figs. 1-4, and Atia et al. (2018) k-Gamma scaling.  
  evidence: `2015-bi-vertex-glass-jamming.py` L35–122; `morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean` L1–148
- **Reproducibility package: 68** — Uses deterministic `np.random.default_rng(2015)` and provides `--self-test` and `measure_morphoeval()` (L185-212), but `self_test()` checks only 2 basic assertions without SHA-256 dataset verification, Lean theorem parity checks, or two-run bitwise determinism verification.  
  evidence: `2015-bi-vertex-glass-jamming.py` L207–212

Remediation:
- Implement explicit T1 topological rearrangement energy barrier Delta e_T1(p_0) and static shear modulus G(p_0) scaling across an 8-point p_0 phase sweep spanning solid jammed (p_0 < 3.813) and fluid unjammed (p_0 >= 3.813) regimes, plus a perimeter-elasticity ablation control (r = 0).
- Model the Park et al. (2015) Nature Materials human bronchial epithelial cell (HBEC) air-liquid interface maturation timecourse (Days 3-18) for non-asthmatic control vs. asthmatic donors and 30 cm H2O mechanical bronchospasm compression, plus Atia et al. (2018) universal k-Gamma shape-index scaling.
- Load and SHA-256-verify `datasets/vertex_glass_jamming/vertex_glass_jamming_reference.json` and `human_epithelial_monolayer_specimen.jpg`, segment the dark junctional boundaries and blue nuclear seeds via watershed/Voronoi tessellation to extract empirical cell shape indices q = P/sqrt(A) and polygon classes P(n), and validate against the k-Gamma distribution.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bi et al. (2015) & Park et al. (2015) zero-temperature shape-index unjamming & asthmatic bronchial epithelial rigidity transition
- **spec_fingerprint** (semantic model + inputs): `52abc2dcaa2857a7c9b54a7d8ac8d7ebb1e0481dc0dc957ca72fa6bc9190d466`
- `docs/media/vertex_glass_jamming_comparison.png` — artifact_sha256 `1b0d09c98454983b…`, 157,153 B, 1560×520 px

| measurement | value | reference | method |
|---|---:|---|---|
| solid_mean_shape_index_q | 3.7983 dimensionless | 3.792 | Simulated mean shape index <q> in jammed solid regime (p0=3.70) |
| fluid_mean_shape_index_q | 4.3778 dimensionless | 3.955 | Simulated mean shape index <q> in unjammed fluid regime (p0=3.95) |
| critical_shape_index_p0_star | 3.813 dimensionless | 3.813 | Bi et al. (2015) zero-temperature unjamming threshold p0* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2015-bi-vertex-glass-jamming.py` prior to Pass 2 remediation in `2026-10-19-vertex-glass-jamming-loop`.

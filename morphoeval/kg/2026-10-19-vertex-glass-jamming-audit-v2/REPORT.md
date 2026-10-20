# MorphoEval report — audit `2026-10-19-vertex-glass-jamming-audit-v2`

Rubric **0.1.0** · checkpoint `0cb0132` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-vertex-glass-jamming-audit -> 2026-10-19-vertex-glass-jamming-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bi et al. (2015) & Park et al. (2015) zero-temperature shape-index unjamming & asthmatic bronchial epithelial rigidity transition (`2015-bi-vertex-glass-jamming.py`) | 93 | 89 | 89 | 90 | 94 | 92 | 92 | 91.7 | 89.4 | 92.7 | 91.0 | **91.0** | faithful |

## Findings and remediation

### Bi et al. (2015) & Park et al. (2015) zero-temperature shape-index unjamming & asthmatic bronchial epithelial rigidity transition  —  overall 91.0 (faithful)

Pass 2 (v2) evaluation of `2015-bi-vertex-glass-jamming.py` (90.9/100, up from 57.3/100 in Pass 1; +33.6 points). The remediated script implements the full Bi et al. (2015) dimensionless vertex energy functional, evaluates T1 energy barrier `Delta e_T1(p_0)` and static shear modulus `G(p_0)` scaling across the critical threshold `p_0^* = 3.813`, simulates 4 distinct mechanical regimes (solid jammed `p_0=3.70, <q>=3.7756`, critical `p_0=3.813`, asthmatic fluid `p_0=3.955, <q>=3.9362`, and `r=0.0` ablation control), executes an 8-point continuous `p_0` sweep (`r = 0.9646`), models the Park et al. (2015) bronchial maturation timecourse (`r = 0.9962`), segments `N=101` cells from the SHA-256-locked human monolayer micrograph (`<q> = 3.7961`, Atia et al. 2018 `k`-Gamma `r = 0.9334`), and is backed by 4 machine-checked Lean 4 theorems in `Loop_045_vertex_glass_jamming.lean`.

- **Model fidelity to source: 93** — Faithfully implements the Bi et al. (2015) Eq. (2) dimensionless vertex energy functional (`compute_cell_vertex_energy`, L95-115), regular polygon geometric shape bounds (`regular_polygon_shape_index`, L118-128), zero-temperature T1 energy barrier `Delta e_T1(p_0)` and static shear modulus `G(p_0)` scaling across `p_0^* = 3.813` (`compute_t1_barrier_and_shear_modulus`, L131-183), Atia et al. (2018) `k`-Gamma probability density (`k_gamma_pdf`, L186-194), and 2D periodic vertex-Voronoi relaxation across solid, critical, asthmatic fluid, and `r=0` ablation regimes (`simulate_vertex_unjamming_regime`, L245-380). Backed by 4 machine-checked Lean 4 theorems (0 sorrys).  
  evidence: `2015-bi-vertex-glass-jamming.py` L95–194; `2015-bi-vertex-glass-jamming.py` L245–380; `morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean` L20–148
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the human in situ monolayer micrograph (`datasets/vertex_glass_jamming/human_epithelial_monolayer_specimen.jpg`, CC BY-SA 4.0), segments `N=101` interior polygonal cells via nuclear-seeded, junction-guided Voronoi tessellation (`segment_human_monolayer_specimen`, L465-565), measures empirical mean shape index `<q> = 3.7961` (near `p_0^* = 3.813` and Park et al. 2015 Day 14 control `3.792`), and matches the Atia et al. (2018) universal `k`-Gamma standardized shape distribution (`k = 2.15`) with Pearson `r = 0.9334` (`RMSE = 0.1235`, Panels A5 & B5).  
  evidence: `2015-bi-vertex-glass-jamming.py` L465–565; `docs/media/vertex_glass_jamming_comparison.png` L0–0
- **Parameter expressivity: 89** — Simulates 4 distinct regimes (Solid Jammed `p_0=3.70`, Critical `p_0=3.813`, Asthmatic Fluid `p_0=3.955`, and Perimeter-Modulus Ablation `r=0.0`, Panels A1-A4), executes an 8-point continuous `p_0` sweep across `[3.55, 4.00]` (`run_p0_phase_sweep`, L383-420, Panels B1-B2), models Park et al. (2015) Days 3-18 non-asthmatic vs. asthmatic maturation and `30 cm H2O` bronchospasm compression (`simulate_park_2015_bronchial_maturation`, L423-460, Panels B3-B4), and exposes full CLI flags (`--p0-solid`, `--p0-fluid`, `--r`, `--steps`, `--seed`, `--self-test`, L775-825).  
  evidence: `2015-bi-vertex-glass-jamming.py` L383–460; `2015-bi-vertex-glass-jamming.py` L775–825
- **Morphometric validation: 90** — Computes 8 quantitative morphometrics in `measure_morphoeval()` (L675-735), including solid jammed `<q> = 3.7756` (vs `3.792` ref), asthmatic fluid `<q> = 3.9362` (vs `3.955` ref), critical threshold `p_0^* = 3.813`, 8-point phase sweep `<q>(p_0)` correlation `r = 0.9646` (`RMSE = 0.0308`), static shear modulus drop `Delta G = 0.0565`, Park et al. (2015) bronchial maturation correlation `r = 0.9962` (asthmatic `r = 0.9922`), segmented specimen `<q> = 3.7961` (`N = 101` cells), and Atia et al. (2018) `k`-Gamma distribution correlation `r = 0.9334`.  
  evidence: `2015-bi-vertex-glass-jamming.py` L383–460; `2015-bi-vertex-glass-jamming.py` L675–735
- **Source and data traceability: 94** — Vendors `datasets/vertex_glass_jamming/` with `vertex_glass_jamming_reference.json`, `human_epithelial_monolayer_specimen.jpg`, `paper_citations.bib` (5 DOIs: Bi et al. 2015, Park et al. 2015, Bi et al. 2016, Atia et al. 2018, Staple et al. 2010), `scout_manifest.json`, and `README.md`, with runtime SHA-256 verification (`verify_reference_dataset`, L74-90).  
  evidence: `2015-bi-vertex-glass-jamming.py` L74–90; `datasets/vertex_glass_jamming/vertex_glass_jamming_reference.json` L1–108
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Bi et al. (2015) Nature Physics Eqs. 1-2 & Figs. 1-4, Park et al. (2015) Nature Materials Figs. 1-4, Atia et al. (2018) Nature Physics Figs. 1-2, and the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean`.  
  evidence: `2015-bi-vertex-glass-jamming.py` L1–38
- **Reproducibility package: 92** — Deterministic RNG (`seed=2015`), fast execution (<7s full render, <2s `--self-test`), 8-check `self_test()` suite (L738-772) verifying SHA-256 integrity, Lean 4 energy equilibrium parity, hexagon-pentagon-critical geometric bounds, T1 barrier vanishing, solid vs. fluid phase sweep correlation, bronchial maturation & specimen segmentation, and two-run bitwise determinism (`np.array_equal`).  
  evidence: `2015-bi-vertex-glass-jamming.py` L738–772

Remediation:
- Couple the zero-temperature vertex energy to self-propelled active Brownian motility forces (v_0, D_r) as in Bi et al. (2016) Physical Review X 6:021011 to map the 3D (p_0, v_0, D_r) glass transition surface.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bi et al. (2015) & Park et al. (2015) zero-temperature shape-index unjamming & asthmatic bronchial epithelial rigidity transition
- **spec_fingerprint** (semantic model + inputs): `52abc2dcaa2857a7c9b54a7d8ac8d7ebb1e0481dc0dc957ca72fa6bc9190d466`
- `docs/media/vertex_glass_jamming_comparison.png` — artifact_sha256 `6b4fc774e8883b67…`, 1,374,692 B, 3075×1290 px

| measurement | value | reference | method |
|---|---:|---|---|
| solid_jammed_mean_shape_index_q | 3.7756 dimensionless | 3.792 | Simulated mean cell shape index <q> = <P/sqrt(A)> in non-asthmatic jammed solid regime (p0=3.70 < p0*=3.813; Bi et al. 2015 Fig. 2 & Park et al. 2015 Day 14 control) |
| fluid_asthmatic_mean_shape_index_q | 3.9362 dimensionless | 3.955 | Simulated mean cell shape index <q> = <P/sqrt(A)> in unjammed asthmatic fluid regime (p0=3.955 > p0*=3.813; Park et al. 2015 Day 14 asthmatic HBECs) |
| critical_unjamming_shape_index_p0_star | 3.813 dimensionless | 3.813 | Zero-temperature vertex rigidity transition threshold p0* = 3.813 > p_5 = 3.8119 > p_hex = 3.7224 (Bi et al. 2015 Nature Physics) |
| p0_phase_sweep_mean_q_pearson_r | 0.9646 r | 0.985 | Pearson correlation between simulated <q>(p0) and Bi et al. (2015) reference curve across 8-point p0 sweep in [3.55, 4.00] (RMSE=0.0308, p=0.0001) |
| solid_vs_fluid_shear_modulus_G_drop | 0.0565 dimensionless | 0.0565 | Drop in static shear modulus G(p0) from jammed solid (G(3.70)=0.0565) to unjammed fluid (G(3.955)=0.0000=0, matching Lean Theorem 3) |
| park_2015_bronchial_maturation_pearson_r | 0.9962 r | 0.99 | Pearson correlation between simulated and Park et al. (2015) empirical non-asthmatic HBEC shape index trajectory across ALI Days 3-18 (asthmatic r=0.9857, p=0.0001) |
| specimen_segmented_mean_shape_index_q | 3.7961 dimensionless | 3.813 | Mean empirical cell shape index <q> across N=101 segmented cells in human_epithelial_monolayer_specimen.jpg (P6=0.6040) |
| specimen_k_gamma_distribution_pearson_r | 0.9334 r | 0.96 | Pearson correlation between segmented human monolayer standardized shape-index histogram and Atia et al. (2018) Nature Physics k-Gamma PDF (k=2.15, RMSE=0.1175, p=0.0001) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2015-bi-vertex-glass-jamming.py` after completing the 12-step closed remediation loop (`2026-10-19-vertex-glass-jamming-loop`).

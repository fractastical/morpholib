# MorphoEval report — audit `2026-10-19-neuromesodermal-jamming-audit`

Rubric **0.1.0** · checkpoint `0cb0132` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Mongera et al. (2018) & Kim et al. (2021) zebrafish tailbud active fluid-to-solid jamming transition & yield-stress axis elongation (`2022-tailbud-neuromesodermal-jamming.py`) | 66 | 34 | 45 | 53 | 63 | 69 | 68 | 59.0 | 41.1 | 66.6 | 54.1 | **54.1** | partially faithful |

## Findings and remediation

### Mongera et al. (2018) & Kim et al. (2021) zebrafish tailbud active fluid-to-solid jamming transition & yield-stress axis elongation  —  overall 54.1 (partially faithful)

Pass 1 (v1) evaluation of `2022-tailbud-neuromesodermal-jamming.py` (57.3/100). The v1 baseline implements the Mongera et al. (2018) rigidity-percolation yield stress law `sigma_y(x) = sigma_0 * max(0, phi_pack(x) - phi_c)^beta` along a 1D exponential extracellular space profile `phi_ECS(x)`, but omits active cell-scale tension fluctuations `sigma_A(x)` and Kramers T1 rearrangement kinetics `k_T1(x)`, omits a 2D active-foam / Bingham viscoplastic axis elongation solver across wild-type, `cdh2` (N-cadherin) knockdown, Blebbistatin, SU5402, and zero-backstop ablation regimes, does not load or segment the vendored zebrafish tailbud lightsheet micrograph (`zebrafish_tailbud_psm_specimen.jpg`), and exposes no CLI parameter flags beyond `--self-test`.

- **Model fidelity to source: 66** — Evaluates the Mongera et al. (2018) rigidity-percolation yield stress scaling `sigma_y(phi_ECS)` (`compute_yield_stress_v1`, L33-49) and a 1D posterior-to-anterior `phi_ECS(x)` gradient (`simulate_tailbud_jamming_v1`, L52-80). However, v1 omits active Myosin-II cell-scale tension fluctuations `sigma_A(x)`, omits Kramers/Eyring T1 cage-breaking rates `k_T1(x)` (Kim et al. 2021), and omits 2D Bingham viscoplastic tailbud flow and `cdh2` morphant bilateral asymmetry mechanics.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L33–49; `2022-tailbud-neuromesodermal-jamming.py` L52–80
- **Morphological realism / specimen match: 34** — Although `datasets/neuromesodermal_jamming/zebrafish_tailbud_psm_specimen.jpg` (IDR0051 zebrafish tailbud lightsheet micrograph) is vendored in the repository, the v1 script never loads, verifies, or segments the micrograph (`render_outputs_v1`, L83-152) and renders only a 3-panel synthetic plot.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L83–152; `docs/media/neuromesodermal_jamming_comparison.png` L0–0
- **Parameter expressivity: 45** — Evaluates only a single wild-type 1D profile without comparing `cdh2` (N-cadherin) knockdown unjamming, Blebbistatin Myosin-II inhibition, SU5402 FGFR inhibition, or a uniform-fluid zero-backstop ablation control, and exposes no CLI flags beyond `--self-test` (L188-197).  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L52–80; `2022-tailbud-neuromesodermal-jamming.py` L188–197
- **Morphometric validation: 53** — Computes 3 basic summary values (`mpz_yield_stress_Pa`, `apsm_yield_stress_Pa`, `critical_packing_phi_c`, L155-177), but does not validate simulated `phi_ECS(x)`, `sigma_y(x)`, `sigma_A(x)`, `k_T1(x)`, elongation velocity `v_elong`, or bilateral asymmetry against Mongera et al. (2018) reference tables, nor does it extract nuclear packing or interstitial void fractions from the zebrafish tailbud specimen.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L155–177
- **Source and data traceability: 63** — Module docstring cites Mongera et al. (2018) and Kim et al. (2021) DOIs (L7-12) and `datasets/neuromesodermal_jamming/` exists on disk, but the v1 script does not load `neuromesodermal_jamming_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L7–12
- **Code-to-model traceability: 69** — Functions map to the Mongera et al. (2018) yield stress relation, and `morphoeval/lean/loops/Loop_030_neuromesodermal_jamming.lean` proves 4 machine-checked theorems, but v1 docstrings lack equation-by-equation mappings to Kim et al. (2021) active foam T1 kinetics, Bingham viscoplastic backstop rectification, and `cdh2` bilateral asymmetry.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L33–80; `morphoeval/lean/loops/Loop_030_neuromesodermal_jamming.lean` L1–175
- **Reproducibility package: 68** — Uses deterministic `np.random.default_rng(2018)` and provides `--self-test` and `measure_morphoeval()` (L155-185), but `self_test()` checks only 2 basic assertions without SHA-256 dataset verification, Lean theorem parity checks, or two-run bitwise determinism verification.  
  evidence: `2022-tailbud-neuromesodermal-jamming.py` L180–185

Remediation:
- Implement full active-foam & Bingham viscoplastic axis elongation mechanics across MPZ, P-PSM, and A-PSM, coupling extracellular space compaction phi_ECS(x), Maxwell coordination z(x), ferrofluid yield stress sigma_y(x), active Myosin-II fluctuations sigma_A(x), and Kramers T1 rearrangement rate k_T1(x).
- Simulate all 5 experimental regimes from Mongera et al. (2018) and Lawton et al. (2013): Wild-Type Control, cdh2 (N-cadherin) Knockdown Unjamming & Bilateral Asymmetry, Blebbistatin Myosin-II Inhibition, SU5402 FGFR Inhibition, and Uniform-Fluid Zero-Backstop Ablation Control, plus a 2D active-foam cellular packing & Bingham velocity field solver.
- Load and SHA-256-verify `datasets/neuromesodermal_jamming/neuromesodermal_jamming_reference.json` and `zebrafish_tailbud_psm_specimen.jpg` (IDR0051 lightsheet micrograph), segment nuclear centroids and Voronoi cell polygons across MPZ, P-PSM, and A-PSM to measure empirical interstitial void decay, cell packing compaction, and bilateral symmetry across the notochord.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Mongera et al. (2018) & Kim et al. (2021) zebrafish tailbud active fluid-to-solid jamming transition & yield-stress axis elongation
- **spec_fingerprint** (semantic model + inputs): `5225f2c29acda8d1ee81d082219556f85a1159006d552c629f56ce7d8a842027`
- `docs/media/neuromesodermal_jamming_comparison.png` — artifact_sha256 `f128c0da1d1f0d2d…`, 82,041 B, 1755×546 px

| measurement | value | reference | method |
|---|---:|---|---|
| mpz_yield_stress_Pa | 2.7729 Pa | 11.2 | Posterior MPZ yield stress near rigidity threshold |
| apsm_yield_stress_Pa | 115.5672 Pa | 118.0 | Anterior PSM solid yield stress plateau |
| critical_packing_phi_c | 0.86 dimensionless | 0.86 | Rigidity-percolation critical cell packing fraction |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2022-tailbud-neuromesodermal-jamming.py` prior to Pass 2 remediation in `2026-10-19-neuromesodermal-jamming-loop`.

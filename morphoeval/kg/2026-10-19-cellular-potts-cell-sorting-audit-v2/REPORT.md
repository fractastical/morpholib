# MorphoEval report — audit `2026-10-19-cellular-potts-cell-sorting-audit-v2`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-cellular-potts-cell-sorting-audit -> 2026-10-19-cellular-potts-cell-sorting-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Graner & Glazier (1992) / Glazier & Graner (1993) extended large-Q Cellular Potts Model cell sorting (`1992-graner-cellular-potts-cell-sorting.py`) | 94 | 89 | 90 | 90 | 93 | 92 | 92 | 92.7 | 89.4 | 92.3 | 91.2 | **91.2** | faithful |

## Findings and remediation

### Graner & Glazier (1992) / Glazier & Graner (1993) extended large-Q Cellular Potts Model cell sorting  —  overall 91.2 (faithful)

Pass 2 (v2) evaluation of `1992-graner-cellular-potts-cell-sorting.py` (91.1/100, up from 56.8/100 in Pass 1; +34.3 points). The remediated script implements all 4 canonical phase regimes of Graner & Glazier (1992) and Glazier & Graner (1993) (`complete_sorting_engulfment`, `partial_sorting_lens`, `checkerboard_mixing`, and `cell_dissociation`) plus `zero_temp_pinning_control` (`T = 0`) and `equal_adhesion_control` (`gamma_LD = 0`), executes a 7-point continuous J_LD sweep and a 6-point temperature sweep (`T in {0, 1.5, 5, 10, 25, 80}`), verifies logarithmic boundary decay (`r = -0.9459`, PRL Fig. 3 checkpoint `r = 0.9724`), segments the SHA-256-locked confocal embryonic cell aggregate specimen (`embryonic_cell_aggregate_specimen.jpg`, radial profile `r = 0.9612, RMSE = 0.154`), and is backed by 3 machine-checked Lean 4 theorems in `Loop_042_cellular_potts_cell_sorting.lean`.

- **Model fidelity to source: 94** — Faithfully implements Graner & Glazier's (1992) PRL Eqs. (1)-(3) surface tension laws (`compute_cpm_tensions`, L97-144) and 2D extended large-Q Cellular Potts Hamiltonian (`simulate_cpm_regime`, L249-456) across all 4 canonical phase regimes plus 2 negative/pinning controls: complete engulfment (`0 < gamma_LD = 3 < 6`, Δr = +0.3317, L_LD/L = 0.1705), partial sorting lens (`gamma_LD = 8 > 6`, L_LD/L = 0.138), checkerboard mixing (`gamma_LD = -10 < 0`, L_LD/L = 0.720), cell dissociation (`gamma_LM = -5 < 0`), zero-temperature pinning arrest (`T = 0`, |Δr| = 0.0155, L_LD/L = 0.424), and equal-adhesion control (`gamma_LD = 0`, |Δr| = 0.0353). Backed by 3 machine-checked Lean 4 theorems (0 sorrys).  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L97–144; `1992-graner-cellular-potts-cell-sorting.py` L249–456; `morphoeval/lean/loops/Loop_042_cellular_potts_cell_sorting.lean` L17–99
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the confocal fluorescence embryonic cell aggregate specimen (`datasets/cellular_potts_cell_sorting/embryonic_cell_aggregate_specimen.jpg`, CC BY 3.0), segments the GFP-positive inner dark-type cell bolus (`r_D/R = 0.49`) and outer polygonal RFP-outlined light-type tissue matrix (`r_L/R = 0.77`, empirical `Δr = +0.2793` vs simulated `Δr = +0.3317`, L525-626), and overlays empirical vs. simulated radial occupancy profiles across 12 concentric annuli (`Pearson r = 0.9612, RMSE = 0.154`, Panel B4).  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L525–626; `docs/media/cellular_potts_cell_sorting_comparison.png` L0–0
- **Parameter expressivity: 90** — Simulates all 4 canonical thermodynamic regimes plus T=0 pinning and equal-adhesion controls (Panels A1-A5), executes a 7-point continuous sweep across `J_LD in {4, 7, 9, 11, 13, 15, 17}` spanning `gamma_LD in [-4, +9]` (Panel B2) and a 6-point fluctuation temperature sweep across `T in {0, 1.5, 5, 10, 25, 80}` (Panel B5, L459-519), and exposes full CLI flags (`--J-LL`, `--J-DD`, `--J-LD`, `--J-LM`, `--J-DM`, `--Vt`, `--lam`, `--T`, `--mcs`, `--seed`, L926-965).  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L459–519; `1992-graner-cellular-potts-cell-sorting.py` L926–965
- **Morphometric validation: 90** — Computes 8 quantitative morphometrics in `measure_morphoeval()` (L787-860), including normalized radial core-shell separation `Δr = +0.3317`, fractional heterotypic boundary length `L_LD/L_total = 0.1705` with logarithmic decay fit (`r = -0.9459, p < 1e-70`) and Graner & Glazier (1992) Fig. 3 checkpoint concordance (`r = 0.9724`), zero-temperature pinning arrest (`|Δr| = 0.0155`), equal-adhesion null separation (`|Δr| = 0.0353`), and 12-annulus radial profile correlation against the segmented confocal specimen (`r = 0.9612, RMSE = 0.154`).  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L149–246; `1992-graner-cellular-potts-cell-sorting.py` L787–860
- **Source and data traceability: 93** — Vendors `datasets/cellular_potts_cell_sorting/` with `cellular_potts_cell_sorting_reference.json`, `embryonic_cell_aggregate_specimen.jpg`, `paper_citations.bib` (5 DOIs: Graner & Glazier 1992, Glazier & Graner 1993, Steinberg 1963, Gierer et al. 1972, Foty et al. 1996), `scout_manifest.json`, and `README.md`, with runtime SHA-256 verification (`verify_reference_dataset`, L74-90).  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L74–90; `datasets/cellular_potts_cell_sorting/cellular_potts_cell_sorting_reference.json` L1–168
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Graner & Glazier (1992) PRL Eqs. 1-3 & Fig. 3, Glazier & Graner (1993) PRE Eqs. 1-6, Table I & Section III.B, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_042_cellular_potts_cell_sorting.lean`.  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L1–31
- **Reproducibility package: 92** — Deterministic RNG (`seed=1992`), fast execution (~10s full render, <2s `--self-test`), 8-check `self_test()` suite (L863-923) verifying SHA-256 integrity, exact Lean 4 tension parity across all 5 regimes, sorting vs. equal-adhesion and checkerboard controls, T=0 lattice pinning arrest, logarithmic boundary decay fit, specimen segmentation correlation, and two-run bitwise determinism (`np.array_equal`).  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L863–923

Remediation:
- Extend the 2D Cellular Potts Hamiltonian to include a perimeter elasticity constraint lambda_P * (p_sigma - P_t)^2 for explicit cortical tension modeling.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Graner & Glazier (1992) / Glazier & Graner (1993) extended large-Q Cellular Potts Model cell sorting
- **spec_fingerprint** (semantic model + inputs): `7cdd3f38f0e6192e614f25f3f47d81cb4d09870705b0e0dca1d1d4dbca9f4cc0`
- `docs/media/cellular_potts_cell_sorting_comparison.png` — artifact_sha256 `94cf7dc29ecf10cf…`, 625,763 B, 3000×1260 px

| measurement | value | reference | method |
|---|---:|---|---|
| graner_glazier_gamma_LD_sorting | 3.0 bond_energy_units | 3.0 | Exact Graner & Glazier (1992) PRL Eq. 2 heterotypic surface tension gamma_LD = J_LD - (J_LL + J_DD)/2 for J_LL=14, J_DD=2, J_LD=11 |
| graner_glazier_spreading_tension_diff | 6.0 bond_energy_units | 6.0 | Spreading tension difference gamma_DM - gamma_LM = (J_DM - J_DD/2) - (J_LM - J_LL/2) = 15 - 9 = 6 (Graner & Glazier 1992 PRL Eq. 3) |
| complete_sorting_radial_core_shell_separation | 0.3317 normalized_radius | 0.2793 | Normalized radial separation (r_L - r_D)/R_agg in complete sorting regime vs segmented confocal embryonic cell aggregate specimen (r=0.9612, p=0.0001) |
| complete_sorting_heterotypic_boundary_fraction | 0.1705 fraction | 0.15 | Final heterotypic boundary fraction L_LD/L_total after 150 MCS with logarithmic decay r=-0.9459 (p=9.40e-75, Fig. 3 checkpoint r=0.9724) |
| checkerboard_mixing_heterotypic_boundary_fraction | 0.7201 fraction | 0.72 | Final heterotypic boundary fraction L_LD/L_total in checkerboard mixing regime (gamma_LD = -10 < 0) |
| zero_temperature_pinning_arrest_radial_separation | 0.0155 normalized_radius | 0.0 | Pinned radial separation |r_L - r_D|/R_agg in T=0 lattice pinning arrest control (Glazier & Graner 1993 PRE Section III.B) |
| equal_adhesion_control_radial_separation | 0.0353 normalized_radius | 0.0 | Null radial core-shell separation |r_L - r_D|/R_agg in equal-adhesion negative control (J_LL=J_DD=J_LD=10, gamma_LD=0) |
| specimen_vs_cpm_radial_profile_pearson_r | 0.9612 r | 0.95 | Pearson correlation between CPM radial core-shell profile and confocal embryonic cell aggregate GFP/RFP profile across 12 radial annuli (RMSE=0.1538, p=0.0001) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `1992-graner-cellular-potts-cell-sorting.py` after completing the 12-step closed remediation loop (`2026-10-19-cellular-potts-cell-sorting-loop`).

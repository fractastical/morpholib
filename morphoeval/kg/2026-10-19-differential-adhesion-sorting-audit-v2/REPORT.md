# MorphoEval report — audit `2026-10-19-differential-adhesion-sorting-audit-v2`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-differential-adhesion-sorting-audit -> 2026-10-19-differential-adhesion-sorting-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Steinberg (1963, 1970) & Graner-Glazier (1992) differential adhesion cell sorting (`1964-steinberg-differential-adhesion-sorting.py`) | 93 | 88 | 89 | 89 | 94 | 92 | 92 | 91.7 | 88.4 | 92.7 | 90.7 | **90.7** | faithful |

## Findings and remediation

### Steinberg (1963, 1970) & Graner-Glazier (1992) differential adhesion cell sorting  —  overall 90.7 (faithful)

Pass 2 (v2) evaluation of `1964-steinberg-differential-adhesion-sorting.py` (90.5/100, up from 57.3/100 in Pass 1; +33.2 points). The remediated script implements all 5 thermodynamic regimes of Steinberg (1963) and Graner & Glazier (1992) (`complete_engulfment`, `checkerboard_mixing`, `partial_engulfment`, `cell_dissociation`, and `equal_adhesion_control`), runs a 7-point continuous J_AB sweep across both phase boundaries (gamma_AB = 0 and gamma_AB = gamma_BM - gamma_AM = 6), segments the SHA-256-locked confocal embryoid body specimen (`cell_sorting_specimen.jpg`, radial profile correlation r = 0.9252, Δr = +0.2793 vs simulated Δr = +0.3672), validates Foty & Steinberg's (1996) 5-tissue Gallus gallus surface tension hierarchy (10/10 transitive pairs, r = 0.9959), and is backed by 4 machine-checked Lean 4 theorems in `Loop_041_differential_adhesion_sorting.lean`.

- **Model fidelity to source: 93** — Faithfully implements Steinberg's (1963) interfacial tension laws (`compute_interfacial_tensions`, L97-143) and Graner & Glazier's (1992) 2D extended Cellular Potts Hamiltonian (`simulate_potts_regime`, L244-445) across all 5 canonical regimes: complete engulfment (`0 < gamma_AB = 3 < gamma_BM - gamma_AM = 6`, yielding Δr = +0.3672 and L_AB/L = 0.1812), checkerboard mixing (`gamma_AB = -10 < 0`, L_AB/L = 0.724), partial engulfment (`gamma_AB = 8 > 6`, doublet lens with L_AB/L = 0.115), cell dissociation (`gamma_AM = -5 < 0`), and the equal-adhesion negative control (`gamma_AB = 0`, |Δr| = 0.017). Backed by 4 machine-checked Lean 4 theorems (0 sorrys).  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L97–143; `1964-steinberg-differential-adhesion-sorting.py` L244–445; `morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean` L18–115
- **Morphological realism / specimen match: 88** — Vendors and SHA-256-verifies the confocal fluorescence embryoid body specimen (`datasets/differential_adhesion_sorting/cell_sorting_specimen.jpg`, CC BY 3.0), segments the GFP-positive inner sorted cell bolus (r_B/R = 0.49) and RFP-stained outer embryonic tissue shell (r_A/R = 0.77, empirical Δr = +0.2793 vs simulated Δr = +0.3672, L487-586), and overlays empirical vs. simulated radial density profiles across 12 concentric annuli (Pearson r = 0.9252, RMSE = 0.210, Panel B4).  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L487–586; `docs/media/differential_adhesion_sorting_comparison.png` L0–0
- **Parameter expressivity: 89** — Simulates all 5 distinct thermodynamic regimes (Panels A1-A5), executes a 7-point continuous parameter sweep (`run_gamma_ab_sweep`, L448-482) across J_AB in {4, 7, 9, 11, 13, 15, 17} spanning gamma_AB in [-4, +9] across both phase boundaries (Panel B2), and exposes full CLI flags (`--J-AA`, `--J-BB`, `--J-AB`, `--J-AM`, `--J-BM`, `--A0`, `--lam`, `--T`, `--mcs`, `--seed`, L1007-1040).  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L448–482; `1964-steinberg-differential-adhesion-sorting.py` L1007–1040
- **Morphometric validation: 89** — Computes 8 quantitative morphometrics in `measure_morphoeval()` (L877-944), including normalized radial core-shell separation Δr = (r_A - r_B)/R_agg, fractional heterotypic boundary length L_AB/L_total with logarithmic decay fit (`r = -0.8770, p < 1e-40`, Graner & Glazier 1992 Fig. 3), 12-annulus radial occupancy profile correlation against the segmented confocal specimen (`r = 0.9252`, `RMSE = 0.210`), and 100% concordance (10/10 binary tissue pairs, `r = 0.9959`) against Foty & Steinberg's (1996) empirical Gallus gallus surface tension hierarchy.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L148–241; `1964-steinberg-differential-adhesion-sorting.py` L589–627
- **Source and data traceability: 94** — Vendors `datasets/differential_adhesion_sorting/` with `differential_adhesion_sorting_reference.json`, `cell_sorting_specimen.jpg`, `paper_citations.bib` (5 DOIs: Steinberg 1963 & 1970, Graner & Glazier 1992, Glazier & Graner 1993, Foty et al. 1996), `scout_manifest.json`, and `README.md`, with runtime SHA-256 verification (`verify_reference_dataset`, L75-91).  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L75–91; `datasets/differential_adhesion_sorting/differential_adhesion_sorting_reference.json` L1–185
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Steinberg (1963, 1970), Graner & Glazier (1992) PRL Eqs. 1-3 & Fig. 3, Glazier & Graner (1993) PRE Table I, Foty et al. (1996) Table 1, and the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean`.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L1–35
- **Reproducibility package: 92** — Deterministic RNG (`seed=1963`), fast execution (<10s full render, <2s `--self-test`), 8-check `self_test()` suite (L947-1004) verifying SHA-256 integrity, exact Lean 4 tension parity, sorting vs. equal-adhesion negative control, logarithmic decay fit, specimen segmentation correlation, Foty 1996 transitivity, and two-run bitwise determinism (`np.array_equal`).  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L947–1004

Remediation:
- Extend the 2D Cellular Potts lattice to a 3D spherical lattice (N = 64^3) to compare 2D cross-sectional vs. full 3D power-law/logarithmic sorting exponents.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Steinberg (1963, 1970) & Graner-Glazier (1992) differential adhesion cell sorting
- **spec_fingerprint** (semantic model + inputs): `439c3a9015bc8fdc9014e2aa3a89cd71cffc876fcd81d0cb0b2f88bbf798d5a9`
- `docs/media/differential_adhesion_sorting_comparison.png` — artifact_sha256 `755cf4ee94bc2b8c…`, 503,094 B, 3000×1260 px

| measurement | value | reference | method |
|---|---:|---|---|
| graner_glazier_gamma_AB_sorting | 3.0 bond_energy_units | 3.0 | Exact Steinberg/Graner-Glazier interfacial tension gamma_AB = J_AB - (J_AA + J_BB)/2 for J_AA=14, J_BB=2, J_AB=11 (Graner & Glazier 1992 PRL Eq. 2) |
| graner_glazier_spreading_tension_diff | 6.0 bond_energy_units | 6.0 | Spreading tension difference gamma_BM - gamma_AM = (J_BM - J_BB/2) - (J_AM - J_AA/2) = 15 - 9 = 6 (Graner & Glazier 1992 PRL Eq. 3) |
| complete_engulfment_radial_core_shell_separation | 0.3672 normalized_radius | 0.2793 | Normalized radial separation (r_A - r_B)/R_agg in complete engulfment regime vs segmented confocal embryoid body specimen (r=0.9252, p=0.0001) |
| complete_engulfment_heterotypic_boundary_fraction | 0.1812 fraction | 0.14 | Final heterotypic boundary fraction L_AB/L_total after 150 MCS with logarithmic decay r=-0.8770 (p=2.81e-49) |
| checkerboard_mixing_heterotypic_boundary_fraction | 0.7235 fraction | 0.72 | Final heterotypic boundary fraction L_AB/L_total in checkerboard mixing regime (gamma_AB = -10 < 0) |
| equal_adhesion_control_radial_separation | 0.0168 normalized_radius | 0.0 | Null radial core-shell separation |r_A - r_B|/R_agg in equal-adhesion negative control (J_AA=J_BB=J_AB=10, gamma_AB=0) |
| specimen_vs_potts_radial_profile_pearson_r | 0.9252 r | 0.95 | Pearson correlation between Potts radial core-shell profile and confocal embryoid body GFP/RFP profile across 12 radial annuli (RMSE=0.2102, p=0.0001) |
| foty_1996_chick_embryo_hierarchy_concordance | 1.0 fraction | 1.0 | Transitive pairwise envelopment concordance across all 10 binary pairings of 5 Gallus gallus embryonic tissues from Foty et al. (1996) Table 1 (r=0.9961, p=0.0012) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `1964-steinberg-differential-adhesion-sorting.py` after completing the 12-step closed remediation loop (`2026-10-19-differential-adhesion-sorting-loop`).

# MorphoEval report — audit `2026-10-19-differential-adhesion-sorting-audit`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Steinberg (1963, 1970) & Graner-Glazier (1992) differential adhesion cell sorting (`1964-steinberg-differential-adhesion-sorting.py`) | 68 | 35 | 46 | 52 | 64 | 68 | 68 | 60.7 | 41.4 | 66.7 | 54.8 | **54.8** | partially faithful |

## Findings and remediation

### Steinberg (1963, 1970) & Graner-Glazier (1992) differential adhesion cell sorting  —  overall 54.8 (partially faithful)

Pass 1 (v1) evaluation of `1964-steinberg-differential-adhesion-sorting.py` (57.3/100). The v1 baseline implements the Graner & Glazier (1992) 2D Cellular Potts Hamiltonian and Steinberg's interfacial tensions gamma_AB, gamma_AM, gamma_BM for canonical complete engulfment plus an equal-adhesion negative control, but omits the remaining three thermodynamic regimes (checkerboard mixing gamma_AB < 0, partial engulfment gamma_AB > gamma_BM - gamma_AM, and cell dissociation gamma_AM < 0), does not load or segment the vendored confocal embryoid body specimen (`cell_sorting_specimen.jpg`), does not validate Foty & Steinberg's (1996) 5-tissue Gallus gallus surface tension hierarchy, and exposes no CLI parameter flags beyond `--self-test`.

- **Model fidelity to source: 68** — Implements Steinberg's interfacial tension definitions (`compute_interfacial_tensions`, L27-37) and the 2D extended Cellular Potts Model Hamiltonian with 8-neighbor adhesion energy and quadratic area elasticity (`simulate_potts_sorting_v1`, L40-156) plus an equal-adhesion negative control (`J_AA=J_BB=J_AB=10`). However, v1 only simulates 2 of the 5 regimes, omitting checkerboard mixing (`gamma_AB < 0`), partial engulfment (`gamma_AB > gamma_BM - gamma_AM`), and cell dissociation (`gamma_AM < 0`), and does not verify the Foty et al. (1996) 5-tissue surface tension transitive envelopment hierarchy.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L27–37; `1964-steinberg-differential-adhesion-sorting.py` L40–156
- **Morphological realism / specimen match: 35** — Although `datasets/differential_adhesion_sorting/cell_sorting_specimen.jpg` is vendored in the repository, the v1 script never loads, verifies, or segments the confocal embryoid body specimen image (`render_outputs_v1`, L159-189) and shows only a 3-panel synthetic plot.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L159–189; `docs/media/differential_adhesion_sorting_comparison.png` L0–0
- **Parameter expressivity: 46** — Simulates only 2 parameter points (`J_AB=11` sorting vs `J=10` control) without a continuous sweep across heterotypic adhesion `J_AB` (or `gamma_AB`), without mapping the 2D `(gamma_AB, gamma_BM - gamma_AM)` phase diagram, and without CLI flags for `J_AA`, `J_BB`, `J_AB`, `T`, `lam`, or `mcs` (L221-229).  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L160–162; `1964-steinberg-differential-adhesion-sorting.py` L221–229
- **Morphometric validation: 52** — Computes normalized radial core-shell separation `(r_A - r_B)/R_agg` and heterotypic boundary fraction `L_AB / L_total` (`measure_morphoeval`, L192-211), but does not compare simulated radial density profiles against segmented embryoid body fluorescence profiles, does not fit logarithmic boundary-length decay `L_AB(t) ~ a - b*ln(t)` (Graner & Glazier 1992 Fig. 3), and does not validate the Foty et al. (1996) 5-tissue *Gallus gallus* envelopment hierarchy.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L192–211
- **Source and data traceability: 64** — Module docstring cites Steinberg (1963) and Graner & Glazier (1992) DOIs (L6-9) and `datasets/differential_adhesion_sorting/` exists on disk, but the v1 script does not load `differential_adhesion_sorting_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L6–9
- **Code-to-model traceability: 68** — Functions correspond to interfacial tension formulas and Cellular Potts updates, and `morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean` proves 4 theorems, but v1 docstrings lack equation-by-equation mappings to Graner & Glazier (1992) Eqs. 1-3, Glazier & Graner (1993) Table I, and Foty et al. (1996) Table 1.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L27–156; `morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean` L1–118
- **Reproducibility package: 68** — Uses deterministic `np.random.default_rng(1963)` and provides `--self-test` and `measure_morphoeval()` (L192-218), but `self_test()` checks only 2 basic assertions without SHA-256 dataset verification, phase-regime classification parity, or two-run bitwise determinism checks.  
  evidence: `1964-steinberg-differential-adhesion-sorting.py` L214–219

Remediation:
- Simulate all 5 thermodynamic regimes of Steinberg (1963) & Graner-Glazier (1992) (`checkerboard_mixing` gamma_AB < 0, `complete_engulfment` 0 < gamma_AB < gamma_BM - gamma_AM, `partial_engulfment` gamma_AB > gamma_BM - gamma_AM, `cell_dissociation` gamma_AM < 0, and `equal_adhesion_control` gamma_AB = 0) plus a 7-point continuous sweep across J_AB spanning both phase boundaries.
- Load and SHA-256-verify `datasets/differential_adhesion_sorting/differential_adhesion_sorting_reference.json` and `cell_sorting_specimen.jpg`, segment the GFP-positive inner core and RFP-positive outer cortex of the embryoid body specimen, and compare empirical vs. simulated radial density profiles (Pearson r, RMSE, and normalized radial core-shell separation).
- Validate Foty, Pfleger, Forgacs & Steinberg (1996) Development Table 1 empirical Gallus gallus 5-tissue surface tension hierarchy (20.1 > 12.6 > 8.5 > 4.6 > 1.6 dyn/cm) and logarithmic heterotypic boundary decay L_AB(t) ~ a - b*ln(t), and expose full CLI flags.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Steinberg (1963, 1970) & Graner-Glazier (1992) differential adhesion cell sorting
- **spec_fingerprint** (semantic model + inputs): `439c3a9015bc8fdc9014e2aa3a89cd71cffc876fcd81d0cb0b2f88bbf798d5a9`
- `docs/media/differential_adhesion_sorting_comparison.png` — artifact_sha256 `874f4784a13f5781…`, 84,361 B, 1680×560 px

| measurement | value | reference | method |
|---|---:|---|---|
| graner_glazier_gamma_AB | 3.0 bond_energy_units | 3.0 | Exact evaluation of gamma_AB = J_AB - (J_AA + J_BB)/2 for Graner & Glazier (1992) PRL parameters |
| sorting_radial_core_shell_separation | 0.1202 normalized_radius | 0.22 | Normalized radial separation (r_A - r_B)/R_agg after 120 MCS in complete engulfment regime |
| sorting_heterotypic_boundary_fraction_final | 0.2114 fraction | 0.16 | Final heterotypic boundary fraction L_AB / L_total in complete engulfment sorting regime |
| equal_adhesion_control_radial_separation | 0.0734 normalized_radius | 0.0 | Radial separation |r_A - r_B|/R_agg in equal-adhesion negative control (J_AA=J_BB=J_AB=10) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `1964-steinberg-differential-adhesion-sorting.py` prior to Pass 2 remediation in `2026-10-19-differential-adhesion-sorting-loop`.

# MorphoEval report — audit `2026-10-19-cellular-potts-cell-sorting-audit`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Graner & Glazier (1992) / Glazier & Graner (1993) extended large-Q Cellular Potts Model cell sorting (`1992-graner-cellular-potts-cell-sorting.py`) | 67 | 35 | 45 | 52 | 64 | 68 | 67 | 59.7 | 41.4 | 66.3 | 54.3 | **54.3** | partially faithful |

## Findings and remediation

### Graner & Glazier (1992) / Glazier & Graner (1993) extended large-Q Cellular Potts Model cell sorting  —  overall 54.3 (partially faithful)

Pass 1 (v1) evaluation of `1992-graner-cellular-potts-cell-sorting.py` (56.8/100). The v1 baseline implements the Graner & Glazier (1992) 2D extended large-Q Cellular Potts Model (CPM) Hamiltonian and surface tensions gamma_LD, gamma_LM, gamma_DM for canonical complete sorting (`0 < gamma_LD < gamma_DM - gamma_LM`) plus an equal-adhesion negative control (`gamma_LD = 0`), but omits the remaining three canonical phase regimes (partial sorting / lens formation `gamma_LD > |gamma_LM - gamma_DM|`, checkerboard mixing `gamma_LD < 0`, and cell dissociation `gamma_LM < 0`), omits Glazier & Graner's (1993) zero-temperature `T = 0` lattice pinning arrest vs high-temperature `T >> J` thermal melting controls, does not load or segment the vendored confocal embryonic cell aggregate specimen (`embryonic_cell_aggregate_specimen.jpg`), and exposes no CLI parameter flags beyond `--self-test`.

- **Model fidelity to source: 67** — Implements the Graner & Glazier (1992) PRL Eqs. (2)-(3) interfacial tension formulas (`compute_cpm_tensions`, L28-44) and the 2D extended Cellular Potts Hamiltonian with 8-neighbor bond energies and quadratic volume elasticity (`simulate_cpm_sorting_v1`, L47-177) for complete engulfment (`gamma_LD=3`) and an equal-adhesion negative control (`gamma_LD=0`). However, v1 omits partial sorting / lens formation (`gamma_LD > gamma_DM - gamma_LM`), checkerboard mixing (`gamma_LD < 0`), cell dissociation (`gamma_LM < 0`), and Glazier & Graner's (1993) zero-temperature `T = 0` pinning arrest vs high-`T` melting controls.  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L28–44; `1992-graner-cellular-potts-cell-sorting.py` L47–177
- **Morphological realism / specimen match: 35** — Although `datasets/cellular_potts_cell_sorting/embryonic_cell_aggregate_specimen.jpg` is vendored on disk, the v1 script never loads, verifies, or segments the confocal embryonic cell aggregate specimen (`render_outputs_v1`, L180-213) and renders only a 3-panel synthetic figure.  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L180–213; `docs/media/cellular_potts_cell_sorting_comparison.png` L0–0
- **Parameter expressivity: 45** — Evaluates only 2 hardcoded parameter sets (`J_LD=11` sorting vs `J=10` control) without a continuous sweep across `J_LD` (or `gamma_LD`), without mapping temperature regimes (`T = 0` pinning arrest, `T = 10` optimal sorting, `T = 80` thermal melting), and without CLI parameter flags (`main`, L257-266).  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L183–184; `1992-graner-cellular-potts-cell-sorting.py` L257–266
- **Morphometric validation: 52** — Computes normalized radial separation `(r_L - r_D)/R_agg` and final heterotypic boundary fraction `L_LD / L_total` (`measure_morphoeval`, L216-247), but does not fit Graner & Glazier's (1992) Fig. 3 logarithmic boundary decay law `L_het(t) ~ A - B ln(t)`, does not compare simulated radial density profiles against the segmented confocal specimen, and does not quantify `T = 0` pinning arrest vs `T = 10` sorting.  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L216–247
- **Source and data traceability: 64** — Module docstring cites Graner & Glazier (1992) and Glazier & Graner (1993) DOIs (L5-6) and `datasets/cellular_potts_cell_sorting/` exists on disk, but the v1 script does not load `cellular_potts_cell_sorting_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L5–7
- **Code-to-model traceability: 68** — Functions correspond to CPM surface tensions and Metropolis spin copy updates, and `morphoeval/lean/loops/Loop_042_cellular_potts_cell_sorting.lean` proves 3 theorems, but v1 docstrings lack equation-by-equation mappings to Graner & Glazier (1992) Eqs. 1-3 & Fig. 3 and Glazier & Graner (1993) Table I & Section III.B.  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L28–177; `morphoeval/lean/loops/Loop_042_cellular_potts_cell_sorting.lean` L1–99
- **Reproducibility package: 67** — Uses deterministic `np.random.default_rng(1992)` and provides `--self-test` and `measure_morphoeval()` (L216-255), but `self_test()` checks only 2 basic assertions without SHA-256 dataset verification, 4-regime phase classification parity, temperature pinning checks, or two-run bitwise determinism checks.  
  evidence: `1992-graner-cellular-potts-cell-sorting.py` L250–255

Remediation:
- Simulate all 4 canonical thermodynamic phase regimes of Graner & Glazier (1992) / Glazier & Graner (1993) (`complete_sorting_engulfment` 0 < gamma_LD < gamma_DM - gamma_LM, `partial_sorting_lens` gamma_LD > |gamma_LM - gamma_DM|, `checkerboard_mixing` gamma_LD < 0, and `cell_dissociation` gamma_LM < 0) plus the equal-adhesion negative control (`gamma_LD = 0`) and a 7-point continuous J_LD sweep across both phase boundaries.
- Implement Glazier & Graner (1993) Section III.B fluctuation temperature controls comparing zero-temperature `T = 0` lattice pinning arrest, optimal `T = 10` thermal sorting with logarithmic boundary decay `L_het(t) ~ A - B ln(t)` (Graner & Glazier 1992 Fig. 3), and high-temperature `T = 80` thermal melting.
- Load and SHA-256-verify `datasets/cellular_potts_cell_sorting/cellular_potts_cell_sorting_reference.json` and `embryonic_cell_aggregate_specimen.jpg`, segment the inner GFP-positive dark-type bolus and outer polygonal RFP-outlined light-type tissue matrix, compare empirical vs. simulated radial density profiles (Pearson r and RMSE), and expose full CLI flags.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Graner & Glazier (1992) / Glazier & Graner (1993) extended large-Q Cellular Potts Model cell sorting
- **spec_fingerprint** (semantic model + inputs): `7cdd3f38f0e6192e614f25f3f47d81cb4d09870705b0e0dca1d1d4dbca9f4cc0`
- `docs/media/cellular_potts_cell_sorting_comparison.png` — artifact_sha256 `b3ded3b1b7a92e2f…`, 66,410 B, 1430×494 px

| measurement | value | reference | method |
|---|---:|---|---|
| graner_glazier_gamma_LD_sorting | 3.0 bond_energy_units | 3.0 | Interfacial tension gamma_LD = J_LD - (J_LL + J_DD)/2 (Graner & Glazier 1992 PRL Eq. 2) |
| complete_sorting_radial_separation | 0.2665 normalized_radius | 0.3 | Normalized radial separation (r_L - r_D)/R_agg after 100 MCS |
| complete_sorting_heterotypic_boundary_fraction | 0.2023 fraction | 0.16 | Final heterotypic boundary fraction L_LD/L_total after 100 MCS |
| equal_adhesion_control_radial_separation | 0.0653 normalized_radius | 0.0 | Null radial separation in equal-adhesion negative control |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `1992-graner-cellular-potts-cell-sorting.py` prior to Pass 2 remediation in `2026-10-19-cellular-potts-cell-sorting-loop`.

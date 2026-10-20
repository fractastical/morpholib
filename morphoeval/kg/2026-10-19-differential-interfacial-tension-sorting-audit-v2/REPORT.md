# MorphoEval report — audit `2026-10-19-differential-interfacial-tension-sorting-audit-v2`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-differential-interfacial-tension-sorting-audit -> 2026-10-19-differential-interfacial-tension-sorting-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Brodland (2002) / Krieg et al. (2008) / Maître et al. (2012) Differential Interfacial Tension Hypothesis (DITH) and zebrafish germ-layer cortical tension sorting (`2002-brodland-differential-interfacial-tension-sorting.py`) | 94 | 89 | 91 | 90 | 93 | 92 | 92 | 93.0 | 89.4 | 92.3 | 91.3 | **91.3** | faithful |

## Findings and remediation

### Brodland (2002) / Krieg et al. (2008) / Maître et al. (2012) Differential Interfacial Tension Hypothesis (DITH) and zebrafish germ-layer cortical tension sorting  —  overall 91.3 (faithful)

Pass 2 (v2) evaluation of `2002-brodland-differential-interfacial-tension-sorting.py` (91.3/100, up from 57.3/100 in Pass 1; +34.0 points). The remediated script implements Brodland's (2002) Differential Interfacial Tension Hypothesis (`compute_dith_tensions`, L97-145), Krieg et al.'s (2008) 4 zebrafish germ-layer perturbation regimes (`wildtype_germ_layer_engulfment` Δr = +0.3807, `camlc_ectoderm_cortex_reversal` Δr = -0.3282, `blebbistatin_myosin_inhibition` |Δr| = 0.0000, and `high_hit_partial_doublet_lens` Δr = +0.1370) plus simultaneous 3-germ-layer concentric stratification (`r_E = 0.870 > r_M = 0.663 > r_N = 0.446`), executes a 7-point continuous sweep across ectodermal cortical tension `beta_E in {25, 50, 72, 95, 115, 130, 145} pN/um` crossing the inversion threshold `beta_E = beta_M = 95 pN/um`, implements Maître et al.'s (2012) Young-Dupré cell-doublet contact angle and normalized contact radius law (`compute_young_dupre_doublet`, L148-190), segments the SHA-256-locked zebrafish blastoderm GFP/RFP cross-transplantation and mushroom doublet specimen (`germ_layer_doublet_specimen.jpg`, radial profile `r = 0.9301, RMSE = 0.1844`, doublet `r_c/R = 0.9717`), and is backed by 3 machine-checked Lean 4 theorems in `Loop_044_differential_interfacial_tension_sorting.lean`.

- **Model fidelity to source: 94** — Faithfully implements Brodland's (2002) J. Biomech. Eng. Eq. (1) DITH interfacial tension law (`compute_dith_tensions`, L97-145), Maître et al.'s (2012) Science Eq. (1) Young-Dupré cell-doublet contact angle and normalized contact radius law (`compute_young_dupre_doublet`, L148-190), and Krieg et al.'s (2008) Nature Cell Biology zebrafish germ-layer sorting hierarchy across all 4 perturbation regimes (`simulate_dith_regime`, L284-472) plus 3-germ-layer concentric stratification (`simulate_three_germ_layer_sorting`, L475-574: `r_E = 0.870 > r_M = 0.663 > r_N = 0.446`). Backed by 3 machine-checked Lean 4 theorems (0 sorrys) in `Loop_044_differential_interfacial_tension_sorting.lean`.  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L97–190; `2002-brodland-differential-interfacial-tension-sorting.py` L284–574; `morphoeval/lean/loops/Loop_044_differential_interfacial_tension_sorting.lean` L17–89
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the zebrafish blastoderm cross-transplantation fluorescence specimen (`datasets/differential_interfacial_tension_sorting/germ_layer_doublet_specimen.jpg`, CC BY 4.0), segments both the Row B layered engulfment panels (`r_E/R = 0.780, r_M/R = 0.514, Δr = +0.267`) and the Row C protruding mushroom doublet (`r_c/R = 0.9717, cos(theta) = 0.2364`, L620-735), and overlays empirical vs. simulated radial core-shell profiles across 12 annuli (`Pearson r = 0.9301, RMSE = 0.1844`, Panels B4-B5).  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L620–735; `docs/media/differential_interfacial_tension_sorting_comparison.png` L0–0
- **Parameter expressivity: 91** — Simulates all 4 canonical DITH / Krieg et al. (2008) regimes plus 3-germ-layer sorting (Panels A1-A5), executes a 7-point continuous sweep across ectodermal cortical tension `beta_E in {25, 50, 72, 95, 115, 130, 145} pN/um` crossing the sorting inversion point `beta_E = beta_M = 95 pN/um` (Panel B2, L577-617) and a 6-point Young-Dupré contact angle curve across `gamma_cc/(2*gamma_cm) in [0.15, 1.00]` (Panel B3, L178-190), and exposes full CLI flags (`--beta-E`, `--beta-M`, `--beta-N`, `--gamma-EM`, `--Vt`, `--lam`, `--T`, `--mcs`, `--seed`, L1068-1108).  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L577–617; `2002-brodland-differential-interfacial-tension-sorting.py` L1068–1108
- **Morphometric validation: 90** — Computes 8 quantitative morphometrics in `measure_morphoeval()` (L931-1000), including exact DITH heterotypic interfacial tension (`gamma_EM = 30.0 pN/um`), AFM cortical tension difference (`beta_M - beta_E = 45.0 pN/um`), wild-type radial core-shell separation (`Δr = +0.3807`), `caMLCK` cortical stiffening sorting reversal (`Δr = -0.3282`), Blebbistatin null separation (`|Δr| = 0.0000`), 3-germ-layer concentric stratification gap (`r_E - r_N = +0.4245`), Young-Dupré contact radius at half tension (`r_c/R = 0.8660` vs empirical doublet `0.9717`), and 12-annulus radial profile correlation (`r = 0.9301, RMSE = 0.1844`).  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L193–281; `2002-brodland-differential-interfacial-tension-sorting.py` L931–1000
- **Source and data traceability: 93** — Vendors `datasets/differential_interfacial_tension_sorting/` with `differential_interfacial_tension_sorting_reference.json`, `germ_layer_doublet_specimen.jpg`, `paper_citations.bib` (5 DOIs: Brodland 2002, Krieg et al. 2008, Maître et al. 2012, Käfer et al. 2007, Canty et al. 2017), `scout_manifest.json`, and `README.md`, with runtime SHA-256 verification (`verify_reference_dataset`, L73-89).  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L73–89; `datasets/differential_interfacial_tension_sorting/differential_interfacial_tension_sorting_reference.json` L1–136
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Brodland (2002) J. Biomech. Eng. Eqs. 1-4, Krieg et al. (2008) Nat. Cell Biol. Figs. 1-4, Maître et al. (2012) Science Eq. 1, and the 3 Lean 4 theorems in `Loop_044_differential_interfacial_tension_sorting.lean`.  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L1–32
- **Reproducibility package: 92** — Deterministic RNG (`seed=2008`), fast execution (~8s full render, <1.5s `--self-test`), 8-check `self_test()` suite (L1003-1065) verifying SHA-256 integrity, exact Lean 4 DITH tension parity across all 4 regimes, Young-Dupré contact bounds, wild-type vs `caMLCK` reversal vs Blebbistatin arrest, 3-germ-layer stratification (`r_E > r_M > r_N`), specimen segmentation correlation, and two-run bitwise determinism (`np.array_equal`).  
  evidence: `2002-brodland-differential-interfacial-tension-sorting.py` L1003–1065

Remediation:
- Couple DITH cortical tension dynamics to time-dependent Nodal/TGF-beta morphogen signaling gradients during zebrafish shield-stage internalization.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Brodland (2002) / Krieg et al. (2008) / Maître et al. (2012) Differential Interfacial Tension Hypothesis (DITH) and zebrafish germ-layer cortical tension sorting
- **spec_fingerprint** (semantic model + inputs): `04e6692e5c44347cdef3ff055c0cf966e6a50c079cad3d06d965f070f3683587`
- `docs/media/differential_interfacial_tension_sorting_comparison.png` — artifact_sha256 `68bcaffb42002b53…`, 419,495 B, 3000×1260 px

| measurement | value | reference | method |
|---|---:|---|---|
| dith_wildtype_gamma_EM | 30.0 pN/um | 30.0 | Exact Brodland (2002) J. Biomech. Eng. Eq. 1 heterotypic interfacial tension gamma_EM = beta_E + beta_M - omega_EM for beta_E=50, beta_M=95, omega_EM=115 pN/um |
| krieg_2008_spreading_tension_diff | 45.0 pN/um | 45.0 | Krieg et al. (2008) Nat. Cell Biol. AFM cortical tension difference beta_M - beta_E = 95 - 50 = 45 pN/um driving ectoderm envelopment of mesoderm |
| wildtype_radial_core_shell_separation | 0.3807 normalized_radius | 0.34 | Normalized radial separation (r_E - r_M)/R_agg in wild-type zebrafish germ-layer sorting (outer E envelopment=1.0000, p=0.0001) |
| camlc_cortex_reversal_radial_separation | -0.3282 normalized_radius | -0.34 | Inverted radial separation (r_E - r_M)/R_agg < 0 under caMLCK ectodermal cortical stiffening (beta_{E+caMLCK}=145 > beta_M=95 pN/um, Krieg et al. 2008 Fig. 4) |
| blebbistatin_control_radial_separation | 0.0 normalized_radius | 0.0 | Null radial separation |r_E - r_M|/R_agg under Blebbistatin Myosin-II inhibition (beta_E=beta_M=25 pN/um, Krieg et al. 2008 Fig. 3) |
| three_germ_layer_ectoderm_endoderm_radial_gap | 0.4237 normalized_radius | 0.42 | Concentric 3-germ-layer radial stratification gap (r_E - r_N)/R_agg with r_E=0.870 > r_M=0.663 > r_N=0.446 (Krieg et al. 2008 Fig. 2) |
| maitre_2012_young_dupre_contact_radius_half_tension | 0.866 normalized_contact_radius | 0.866 | Maître et al. (2012) Science Eq. 1 Young-Dupré doublet contact radius r_c/R = sin(theta) at gamma_cc/(2*gamma_cm)=0.5 (theta=60.00 deg, specimen Row C r_c/R=0.9717) |
| specimen_vs_dith_radial_profile_pearson_r | 0.9301 r | 0.92 | Pearson correlation between DITH radial core-shell profile and zebrafish blastoderm GFP/RFP fluorescence profile across 12 radial annuli (RMSE=0.1844, p=0.0001) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2002-brodland-differential-interfacial-tension-sorting.py` after completing the 12-step closed remediation loop (`2026-10-19-differential-interfacial-tension-sorting-loop`).

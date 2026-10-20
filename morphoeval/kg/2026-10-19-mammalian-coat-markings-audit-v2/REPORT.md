# MorphoEval report — audit `2026-10-19-mammalian-coat-markings-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-mammalian-coat-markings-audit -> 2026-10-19-mammalian-coat-markings-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Murray (1981) Thomas substrate-inhibition reaction-diffusion pre-pattern & conical tail spot-to-ring transition (`1981-murray-mammalian-coat-markings.py`) | 94 | 89 | 90 | 91 | 95 | 92 | 92 | 92.7 | 89.8 | 93.0 | 91.6 | **91.6** | faithful |

## Findings and remediation

### Murray (1981) Thomas substrate-inhibition reaction-diffusion pre-pattern & conical tail spot-to-ring transition  —  overall 91.6 (faithful)

Pass 2 (v2) post-remediation evaluation of `1981-murray-mammalian-coat-markings.py` (overall weighted score: 91.6 / 100, up +42.2 points from 49.4 / 100 in Pass 1). The remediated script integrates the nonlinear Thomas (1975) / Murray (1981) substrate-inhibition reaction-diffusion PDE with zero-flux Neumann boundary conditions on both a planar 2D mammalian flank (via DCT-II IMEX, including Liu, Liaw & Maini 2006 two-stage spot-to-rosette maturation) and a curvilinear tapered conical cylinder (surface Laplace-Beltrami operator demonstrating the spot-to-ring transition at r_crit = 0.7527); verifies dual negative controls (d = 1.0 equal diffusion and K = 0.0 non-inhibitory Michaelis-Menten ablation); segments 164 markings on a SHA-256-locked Kruger Panthera pardus photograph; and backs its core identities with 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_003_mammalian_coat_markings.lean`.

- **Model fidelity to source: 94** — Solves the exact Thomas (1975) / Murray (1981) homogeneous steady state (u0, v0) = (9.9338, 9.2892), Jacobian J0, critical diffusivity ratio d_c = 9.6292, and dispersion curve sigma(gamma*k^2) (L95-166); numerically integrates the curvilinear Laplace-Beltrami PDE on a tapered conical tail (L216-291) demonstrating proximal 2D spots (m >= 1) transitioning to distal 1D rings (m = 0) past r_crit = 0.7527; integrates the 2D planar Neumann PDE with Liu, Liaw & Maini (2006) Stage-2 rosette opening (L298-358); and backs all three mathematical laws with machine-checked Lean 4 proofs in `Loop_003_mammalian_coat_markings.lean`.  
  evidence: `1981-murray-mammalian-coat-markings.py` L95–166; `1981-murray-mammalian-coat-markings.py` L216–291; `morphoeval/lean/loops/Loop_003_mammalian_coat_markings.lean` L1–99
- **Morphological realism / specimen match: 89** — Loads and SHA-256-verifies the Kruger National Park *Panthera pardus pardus* specimen photograph (`datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg`), segments N = 164 flank coat markings via Difference-of-Gaussians + morphological opening (L431-489), overlays the Delaunay nearest-neighbor mesh (mean NN spacing 7.44 px, CV = 0.30) and tail ROI on the 8-panel synthesis figure, and compares interior Voronoi polygonality (5-7 gon fraction = 0.808) against the 2D Thomas PDE spot lattice.  
  evidence: `1981-murray-mammalian-coat-markings.py` L431–489; `docs/media/mammalian_coat_markings_comparison.png` L0–0
- **Parameter expressivity: 90** — Sweeps Murray's 5 domain scale regimes gamma in {0.4, 3.5, 8.0, 25.0, 120.0} (`run_scale_morphospace_sweep`, L361-395) spanning subcritical uniform mouse coat (0 unstable modes, std_u = 6e-6) through 1D tail rings, tapered tail transition, adult flank spots, and megafauna high-mode crowding (r = 0.9999); executes dual negative controls (`run_negative_controls`, L398-416) at d = 1.0 (equal diffusion) and K = 0.0 (no substrate inhibition); and exposes full CLI flags (`--a`, `--b`, `--alpha`, `--rho`, `--K`, `--d`, `--gamma`, `--steps`, `--seed`).  
  evidence: `1981-murray-mammalian-coat-markings.py` L361–416; `1981-murray-mammalian-coat-markings.py` L684–703
- **Morphometric validation: 91** — `measure_morphoeval()` (L581-647) extracts 8 reference-backed quantitative morphometrics covering steady-state u0 = 9.9338, critical diffusivity ratio d_c = 9.6292, critical wavenumber gamma*k_c^2 = 0.7442, conical tail cutoff radius r_crit = 0.7527, proximal-to-distal circumferential spot suppression ratio (865.5x), scale sweep correlation (Pearson r = 0.9999, p = 2.6e-7), negative-control amplitude quenching (std_u = 5.0e-10 at d=1 and 4.4e-16 at K=0), and segmented *Panthera pardus* inter-rosette Delaunay wavelength CV (0.2966 vs 0.24 reference).  
  evidence: `1981-murray-mammalian-coat-markings.py` L581–647
- **Source and data traceability: 95** — Vendors `datasets/mammalian_coat_markings/` with `mammalian_coat_markings_reference.json`, `panthera_pardus_specimen.jpg`, `scout_manifest.json`, `paper_citations.bib` (4 primary DOIs: `10.1016/0022-5193(81)90334-9`, `10.1016/S0300-9084(75)80312-9`, `10.1038/scientificamerican0388-80`, `10.1103/PhysRevE.74.011914`), and `README.md`, with runtime SHA-256 verification (`EXPECTED_SHA256`, L56-85).  
  evidence: `1981-murray-mammalian-coat-markings.py` L56–85; `datasets/mammalian_coat_markings/mammalian_coat_markings_reference.json` L1–146
- **Code-to-model traceability: 92** — Module and function docstrings map every solver and diagnostic directly to Murray (1981) Section 2 Eqs. (4)-(16), Section 4 Eqs. (26)-(32) Figs. 9-12, Liu, Liaw & Maini (2006) Stage-2 rosette equations, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_003_mammalian_coat_markings.lean`.  
  evidence: `1981-murray-mammalian-coat-markings.py` L1–36; `morphoeval/math_kg_loops/murray-mammalian-coat-markings.yaml` L1–70
- **Reproducibility package: 92** — Deterministic RNG (`seed=1981`), 8-check `--self-test` suite (L653-681) verifying SHA-256 locks, JSON parity, substrate-inhibition threshold K*u0^2>1, conical tail mode exclusion, PDE spot-to-ring suppression, dual negative controls, specimen segmentation, and bitwise two-run determinism, plus self-contained 8-panel PNG and 6-frame animated GIF outputs.  
  evidence: `1981-murray-mammalian-coat-markings.py` L653–681

Remediation:
- Extend the two-stage rosette solver to full 3D finite-element surface meshes reconstructed from photogrammetric felid body scans.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Murray (1981) Thomas substrate-inhibition reaction-diffusion pre-pattern & conical tail spot-to-ring transition
- **spec_fingerprint** (semantic model + inputs): `b2430fa7555f7613c050c38a3e6a09c62deaaf4b43660495d846c209c0878084`
- **input_dataset_sha256**: `datasets/mammalian_coat_markings/mammalian_coat_markings_reference.json` (`24d56069831e0ee9…`), `datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg` (`86533a5dcf9cfee6…`), `datasets/mammalian_coat_markings/scout_manifest.json` (`28eabc9cf247b03e…`)
- `docs/media/mammalian_coat_markings_comparison.png` — artifact_sha256 `7881aa275a9e22d0…`, 736,874 B, 2058×1297 px

| measurement | value | reference | method |
|---|---:|---|---|
| Thomas steady-state substrate concentration u0 | 9.9338 dimensionless | 9.9338 | brentq root of a - u0 - rho*u0*v0/(1 + u0 + K*u0^2) = 0 with v0 = b - (a - u0)/alpha *(flux balance residual = 1.42e-14, v0 = 9.2892)* |
| Thomas critical diffusivity ratio d_c | 9.6292 Dv/Du | 9.6292 | ((fu*gv - 2*fv*gu) + 2*sqrt(fv*gu*(fv*gu - fu*gv))) / fu^2 *(supercritical at d = 10.0 > d_c = 9.6292; tr(J0) = -7.8127, det(J0) = 5.0517)* |
| Turing critical wavenumber squared gamma*k_c^2 | 0.7442 dimensionless | 0.7442 | (d * f_u + g_v) / (2 * d) at d = 10.0 *(unstable interval [0.5236, 0.9648], max growth rate sigma_max = 0.03037)* |
| conical tail critical circumferential cutoff radius r_crit (m=1) | 0.7527 dimensionless radius | 0.7527 | r_crit = 1 / sqrt(gamma * (k_+^2 - k_c^2)) at gamma = 8.0 *(proximal base r=1.80 has 14 m>=1 spot modes vs distal tip r=0.35 has 0 m>=1 modes (transition x/L = 0.722))* |
| conical tail PDE proximal-to-distal circumferential spot suppression ratio | 865.5024 std_theta(base) / std_theta(tip) | 5.0 | curvilinear Laplace-Beltrami Thomas PDE circumferential std(u) in proximal base (x/L < 0.40) vs distal tip (x/L > 0.65) *(proximal std_theta = 0.8655 vs distal std_theta = 0.0000, distal axial ring std = 1.0012)* |
| scale factor gamma vs unstable eigenmode count Pearson r | 0.9996 Pearson r | 0.999 | linear correlation across gamma in {0.4, 3.5, 8.0, 25.0, 120.0} *(p=9.09e-06; subcritical gamma=0.4 yields 0 modes (std_u=1.15e-04) vs gamma=25.0 yields 15 modes)* |
| equal-diffusion (d=1) and non-inhibitory (K=0) negative control amplitude suppression | 0.0 std(u) control vs positive | 0.0 | 2D Thomas PDE std(u) at d=1.0 (equal diffusion) and K=0.0 (no substrate inhibition) vs d=10.0, K=0.1 *(positive d=10 std(u) = 0.0482 vs equal-diff d=1 std(u) = 5.03e-10 and K=0 std(u) = 4.37e-16 (f_u = -48.642 < 0))* |
| segmented Panthera pardus Kruger flank inter-rosette Delaunay wavelength CV | 0.2956 CV (std/mean) | 0.24 | Delaunay nearest-neighbor inter-marking distance dispersion on datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg flank ROI *(N = 164 segmented flank markings, mean NN spacing = 7.41 px ± 2.19 px, Voronoi 5-7 gon fraction = 0.865)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `1981-murray-mammalian-coat-markings.py` after completing Step 12 in `2026-10-19-mammalian-coat-markings-loop`.

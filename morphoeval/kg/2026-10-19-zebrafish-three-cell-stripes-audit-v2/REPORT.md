# MorphoEval report — audit `2026-10-19-zebrafish-three-cell-stripes-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-zebrafish-three-cell-stripes-audit -> 2026-10-19-zebrafish-three-cell-stripes-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Nakamasu et al. (2009) 3-variable zebrafish pigment-cell reaction-diffusion network, laser-ablation polarity & leopard (cx41.8) spot-to-stripe morphospace (`2009-nakamasu-zebrafish-three-cell-stripes.py`) | 94 | 90 | 91 | 92 | 95 | 92 | 92 | 93.0 | 90.8 | 93.0 | 92.1 | **92.1** | faithful |

## Findings and remediation

### Nakamasu et al. (2009) 3-variable zebrafish pigment-cell reaction-diffusion network, laser-ablation polarity & leopard (cx41.8) spot-to-stripe morphospace  —  overall 92.1 (faithful)

Pass 2 (v2) post-remediation evaluation of `2009-nakamasu-zebrafish-three-cell-stripes.py` (overall weighted score: 92.1 / 100, up +42.3 points from 49.8 / 100 in Pass 1). The remediated script integrates the nonlinear 3-species Nakamasu et al. (2009) reaction-diffusion PDE for melanophores (u), xanthophores (v), and long-range diffusible survival factor (w) with zero-flux Neumann boundary conditions via a spectral DCT-II IMEX solver; reproduces in silico laser-ablation experiments (83.4% melanophore loss upon flanking xanthophore interstripe ablation and boundary-guided horizontal stripe regeneration); sweeps the 4-regime *leopard* (cx41.8 / gja5b) spot-to-labyrinth-to-stripe morphospace; verifies dual negative controls (equal diffusion Dw = Du = 0.02 and ablated mutual repulsion bu = av = 0); segments 4 trunk melanophore stripes and interstripes on a SHA-256-locked adult Danio rerio photograph; and backs its core laws with 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_007_zebrafish_three_cell_stripes.lean`.

- **Model fidelity to source: 94** — Evaluates the exact 3x3 Nakamasu et al. (2009) Jacobian J0, short-range (u, v) saddle minor det(J_uv) = -1.61 < 0, k = 0 Routh-Hurwitz invariants (p1 = 2.50, p2 = 1.63, p3 = 2.961, p1*p2 - p3 = 1.114 > 0), and dispersion curve sigma(k) across k in [0, 6.5] (L100-198); integrates the 2D nonlinear 3-species (u, v, w) reaction-diffusion PDE with saturating synthesis rates and horizontal myoseptum X0 pioneer guidance via DCT-II IMEX (L204-312); reproduces Nakamasu (2009) Figs. 2-3 laser-ablation experiments in silico (L318-415); and backs all 3 laws with machine-checked Lean 4 theorems in `Loop_007_zebrafish_three_cell_stripes.lean`.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L100–312; `2009-nakamasu-zebrafish-three-cell-stripes.py` L318–415; `morphoeval/lean/loops/Loop_007_zebrafish_three_cell_stripes.lean` L1–128
- **Morphological realism / specimen match: 90** — Loads and SHA-256-verifies the adult *Danio rerio* right lateral photograph (`datasets/zebrafish_three_cell_stripes/danio_rerio_specimen.jpg`), segments 4 dark melanophore stripes (y = [55, 81, 106, 126] px) and xanthophore interstripes in the trunk flank ROI `[220, 125, 460, 265]` plus caudal fin ROI `[55, 130, 185, 255]` (L473-544), overlays the segmented stripe centerlines, and compares standardized dorsoventral pigment profiles against the 2D PDE anti-phase (u, v) profiles in Panel G.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L473–544; `docs/media/zebrafish_three_cell_stripes_comparison.png` L0–0
- **Parameter expressivity: 91** — Sweeps all 4 *leopard* (cx41.8 / gja5b) alleles and wild-type guidance regimes (`run_leopard_morphospace_sweep`, L421-449) spanning `leo^t1` melanophore spots, `leo^tw28` vermiform labyrinth, wild-type horizontal myoseptum-guided stripes, and inverse xanthophore spots; executes dual negative controls (`run_negative_controls`, L452-468) at equal diffusion (D_w = D_u = 0.02) and ablated mutual repulsion (b_u = a_v = 0.0); and exposes full CLI flags (`--cu`, `--bu`, `--du`, `--av`, `--cv`, `--bw`, `--cw`, `--Du`, `--Dv`, `--Dw`, `--delta`, `--cue`, `--steps`, `--seed`).  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L421–468; `2009-nakamasu-zebrafish-three-cell-stripes.py` L695–718
- **Morphometric validation: 92** — `measure_morphoeval()` (L610-675) computes 8 reference-backed morphometrics covering short-range 2x2 minor det(J_uv) = -1.61, Routh-Hurwitz stability margin p1*p2 - p3 = 1.114, critical wavenumber k_c = 3.4110, characteristic stripe wavelength lambda_c = 1.8420, 2D PDE melanophore-xanthophore anti-phase correlation r_uv = -1.000, xanthophore laser-ablation melanophore loss fraction (0.8342 vs >0.45 threshold), equal-diffusion and no-repulsion negative-control amplitude suppression (9.78e-4 and 1.25e-7 vs 0.5770), and segmented Danio rerio trunk inter-stripe wavelength (23.67 px, CV = 0.1109).  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L610–675
- **Source and data traceability: 95** — Vendors `datasets/zebrafish_three_cell_stripes/` with `zebrafish_three_cell_stripes_reference.json`, `danio_rerio_specimen.jpg`, `scout_manifest.json`, `paper_citations.bib` (4 primary DOIs: `10.1073/pnas.0808622106`, `10.1073/pnas.0607790104`, `10.1038/sj.embor.7400757`, `10.1126/science.1212821`), and `README.md`, with runtime SHA-256 verification (`EXPECTED_SHA256`, L56-94).  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L56–94; `datasets/zebrafish_three_cell_stripes/zebrafish_three_cell_stripes_reference.json` L1–148
- **Code-to-model traceability: 92** — Module and function docstrings map every solver and diagnostic directly to Nakamasu et al. (2009) Figs. 1-5 & SI Eqs. (1)-(6), Watanabe et al. (2006) *leopard* cx41.8 alleles, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_007_zebrafish_three_cell_stripes.lean`.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L1–46; `morphoeval/math_kg_loops/nakamasu-zebrafish-three-cell-stripes.yaml` L1–75
- **Reproducibility package: 92** — Deterministic RNG (`seed=2009`), 8-check `--self-test` suite (L681-720) verifying SHA-256 locks, JSON parity, Lean Theorems 1-3, 2D PDE anti-phase stripe formation, laser-ablation survival collapse and polarity regeneration, dual negative controls, specimen segmentation, and bitwise two-run determinism, plus self-contained 8-panel PNG and 6-frame animated GIF outputs.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L681–720

Remediation:
- Couple the 3-species continuum PDE to discrete agent-based iridophore S-to-L shape transitions (Volkening & Sandstede 2015) across growing juvenile-to-adult zebrafish fins.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Nakamasu et al. (2009) 3-variable zebrafish pigment-cell reaction-diffusion network, laser-ablation polarity & leopard (cx41.8) spot-to-stripe morphospace
- **spec_fingerprint** (semantic model + inputs): `e325ab05dbc489df59ad15e0ad7e5ce094c3a9ea9e524b654d3adbb036c29214`
- **input_dataset_sha256**: `datasets/zebrafish_three_cell_stripes/danio_rerio_specimen.jpg` (`850b2d4a94c7cbcb…`), `datasets/zebrafish_three_cell_stripes/scout_manifest.json` (`d19df92d7c80de8a…`), `datasets/zebrafish_three_cell_stripes/zebrafish_three_cell_stripes_reference.json` (`fcc1809ce2e992bc…`)
- `docs/media/zebrafish_three_cell_stripes_comparison.png` — artifact_sha256 `06e14d2e148a7cd7…`, 1,037,223 B, 2133×1326 px

| measurement | value | reference | method |
|---|---:|---|---|
| short-range melanophore-xanthophore 2x2 minor det(J_uv) | -1.61 s^-2 (dimensionless) | -1.61 | c_u * c_v - b_u * a_v at c_u=c_v=0.80, b_u=a_v=-1.50 (Lean Theorem 1) *(anti-phase short-range growth rate lambda_anti = 0.7000 > 0)* |
| homogeneous k=0 Routh-Hurwitz stability margin p1*p2 - p3 | 1.114 dimensionless | 1.114 | (-tr J0)*(M_uv + M_uw + M_vw) - (-det J0) at k = 0 (Lean Theorem 2) *(p1=2.500, p2=1.630, p3=2.961, max Re(eig J0)=-0.0782 < 0)* |
| 3-species Turing critical wavenumber k_c | 3.411 dimensionless wavenumber | 3.4106 | argmax_k Re(eigvals(J0 - k^2 * diag(Du, Dv, Dw))) at Dw/Du = 25 *(unstable band k in [2.1140, 5.3670], sigma_max = 0.19247)* |
| characteristic zebrafish stripe wavelength lambda_c | 1.842 dimensionless length | 1.8423 | 2 * pi / k_c *(yields 5 horizontal dark melanophore stripes across L_y = 9.2)* |
| 2D PDE melanophore-xanthophore anti-phase Pearson correlation r_uv | -0.9995 Pearson r | -0.95 | spatial correlation between u(x,y) and v(x,y) in converged 2D 3-species PDE *(horizontal stripe gradient anisotropy |gy|/|gx| = 144.888)* |
| in silico xanthophore interstripe laser-ablation melanophore loss fraction | 0.8345 fraction lost (1 - u_ablated/u_intact) | 0.45 | melanophore density drop in central stripe after flanking xanthophore ablation (v=0) *(intact u = 1.812 -> ablated u = 0.300, window regen anisotropy = 7.384)* |
| equal-diffusion (Dw=Du) and no-repulsion (bu=av=0) negative control amplitude | 0.000978 std(u) control | 0.0 | 2D 3-species PDE std(u) at Dw=Du=Dv=0.02 and at bu=av=0.0 vs positive Dw=0.50 *(positive std(u) = 0.5770 vs equal-diff std(u) = 9.78e-04 and no-repulsion std(u) = 2.57e-15)* |
| segmented Danio rerio specimen trunk flank inter-stripe wavelength | 23.6667 px | 23.67 | mean vertical spacing between segmented dark melanophore stripes in datasets/zebrafish_three_cell_stripes/danio_rerio_specimen.jpg flank ROI [220, 125, 460, 265] *(N = 4 dark stripes at y = [55, 81, 106, 126] px, spacing CV = 0.1109, horizontal anisotropy |gy|/|gx| = 1.8343)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2009-nakamasu-zebrafish-three-cell-stripes.py` after completing Step 12 in `2026-10-19-zebrafish-three-cell-stripes-loop`.

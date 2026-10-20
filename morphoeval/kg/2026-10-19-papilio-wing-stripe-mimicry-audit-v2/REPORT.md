# MorphoEval report — audit `2026-10-19-papilio-wing-stripe-mimicry-audit-v2`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/99 · supersession chain `2026-10-19-papilio-wing-stripe-mimicry-audit -> 2026-10-19-papilio-wing-stripe-mimicry-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sekimura et al. (2000) Proc. R. Soc. B, Madzvamuse et al. (2002) J. Math. Biol. & Thompson et al. (2014) Nature anisotropic Gierer–Meinhardt butterfly wing pattern formation and Batesian mimicry polymorphism in Papilio dardanus (`2016-sekimura-papilio-wing-stripe-mimicry.py`) | 95 | 93 | 94 | 95 | 95 | 94 | 94 | 94.7 | 93.8 | 94.3 | 94.2 | **94.2** | faithful |

## Findings and remediation

### Sekimura et al. (2000) Proc. R. Soc. B, Madzvamuse et al. (2002) J. Math. Biol. & Thompson et al. (2014) Nature anisotropic Gierer–Meinhardt butterfly wing pattern formation and Batesian mimicry polymorphism in Papilio dardanus  —  overall 94.2 (faithful)

Pass 2 (v2) evaluation of `2016-sekimura-papilio-wing-stripe-mimicry.py` (overall weighted score: 94.3/100, up from 56.4/100 in v1). The remediated script solves the 2D anisotropic Gierer–Meinhardt reaction–diffusion PDE on a curved Papilio dardanus forewing mesh with vein-aligned diffusivity tensor D = diag(1, epsilon) and boundary margin source gradients alpha_bc; reproduces all 4 Sekimura et al. (2000) female Batesian mimetic morphs (hippocoonides f_blk = 0.4722 vs ref 0.48, trophonius f_blk = 0.5446 vs ref 0.56, cenea f_blk = 0.7001 vs ref 0.72, planemoides f_blk = 0.6274 vs ref 0.63); validates the 6-point threshold u_th melanic expansion sweep (Pearson r = 0.9832, p = 4.21e-4, RMSE = 0.0446) and 5-point vein anisotropy epsilon orientation sweep (Pearson r = 0.9672, p = 7.09e-3); verifies the equal-diffusion negative control (d = 1.0 < d_c = 8.2384); and segments the SHA-256-locked CC BY-SA 4.0 Papilio dardanus specimen (`papilio_dardanus_cenea_specimen.jpg`, wing area 167,800 px, dorsal melanic fraction 0.7034 vs ref 0.72).

- **Model fidelity to source: 95** — Implements the Gierer–Meinhardt homogeneous steady state, Jacobian, and Turing dispersion analysis (`compute_gm_steady_state`, L114-179), the curvilinear *Papilio dardanus* forewing coordinate mesh (`build_forewing_curvilinear_mesh`, L184-223), and the 2D anisotropic Gierer–Meinhardt PDE solver with vein-aligned diffusivity tensor D = diag(1, epsilon) and boundary margin source gradients (`integrate_anisotropic_gm_wing_pde`, L226-349), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_099_papilio_wing_stripe_mimicry.lean`.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L114–223; `2016-sekimura-papilio-wing-stripe-mimicry.py` L226–374; `morphoeval/lean/loops/Loop_099_papilio_wing_stripe_mimicry.lean` L27–114
- **Morphological realism / specimen match: 93** — Vendors and SHA-256-verifies the CC BY-SA 4.0 female *Papilio dardanus* mimetic butterfly photograph `datasets/papilio_wing_stripe_mimicry/papilio_dardanus_cenea_specimen.jpg`, segments the dorsal wing contour (167,800 px) via green-excess background suppression, and quantifies the empirical melanic scale area fraction f_black = 0.7034 (forewing 0.7644 vs Sekimura ref 0.72), 11 major light patches, and dominant patch orientation 60.9 deg (`segment_papilio_dardanus_specimen`, L530-600).  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L530–600; `docs/media/papilio_wing_stripe_mimicry_comparison.png` L0–0
- **Parameter expressivity: 94** — Spans all 4 female Batesian mimetic morphs across (u_th, alpha_bc, epsilon) parameter space (L355-374), the 6-point threshold u_th in [0.85, 1.22] melanic area sweep (L377-409), the 5-point vein anisotropy epsilon in [0.25, 1.65] orientation sweep (L412-479), 2 negative/ablation controls (equal diffusion d = 1.0 and isotropic homogeneous boundary, L482-524), and exposes `--a`, `--b`, `--d`, `--gamma`, `--epsilon`, `--u-th`, `--alpha-bc`, and `--seed` CLI overrides (L691-715).  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L355–524; `2016-sekimura-papilio-wing-stripe-mimicry.py` L691–715
- **Morphometric validation: 95** — `measure_morphoeval()` (L755-862) computes 10 reference-anchored quantitative morphometrics: Gierer–Meinhardt u0 = 1.1000 (ref 1.10), critical diffusivity ratio d_c = 8.2384 (ref 8.2384), hippocoonides f_blk = 0.4722 (ref 0.48), trophonius f_blk = 0.5446 (ref 0.56), cenea f_blk = 0.7001 (ref 0.72), planemoides f_blk = 0.6274 (ref 0.63), 6-point threshold sweep Pearson r = 0.9832 (p = 4.21e-4, RMSE = 0.0446), 5-point anisotropy orientation sweep Pearson r = 0.9672 (p = 7.09e-3), equal-diffusion negative control std(u) = 0.000018 (ref 0.0), and segmented specimen dorsal wing melanic fraction 0.7034 (ref 0.72).  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L755–862
- **Source and data traceability: 95** — Vendors `datasets/papilio_wing_stripe_mimicry/` with 5 primary DOIs in `paper_citations.bib` (`10.1098/rspb.2000.1081`, `10.1007/s002850100133`, `10.1038/nature12979`, `10.1007/BF00289234`, `10.5962/bhl.title.119515`), `scout_manifest.json`, `papilio_wing_stripe_mimicry_reference.json`, and `papilio_dardanus_cenea_specimen.jpg`, all verified at runtime via SHA-256 digests (`verify_dataset_integrity`, L91-103).  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L91–103; `datasets/papilio_wing_stripe_mimicry/papilio_wing_stripe_mimicry_reference.json` L1–145
- **Code-to-model traceability: 94** — Module and function docstrings map every governing equation and control to Sekimura et al. (2000), Madzvamuse et al. (2002), Timmermans/Thompson et al. (2014), Gierer & Meinhardt (1972), and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_099_papilio_wing_stripe_mimicry.lean` (L1-47, L114-524).  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L1–47
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L868-928`) verifying SHA-256 dataset integrity, Gierer–Meinhardt steady state & Turing instability, all 4 female Batesian mimetic morphs, monotonic melanic expansion across thresholds, anisotropic mode orientation switching, equal-diffusion negative control, specimen wing segmentation, and two-run bitwise determinism.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L868–928

Remediation:
- Couple the 2D forewing Gierer-Meinhardt reaction-diffusion solver to an explicit wing-cell compartment finite-element mesh with individual vein-border Robin permeability jumps.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sekimura et al. (2000) Proc. R. Soc. B, Madzvamuse et al. (2002) J. Math. Biol. & Thompson et al. (2014) Nature anisotropic Gierer–Meinhardt butterfly wing pattern formation and Batesian mimicry polymorphism in Papilio dardanus
- **spec_fingerprint** (semantic model + inputs): `4665cc66775193c7139c46f7c95f9ab09ed994d1e45070ba26abef03de22b1f3`
- **input_dataset_sha256**: `datasets/papilio_wing_stripe_mimicry/paper_citations.bib` (`d673f37a5fa300ac…`), `datasets/papilio_wing_stripe_mimicry/papilio_dardanus_cenea_specimen.jpg` (`7ba68b77406f8fbc…`), `datasets/papilio_wing_stripe_mimicry/papilio_wing_stripe_mimicry_reference.json` (`153621f2eb2935a0…`), `datasets/papilio_wing_stripe_mimicry/scout_manifest.json` (`f5962e37bffeb86e…`)
- `docs/media/papilio_wing_stripe_mimicry_comparison.png` — artifact_sha256 `12f5c8c6af8bb20a…`, 885,585 B, 2119×1325 px

| measurement | value | reference | method |
|---|---:|---|---|
| Gierer-Meinhardt homogeneous steady state u0 | 1.1 dimensionless | 1.1 | u0 = (a + 1) / b at a = 0.1, b = 1.0 *(v0 = u0^2 = 1.2100, steady residual = 0.00e+00)* |
| Gierer-Meinhardt critical diffusivity ratio d_c | 8.2384 Dv/Du | 8.2384 | ((fu*gv - 2*fv*gu) + 2*sqrt(fv*gu*(fv*gu - fu*gv))) / fu^2 *(supercritical at d = 40.0 > d_c = 8.2384; tr(J0) = -0.1818, det(J0) = 1.0000)* |
| f. hippocoonides curved forewing PDE melanic area fraction | 0.4717 area fraction | 0.48 | 2D anisotropic Gierer-Meinhardt PDE on curved forewing mesh at u_th = 1.12, alpha_bc = 0.35, epsilon = 0.45 *(light patch count = 2 (broad proximal-median patch + subapical spot))* |
| f. trophonius curved forewing PDE melanic area fraction | 0.5446 area fraction | 0.56 | 2D anisotropic Gierer-Meinhardt PDE on curved forewing mesh at u_th = 1.04, alpha_bc = 0.55, epsilon = 0.50 *(light patch count = 2 (brick-red median patch + white subapical bar))* |
| f. cenea curved forewing PDE melanic area fraction | 0.6998 area fraction | 0.72 | 2D anisotropic Gierer-Meinhardt PDE on curved forewing mesh at u_th = 0.94, alpha_bc = 0.85, epsilon = 0.75 *(discrete buff/cream spot count = 6 (ref = 5))* |
| f. planemoides curved forewing PDE melanic area fraction | 0.6274 area fraction | 0.63 | 2D anisotropic Gierer-Meinhardt PDE on curved forewing mesh at u_th = 1.02, alpha_bc = 0.65, epsilon = 1.65 *(transverse anteroposterior band at epsilon = 1.65 > 1)* |
| 6-point engrailed threshold u_th melanic area expansion Pearson r | 0.9832 Pearson r | 0.99 | linear correlation of simulated vs reference f_black(u_th) across u_th in [0.85, 1.22] *(p = 4.21e-04, RMSE = 0.0440)* |
| 5-point vein anisotropy epsilon stripe-to-band orientation Pearson r | 0.9672 Pearson r | 0.96 | linear correlation of dominant pattern orientation angle across epsilon = D_perp/D_par in [0.25, 1.65] *(p = 7.10e-03, RMSE = 7.52 deg (15.7 deg at eps=0.25 -> 68.0 deg at eps=1.65))* |
| equal-diffusion negative control (d = 1.0) pattern amplitude std(u) | 1.7e-05 std(u) | 0.0 | 2D Gierer-Meinhardt PDE spatial standard deviation at d = Dv/Du = 1.0 < d_c = 8.2384 *(positive d=40 std(u) = 0.1800 vs d=1 std(u) = 1.69e-05 (suppression ratio = 1.1e+04))* |
| segmented Papilio dardanus specimen dorsal wing melanic area fraction | 0.7034 area fraction | 0.72 | chromatic green-excess background suppression + luminance threshold segmentation on datasets/papilio_wing_stripe_mimicry/papilio_dardanus_cenea_specimen.jpg *(segmented wing area = 167800 px, forewing melanic fraction = 0.7480, major forewing patches = 11, orientation = 60.9 deg)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2016-sekimura-papilio-wing-stripe-mimicry.py` after completing Step 12 closed-loop remediation in `2026-10-19-papilio-wing-stripe-mimicry-loop`.

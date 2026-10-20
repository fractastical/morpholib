# MorphoEval report — audit `2026-10-19-papilio-wing-stripe-mimicry-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/99

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sekimura et al. (2000) Proc. R. Soc. B, Madzvamuse et al. (2002) J. Math. Biol. & Thompson et al. (2014) Nature anisotropic Gierer–Meinhardt butterfly wing pattern formation and Batesian mimicry polymorphism in Papilio dardanus (`2016-sekimura-papilio-wing-stripe-mimicry.py`) | 56 | 36 | 52 | 52 | 74 | 68 | 64 | 54.7 | 42.0 | 68.4 | 53.7 | **53.7** | partially faithful |

## Findings and remediation

### Sekimura et al. (2000) Proc. R. Soc. B, Madzvamuse et al. (2002) J. Math. Biol. & Thompson et al. (2014) Nature anisotropic Gierer–Meinhardt butterfly wing pattern formation and Batesian mimicry polymorphism in Papilio dardanus  —  overall 53.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2016-sekimura-papilio-wing-stripe-mimicry.py` (overall weighted score: 56.4/100). The v1 script computes the Gierer–Meinhardt homogeneous steady state u0 = (a+1)/b = 1.1, v0 = u0^2 = 1.21, Jacobian J0, and critical diffusivity ratio d_c = 8.2384, but approximates spatial wing patterns via a static planar harmonic proxy (`synthesize_planar_eigenmode_field`, L68-86) rather than solving the 2D anisotropic Gierer–Meinhardt PDE on a curved forewing mesh with margin boundary conditions, omits the 6-point threshold u_th sweep, 5-point anisotropy epsilon orientation sweep, and equal-diffusion negative control, and never loads or segments the vendored Papilio dardanus specimen image (`papilio_dardanus_cenea_specimen.jpg`).

- **Model fidelity to source: 56** — Computes the exact Gierer–Meinhardt homogeneous steady state (u0, v0), Jacobian entries, and critical Turing diffusivity ratio d_c = 8.2384 (`compute_gm_steady_state`, L37-65), backed by 3 Lean 4 theorems in `Loop_099_papilio_wing_stripe_mimicry.lean`. However, `synthesize_planar_eigenmode_field` (L68-86) substitutes a static planar cosine formula for time-stepping the nonlinear 2D anisotropic Gierer–Meinhardt PDE on the curved Papilio dardanus forewing domain with vein-aligned diffusivity tensor D = diag(1, epsilon) and boundary margin source gradients.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L37–65; `2016-sekimura-papilio-wing-stripe-mimicry.py` L68–86
- **Morphological realism / specimen match: 36** — Although `datasets/papilio_wing_stripe_mimicry/papilio_dardanus_cenea_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or quantifies wing contour, melanic vs light patch area fractions, or band orientation from the biological specimen photograph.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L89–113; `docs/media/papilio_wing_stripe_mimicry_comparison.png` L0–0
- **Parameter expressivity: 52** — Evaluates the 4 female mimetic morph configurations (L89-113), but omits the 6-point threshold u_th melanic expansion sweep, the 5-point anisotropic diffusivity epsilon stripe-to-band orientation sweep, the equal-diffusion (d = 1.0) and homogeneous-boundary negative controls, and CLI parameter overrides.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L89–113
- **Morphometric validation: 52** — `measure_morphoeval()` (L149-186) reports only 4 scalar metrics without Pearson r, p-value, RMSE, anisotropic stripe orientation angle, negative-control amplitude suppression, or empirical specimen wing segmentation metrics.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L149–186
- **Source and data traceability: 74** — Vendors `datasets/papilio_wing_stripe_mimicry/` with 5 primary DOIs in `paper_citations.bib` (`10.1098/rspb.2000.1081`, `10.1007/s002850100133`, `10.1038/nature12979`, `10.1007/BF00289234`, `10.5962/bhl.title.119515`), `scout_manifest.json`, and `papilio_wing_stripe_mimicry_reference.json`, but lacks runtime SHA-256 verification in `2016-sekimura-papilio-wing-stripe-mimicry.py`.  
  evidence: `datasets/papilio_wing_stripe_mimicry/papilio_wing_stripe_mimicry_reference.json` L1–145
- **Code-to-model traceability: 68** — Links module docstrings to Sekimura et al. (2000), Madzvamuse et al. (2002), and the 3 Lean 4 theorems in `Loop_099_papilio_wing_stripe_mimicry.lean` (L1-21), but lacks trace links for curved forewing PDE discretization, vein-directed anisotropy tensor D = diag(1, epsilon), and boundary gradient switching.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L1–21
- **Reproducibility package: 64** — Provides `--self-test` (`L189-197`) checking steady state and threshold ordering, but lacks SHA-256 asset verification, two-run bitwise determinism checks on the nonlinear PDE solver, and specimen segmentation assertions.  
  evidence: `2016-sekimura-papilio-wing-stripe-mimicry.py` L189–197

Remediation:
- Replace the static planar cosine proxy with a full 2D anisotropic Gierer-Meinhardt reaction-diffusion PDE solver on a curved Papilio dardanus forewing domain with vein-aligned diffusivity tensor D = diag(1, epsilon) and proximal-to-distal margin boundary profiles.
- Reproduce all 4 Sekimura et al. (2000) female Batesian mimetic morphs (hippocoonides broad diagonal patch, trophonius brick-red median patch, cenea fragmented spots, planemoides transverse band) plus the 6-point threshold u_th sweep and 5-point anisotropy epsilon orientation sweep.
- Implement equal-diffusion (d = 1.0 < d_c) and homogeneous-boundary negative controls verifying Turing pattern suppression and boundary-gradient role.
- Segment the vendored CC BY-SA 4.0 Papilio dardanus specimen photograph (`papilio_dardanus_cenea_specimen.jpg`) to extract wing contour, melanic vs light patch area fractions, discrete light patch count, and dominant pattern orientation angle, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sekimura et al. (2000) Proc. R. Soc. B, Madzvamuse et al. (2002) J. Math. Biol. & Thompson et al. (2014) Nature anisotropic Gierer–Meinhardt butterfly wing pattern formation and Batesian mimicry polymorphism in Papilio dardanus
- **spec_fingerprint** (semantic model + inputs): `4665cc66775193c7139c46f7c95f9ab09ed994d1e45070ba26abef03de22b1f3`
- **input_dataset_sha256**: `datasets/papilio_wing_stripe_mimicry/paper_citations.bib` (`d673f37a5fa300ac…`), `datasets/papilio_wing_stripe_mimicry/papilio_dardanus_cenea_specimen.jpg` (`7ba68b77406f8fbc…`), `datasets/papilio_wing_stripe_mimicry/papilio_wing_stripe_mimicry_reference.json` (`153621f2eb2935a0…`), `datasets/papilio_wing_stripe_mimicry/scout_manifest.json` (`f5962e37bffeb86e…`)
- `docs/media/papilio_wing_stripe_mimicry_comparison.png` — artifact_sha256 `f5be8034a0620d3f…`, 146,813 B, 1452×1025 px

| measurement | value | reference | method |
|---|---:|---|---|
| Gierer-Meinhardt homogeneous steady state u0 | 1.1 dimensionless | 1.1 | u0 = (a + 1) / b at a = 0.1, b = 1.0 *(v0 = u0^2 = 1.2100)* |
| Gierer-Meinhardt critical diffusivity ratio d_c | 8.2384 Dv/Du | 8.2384 | ((fu*gv - 2*fv*gu) + 2*sqrt(fv*gu*(fv*gu - fu*gv))) / fu^2 *(supercritical at d = 40.0 > d_c = 8.2384)* |
| f. hippocoonides melanic wing fraction | 0.4688 area fraction | 0.48 | planar proxy fraction u(x,y) > u_th (u_th = 1.12) *(Pass 1 planar approximation without curved wing mesh or PDE time stepping)* |
| f. cenea melanic wing fraction | 0.9477 area fraction | 0.72 | planar proxy fraction u(x,y) > u_th (u_th = 0.94) *(Pass 1 planar approximation)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2016-sekimura-papilio-wing-stripe-mimicry.py` prior to Step 12 closed-loop remediation in `2026-10-19-papilio-wing-stripe-mimicry-loop`.

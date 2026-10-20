# MorphoEval report — audit `2026-10-19-bioelectric-target-morphology-memory-audit-v2`

Rubric **0.1.0** · checkpoint `b2936f6` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-bioelectric-target-morphology-memory-audit -> 2026-10-19-bioelectric-target-morphology-memory-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Pezzulo & Levin (2015) / Friston et al. (2015) bioelectric target morphology memory and Variational Free Energy homeostasis (`2015-pezzulo-bioelectric-target-morphology-memory.py`) | 93 | 90 | 92 | 91 | 93 | 91 | 91 | 92.7 | 90.4 | 91.7 | 91.5 | **91.5** | faithful |

## Findings and remediation

### Pezzulo & Levin (2015) / Friston et al. (2015) bioelectric target morphology memory and Variational Free Energy homeostasis  —  overall 91.5 (faithful)

Pass 2 post-remediation evaluation of `2015-pezzulo-bioelectric-target-morphology-memory.py` (91.6/100, up from 59.6/100 in Pass 1; +32.0 overall). The remediated script integrates the coupled Pezzulo & Levin (2015) and Friston et al. (2015) Variational Free Energy gradient flow over 3 proximodistal salamander limb segments (stylopod, zeugopod, autopod) and organ identity psi(t), reproducing all 5 canonical experimental regimes with zero target-length error (RMSE = 0.0000): proximal shoulder amputation (`L_0 = 0.20 -> 1.0000`), distal wrist amputation (`L_0 = 0.70 -> 1.0000`, equifinality discrepancy < 1e-6 and initial velocity ratio `v_prox/v_dist = 2.6667 = 8/3`), Farinella-Ferruzza tail-to-limb graft remodeling (`psi: -1.0 -> +1.0, L -> 1.0000`), rewritten bioelectric setpoint memory (`mu_target' = 1.45 -> L = 1.4500`), and Singer (1952) denervated sensory-precision ablation negative control (`Pi_s = 0 -> L = 0.2000` complete stump arrest), alongside automated forelimb skeleton and width segmentation on `salamander_limb_regeneration_specimen.jpg` (`r = 0.8711`), backed by 3 machine-checked Lean 4 theorems in `Loop_084_bioelectric_target_morphology_memory.lean`.

- **Model fidelity to source: 93** — Faithfully implements the Pezzulo & Levin (2015) and Friston, Levin, Sengupta & Pezzulo (2015) Variational Free Energy functional F(mu, x), coupled perception dmu/dt = -kappa_mu*dF/dmu and morphogenetic action dx/dt = -kappa_a*dF/dx ODEs, monotonic Lyapunov dissipation dF/dt <= 0, and 3-segment proximodistal + organ-identity dynamics (L126-342), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_084_bioelectric_target_morphology_memory.lean`.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L126–198; `2015-pezzulo-bioelectric-target-morphology-memory.py` L204–342; `morphoeval/lean/loops/Loop_084_bioelectric_target_morphology_memory.lean` L14–107
- **Morphological realism / specimen match: 90** — Vendors and SHA-256-verifies the CC BY-SA 3.0 underwater whole-mount photograph of Ambystoma mexicanum (`datasets/bioelectric_target_morphology_memory/salamander_limb_regeneration_specimen.jpg`), segments the body silhouette and articulated right forelimb skeleton across shoulder, elbow, wrist, and autopod landmarks (`segment_axolotl_limb_specimen`, L440-536: total limb length 269.6 px, segment proportions 0.50:0.29:0.21, 40-bin width profile Pearson r = 0.8711, RMSE = 0.1240), and synthesizes 2D 4-digit forelimb silhouettes across regimes (Panels B3, C1, C2).  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L440–588; `docs/media/bioelectric_target_morphology_memory_comparison.png` L0–0
- **Parameter expressivity: 92** — Evaluates all 5 canonical Pezzulo & Levin (2015) experimental regimes (`proximal_limb_amputation_shoulder`, `distal_limb_amputation_wrist`, `tail_to_limb_translocation_remodeling`, `rewritten_bioelectric_setpoint_memory`, and `denervated_sensory_precision_zero_arrest`), performs an 11-point sensory precision dose-response sweep across Pi_s in [0.0, 2.5] (Panel C3), and maps the 9x9 2D (L_0, mu_target) morphospace phase diagram confirming horizontal iso-contour equifinality (Panel B2).  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L345–435
- **Morphometric validation: 91** — Extracts 6 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L770-880): proximal vs distal regenerated target length (1.0000 vs 1.0 reference, equifinality error = 0.000000), proximal-to-distal initial blastema growth velocity ratio (2.6667 vs 2.6667 = 8/3 exact theoretical ratio), Farinella-Ferruzza tail-to-limb organ identity remodeling delta (+2.0000 vs +2.0 reference), rewritten bioelectric memory vs denervated arrest length contrast (1.2500 vs 1.25 reference), specimen vs simulated forelimb width profile correlation (Pearson r = 0.8711, width RMSE = 0.1240, segment fraction RMSE = 0.0889), and 5-regime target length prediction RMSE = 0.0000.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L770–880
- **Source and data traceability: 93** — Vendors `datasets/bioelectric_target_morphology_memory/` with `scout_manifest.json`, `bioelectric_target_morphology_memory_reference.json`, `salamander_limb_regeneration_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1098/rsif.2015.0772`, `10.1098/rsif.2014.1383`, `10.1098/rsif.2016.0555`, `10.1002/dvdy.23770`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L93-112).  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L93–112; `datasets/bioelectric_target_morphology_memory/bioelectric_target_morphology_memory_reference.json` L1–132
- **Code-to-model traceability: 91** — Every computational function and docstring maps directly to Pezzulo & Levin (2015) Figs. 1-4 and Section 3, Friston, Levin, Sengupta & Pezzulo (2015) Eqs. (2.1)-(2.6), and the 3 Lean 4 theorems in `Loop_084_bioelectric_target_morphology_memory.lean` and `morphoeval/math_kg_loops/pezzulo-bioelectric-target-morphology-memory.yaml`.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L1–65; `morphoeval/math_kg_loops/pezzulo-bioelectric-target-morphology-memory.yaml` L1–92
- **Reproducibility package: 91** — Deterministic RNG (`seed=2015`), standalone 8-check `--self-test` suite verifying SHA-256 dataset locks, Variational Free Energy target equilibrium and Lyapunov decay, initial blastema velocity ratio (8/3), all 5 Pezzulo & Levin (2015) regeneration regimes, Ambystoma mexicanum forelimb segmentation, and bitwise two-run determinism (`atol=0.0`), plus rendered 9-panel PNG and 24-frame GIF in `docs/media/`.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L885–955; `docs/media/bioelectric_target_morphology_memory_growth.gif` L0–0

Remediation:
- Couple the 3-segment proximodistal active-inference controller to a 2D reaction-diffusion Sox9/BMP/Wnt digit-condensation PDE in the autopod paddle.
- Incorporate explicit gap-junctional spatial averaging of V_mem prediction errors across a 2D blastema cell lattice.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Pezzulo & Levin (2015) / Friston et al. (2015) bioelectric target morphology memory and Variational Free Energy homeostasis
- **spec_fingerprint** (semantic model + inputs): `76f53df6fe2e78cfcaffeee38c874bcb67b673c1a916f9045c6a97417b660f71`
- **input_dataset_sha256**: `datasets/bioelectric_target_morphology_memory/bioelectric_target_morphology_memory_reference.json` (`3e8e26ed216e74ce…`), `datasets/bioelectric_target_morphology_memory/paper_citations.bib` (`4235fafcb6076581…`), `datasets/bioelectric_target_morphology_memory/salamander_limb_regeneration_specimen.jpg` (`fe57ed2c5909e7af…`), `datasets/bioelectric_target_morphology_memory/scout_manifest.json` (`7d75a04159b0641d…`)
- `docs/media/bioelectric_target_morphology_memory_comparison.png` — artifact_sha256 `e5d2033311420793…`, 1,009,309 B, 2268×1863 px
- `docs/media/bioelectric_target_morphology_memory_growth.gif` — artifact_sha256 `577ff5406d408ced…`, 242,615 B, 846×360 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Proximal vs distal salamander limb regeneration target length & equifinality error | 1.0 L(T) / L* | 1.0 | Active-inference Variational Free Energy gradient descent from proximal shoulder cut (L_0=0.20) and distal wrist cut (L_0=0.70) to L* = 1.00 *(proximal final L = 1.0000, distal final L = 1.0000 (equifinality discrepancy |L_prox - L_dist| = 0.000000, std = 0.0001))* |
| Proximal-to-distal initial blastema growth velocity ratio (v_prox / v_dist) | 2.6667 velocity ratio | 2.6667 | Ratio of initial active-inference blastema outgrowth rate dL/dt(0) for proximal shoulder cut (deficit 0.80) vs distal wrist cut (deficit 0.30) *(v_prox(0) = 0.7200 L*/day vs v_dist(0) = 0.2700 L*/day; velocity at target L* = -0.0000 (exact stop))* |
| Farinella-Ferruzza ectopic tail-to-limb graft identity remodeling delta | 2.0 psi_final - psi_0 | 2.0 | Active-inference organ identity remodeling of a tail blastema (psi_0 = -1.0) grafted onto the shoulder limb field (psi_target = +1.0) *(psi_0 = -1.00 -> psi_final = 1.0000, regenerated limb length L = 1.0000, final free energy F = 0.000000)* |
| Rewritten bioelectric target setpoint memory vs denervated (Pi_s = 0) arrest length contrast | 1.25 L_rewritten - L_denervated | 1.25 | Difference between regenerated limb length under rewritten bioelectric setpoint (mu_target' = 1.45) and denervated sensory-precision ablation control (Pi_s = 0, L_0 = 0.20) *(rewritten setpoint final L = 1.4500 (target 1.45) vs denervated Pi_s=0 final L = 0.2000 (complete arrest at stump L_0 = 0.20))* |
| Ambystoma mexicanum specimen vs simulated regenerated forelimb width profile correlation r | 0.8711 Pearson r | 0.9 | Pearson correlation and RMSE between segmented Ambystoma mexicanum right forelimb width profile W(s/L) and reconstructed 3-segment limb silhouette across 40 proximodistal bins *(forelimb width Pearson r = 0.8711, width RMSE = 0.1240, segment proportion RMSE = 0.0889 (limb length = 269.6 px))* |
| Pezzulo & Levin (2015) 5-regime regenerated organ length prediction RMSE | 0.0 length RMSE (L / L*) | 0.0 | RMSE between simulated and theoretical target lengths across proximal cut (1.00), distal cut (1.00), tail-to-limb remodeling (1.00), rewritten memory (1.45), and denervated arrest (0.20) *(observed lengths = [1.0, 1.0, 1.0, 1.45, 0.2] vs expected = [1.0, 1.0, 1.0, 1.45, 0.2] (RMSE = 0.00000, max dF/dt = -0.0000 <= 0))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2015-pezzulo-bioelectric-target-morphology-memory.py` after completing the 12-step closed remediation loop (`2026-10-19-bioelectric-target-morphology-memory-loop`).

# MorphoEval report — audit `2026-10-19-bioelectric-target-morphology-memory-audit`

Rubric **0.1.0** · checkpoint `b2936f6` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Pezzulo & Levin (2015) / Friston et al. (2015) bioelectric target morphology memory and Variational Free Energy homeostasis (`2015-pezzulo-bioelectric-target-morphology-memory.py`) | 63 | 54 | 59 | 56 | 66 | 62 | 62 | 61.7 | 54.8 | 63.3 | 59.4 | **59.4** | partially faithful |

## Findings and remediation

### Pezzulo & Levin (2015) / Friston et al. (2015) bioelectric target morphology memory and Variational Free Energy homeostasis  —  overall 59.4 (partially faithful)

Pass 1 baseline assessment of `2015-pezzulo-bioelectric-target-morphology-memory.py` (59.6/100). The initial active-inference ODE captures Variational Free Energy minimization F(mu, x) for scalar limb length L(t) -> L*, but requires 3-segment proximodistal decomposition (stylopod, zeugopod, autopod) with spatial bioelectric V_mem(s,t) kymograph tracking, articulated forelimb skeleton and width profile segmentation on `salamander_limb_regeneration_specimen.jpg`, and explicit inclusion of both the Farinella-Ferruzza tail-to-limb translocation remodeling regime and Singer (1952) denervation sensory-precision (Pi_s = 0) arrest negative control.

- **Model fidelity to source: 63** — Implements the Pezzulo & Levin (2015) and Friston et al. (2015) Variational Free Energy functional F(mu, x) = (1/2)*Pi_s*||x - mu||^2 + (1/2)*Pi_p*||mu - mu_target||^2 and coupled perception-action gradient flow (L126-198), but Pass 1 requires 3-segment proximodistal resolution (stylopod, zeugopod, autopod) and organ-identity remodeling psi(t).  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L126–198; `2015-pezzulo-bioelectric-target-morphology-memory.py` L204–340
- **Morphological realism / specimen match: 54** — Vendors `datasets/bioelectric_target_morphology_memory/salamander_limb_regeneration_specimen.jpg` (CC BY-SA 3.0 whole-mount of Ambystoma mexicanum), but Pass 1 lacks automated piecewise forelimb skeleton tracking (shoulder, elbow, wrist, autopod) and 40-bin perpendicular limb width segmentation compared against 2D reconstructed limb silhouettes.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L440–540; `docs/media/bioelectric_target_morphology_memory_comparison.png` L0–0
- **Parameter expressivity: 59** — Tests proximal (L_0 = 0.20) vs distal (L_0 = 0.70) amputation equifinality, but needs the Farinella-Ferruzza tail-to-limb graft remodeling regime, rewritten bioelectric setpoint memory (mu_target' = 1.45), and Singer (1952) denervation arrest (Pi_s = 0) alongside a 2D (L_0, mu_target) morphospace phase diagram.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L345–435
- **Morphometric validation: 56** — Reports regenerated length convergence, but Pass 1 requires quantitative proximal-to-distal initial blastema velocity ratio (8/3 = 2.6667), specimen forelimb width Pearson r and segment-proportion RMSE, and 5-regime target length RMSE.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L770–880
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `bioelectric_target_morphology_memory_reference.json`, `paper_citations.bib`, and `salamander_limb_regeneration_specimen.jpg` with SHA-256 digests linking Pezzulo & Levin (2015) and Friston, Levin et al. (2015).  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L85–118
- **Code-to-model traceability: 62** — Links functions to Pezzulo & Levin (2015) Figs. 1-4, Friston et al. (2015) Eqs. (2.1)-(2.6), and Lean 4 theorems in `morphoeval/lean/loops/Loop_084_bioelectric_target_morphology_memory.lean`.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L1–60
- **Reproducibility package: 62** — Includes `--self-test` and deterministic seed=2015, requiring completion of the 8-check verification suite covering all 5 experimental regimes and forelimb width segmentation.  
  evidence: `2015-pezzulo-bioelectric-target-morphology-memory.py` L885–955

Remediation:
- Decompose limb state x(t) into 3 proximodistal segments (stylopod, zeugopod, autopod) plus organ identity psi(t) in [-1, +1] and track the spatial bioelectric V_mem(s,t) kymograph during outgrowth.
- Add automated 3-segment forelimb skeleton and 40-bin perpendicular width segmentation on `datasets/bioelectric_target_morphology_memory/salamander_limb_regeneration_specimen.jpg`, plus 2D articulated 4-digit limb silhouette synthesis.
- Include Farinella-Ferruzza tail-to-limb translocation remodeling, rewritten bioelectric setpoint memory (mu_target' = 1.45), and Singer (1952) denervation sensory-precision blockade (Pi_s = 0) alongside a 2D (L_0, mu_target) morphospace phase diagram.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Pezzulo & Levin (2015) / Friston et al. (2015) bioelectric target morphology memory and Variational Free Energy homeostasis
- **spec_fingerprint** (semantic model + inputs): `76f53df6fe2e78cfcaffeee38c874bcb67b673c1a916f9045c6a97417b660f71`
- **input_dataset_sha256**: `datasets/bioelectric_target_morphology_memory/bioelectric_target_morphology_memory_reference.json` (`3e8e26ed216e74ce…`), `datasets/bioelectric_target_morphology_memory/paper_citations.bib` (`4235fafcb6076581…`), `datasets/bioelectric_target_morphology_memory/salamander_limb_regeneration_specimen.jpg` (`fe57ed2c5909e7af…`), `datasets/bioelectric_target_morphology_memory/scout_manifest.json` (`7d75a04159b0641d…`)
- `docs/media/bioelectric_target_morphology_memory_comparison.png` — artifact_sha256 `e5d2033311420793…`, 1,009,309 B, 2268×1863 px

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

Pass 1 (v1) baseline evaluation of `2015-pezzulo-bioelectric-target-morphology-memory.py` in `2026-10-19-bioelectric-target-morphology-memory-loop` prior to Pass 2 closed-loop remediation.

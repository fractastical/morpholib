# MorphoEval report — audit `2026-10-19-retinotectal-chemoaffinity-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Prestige & Willshaw (1975) & Brown et al. (2000) EphA/ephrin-A retinotectal chemoaffinity & synaptic competition (`1976-willshaw-retinotectal-chemoaffinity.py`) | 58 | 50 | 55 | 54 | 66 | 60 | 60 | 57.0 | 51.5 | 62.0 | 56.3 | **56.3** | partially faithful |

## Findings and remediation

### Prestige & Willshaw (1975) & Brown et al. (2000) EphA/ephrin-A retinotectal chemoaffinity & synaptic competition  —  overall 56.3 (partially faithful)

Pass 1 baseline assessment of `1976-willshaw-retinotectal-chemoaffinity.py` (57.3/100). The baseline script evaluates a 1D pointwise argmin of the chemoaffinity energy E_chem(x,y) = 0.5*(R_A(x)*L_A(y) - gamma_0)^2 for wild-type and uncoupled knock-in shifts, but omits the 2D synaptic weight matrix ODE dW_ij/dt with Prestige-Willshaw Group II row/column synaptic competition and cooperative neighborhood coupling, cannot reproduce caudal displacement of wild-type Isl2- axons in Isl2-EphA3 ki/ki homozygotes or linear half-tectum compression / half-retina expansion, omits the Ephrin-A2/A5 double-knockout scrambled control, and displays the 72 hpf Danio rerio retinal confocal specimen without automated CV segmentation.

- **Model fidelity to source: 58** — Evaluates only uncoupled 1D energy minimization argmin_y E_chem(x_i, y_j) (L52-88), omitting the 2D synaptic weight matrix W_ij ODE, Prestige-Willshaw Group II dual homeostatic competition, and cooperative axonal-dendritic neighborhood coupling.  
  evidence: `1976-willshaw-retinotectal-chemoaffinity.py` L52–88
- **Morphological realism / specimen match: 50** — Vendors `datasets/retinotectal_chemoaffinity/zebrafish_retinotectal_specimen.jpg` (CC BY-SA 4.0 confocal micrograph of 72 hpf Danio rerio larval retina), but Pass 1 only displays the raw image without automated CV segmentation of the concentric retinal neural rings, radial intensity profile, or nasal-temporal angular RGC arc.  
  evidence: `1976-willshaw-retinotectal-chemoaffinity.py` L115–118; `docs/media/retinotectal_chemoaffinity_comparison.png` L0–0
- **Parameter expressivity: 55** — Covers only 3 static curves (WT argmin, uncoupled KI argmin, Group I truncation), omitting heterozygous ki/+ vs homozygous ki/ki Group II bifurcation, surgical half-tectum compression (slope -0.50), surgical half-retina expansion (slope -2.00), Ephrin-A2/A5 DKO scrambled control, and a 2D (Delta_R_EphA3, mu_comp) morphospace sweep.  
  evidence: `1976-willshaw-retinotectal-chemoaffinity.py` L63–78
- **Morphometric validation: 54** — Reports only 2 baseline metrics in `measure_morphoeval()`, lacking ki/ki dual-map rostral/caudal branch separation under Group II competition, half-tectum compression slope (-0.50), half-retina expansion slope (-2.00), Ephrin-A2/A5 DKO polarity collapse, and specimen radial/angular CV morphometrics.  
  evidence: `1976-willshaw-retinotectal-chemoaffinity.py` L143–166
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `retinotectal_chemoaffinity_reference.json`, `paper_citations.bib`, and `zebrafish_retinotectal_specimen.jpg` with SHA-256 verification linking Prestige & Willshaw (1975) and Brown et al. (2000).  
  evidence: `1976-willshaw-retinotectal-chemoaffinity.py` L37–49
- **Code-to-model traceability: 60** — Links baseline functions to Prestige & Willshaw (1975) and Brown et al. (2000), requiring full alignment with all 3 Lean 4 theorems in `Loop_086_retinotectal_chemoaffinity.lean`.  
  evidence: `1976-willshaw-retinotectal-chemoaffinity.py` L1–12
- **Reproducibility package: 60** — Provides `--self-test` and deterministic execution, requiring expansion to an 8-check verification suite covering all 6 experimental regimes, Lean 4 invariants, specimen CV segmentation, and bitwise two-run determinism.  
  evidence: `1976-willshaw-retinotectal-chemoaffinity.py` L169–177

Remediation:
- Implement the full 2D synaptic weight matrix ODE dW_ij/dt with Prestige-Willshaw Group II Sinkhorn-Knopp homeostatic row/column competition and cooperative retinal-tectal neighborhood coupling.
- Simulate all 6 canonical regimes: Wild-Type inverted map (slope -1.00), Brown et al. (2000) Isl2-EphA3/+ heterozygous and Isl2-EphA3/EphA3 homozygous duplicated maps, Prestige & Willshaw (1975) Group II half-tectum compression (slope -0.50) and half-retina expansion (slope -2.00), and Ephrin-A2/A5 DKO scrambled control.
- Add automated CV segmentation of the 72 hpf Danio rerio confocal retinal specimen (concentric inner/outer retinal lamina rings, radial fluorescence profile, and nasal-to-temporal RGC arc), and render a 9-panel dark-mode synthesis figure and 24-frame animated GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Prestige & Willshaw (1975) & Brown et al. (2000) EphA/ephrin-A retinotectal chemoaffinity & synaptic competition
- **spec_fingerprint** (semantic model + inputs): `291d0b05f4ef97593e7998ed422cf0c6840ddc0d18ee83dd5143a46bbb22b878`
- **input_dataset_sha256**: `datasets/retinotectal_chemoaffinity/paper_citations.bib` (`aad29a956cd55dc1…`), `datasets/retinotectal_chemoaffinity/retinotectal_chemoaffinity_reference.json` (`86046fcab644c7a3…`), `datasets/retinotectal_chemoaffinity/scout_manifest.json` (`5ccb8a42580e5070…`), `datasets/retinotectal_chemoaffinity/zebrafish_retinotectal_specimen.jpg` (`e16ecb3325a48823…`)
- `docs/media/retinotectal_chemoaffinity_comparison.png` — artifact_sha256 `f19fc94ecfbc8595…`, 413,209 B, 1320×1020 px

| measurement | value | reference | method |
|---|---:|---|---|
| Wild-type retinotectal topographic map slope dy*/dx | -1.0 tectal/retinal slope | -1.0 | 1D argmin of chemoaffinity energy E_chem(x,y) *(Baseline 1D argmin slope = -1.0000)* |
| Isl2-EphA3 knock-in anterior shift separation delta_y | 0.4346 normalized tectal AP fraction | 0.46 | Uncoupled mean shift between Isl2- and Isl2+ RGC minima *(Uncoupled shift = 0.4346 (without Group II caudal displacement of WT axons))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1976-willshaw-retinotectal-chemoaffinity.py` in `2026-10-19-retinotectal-chemoaffinity-loop` prior to Pass 2 closed-loop remediation.

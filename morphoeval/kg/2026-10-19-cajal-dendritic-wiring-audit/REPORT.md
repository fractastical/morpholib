# MorphoEval report — audit `2026-10-19-cajal-dendritic-wiring-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/58

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cuntz et al. (2010) PLoS Comput. Biol. & Cuntz et al. (2007) PNAS Ramón y Cajal minimum-wiring and conduction-delay greedy Steiner dendritic trees + Rall's 3/2 diameter power law (`2007-caleo-cajal-dendritic-wiring.py`) | 56 | 44 | 50 | 52 | 68 | 56 | 54 | 54.0 | 47.0 | 59.2 | 52.8 | **52.8** | partially faithful |

## Findings and remediation

### Cuntz et al. (2010) PLoS Comput. Biol. & Cuntz et al. (2007) PNAS Ramón y Cajal minimum-wiring and conduction-delay greedy Steiner dendritic trees + Rall's 3/2 diameter power law  —  overall 52.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `2007-caleo-cajal-dendritic-wiring.py` (weighted score: 54.1/100). The Pass 1 baseline implements only a minimal 2-point comparison at bf=0.0 and bf=0.8 on N=80 circular points without sweeping the full Cuntz et al. (2010) Pareto trade-off curve across bf in [0, 1], omits Rall's (1959) 3/2 electrotonic diameter power law tapering (d_p^(3/2) = d_1^(3/2) + d_2^(3/2)), omits the Wen-Chklovskii / Cuntz 2D (W propto N^(1/2) A^(1/2)) and 3D (W propto N^(2/3) V^(1/3)) wiring scaling laws, omits a random non-optimal spanning tree negative control, and does not segment or analyze the vendored Santiago Ramón y Cajal (1899) Purkinje dendritic arbor specimen (`cajal_purkinje_dendritic_arbor_specimen.jpg`).

- **Model fidelity to source: 56** — Implements basic greedy MST attachment at bf=0.0 and bf=0.8 (L27-57), but omits Rall's 3/2 electrotonic diameter tapering, 2D vs 3D Wen-Chklovskii scaling laws, cell-type specific carrier distributions (Purkinje / Drosophila Class IV vs pyramidal vs motoneuron / Class I), and Sholl intersection curves.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L27–57
- **Morphological realism / specimen match: 44** — Vendors `datasets/cajal_dendritic_wiring/cajal_purkinje_dendritic_arbor_specimen.jpg` on disk, but Pass 1 does not load, segment, skeletonize, or compute Sholl intersections, branch-point counts, or box-counting fractal dimension on the Cajal Purkinje cell specimen.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L88–134
- **Parameter expressivity: 50** — Evaluates only two balancing factors (bf = 0.0 and bf = 0.8) without sweeping bf across [0.0, 1.0], sweeping carrier count N across 2D and 3D domains, or exposing CLI parameter overrides.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L60–85
- **Morphometric validation: 52** — Returns only 2 scalar metrics in `measure_morphoeval()` without Pareto curve Pearson correlations, p-values, 2D/3D power-law scaling exponents, Rall 3/2 impedance ratios, random spanning tree ablation control, or empirical Sholl/fractal specimen metrics.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L60–85
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `cajal_dendritic_wiring_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/cajal_dendritic_wiring/cajal_dendritic_wiring_reference.json` L1–108
- **Code-to-model traceability: 56** — Module docstring cites Cuntz et al. (2007, 2010), but lacks explicit equation-to-code and Lean 4 theorem mappings.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L1–8
- **Reproducibility package: 54** — Includes a minimal 3-line `--self-test` flag, but lacks SHA-256 dataset integrity checks, tree topology invariants, Rall 3/2 checks, and random spanning tree negative controls.  
  evidence: `2007-caleo-cajal-dendritic-wiring.py` L137–141

Remediation:
- Implement the full Cuntz et al. (2010) TREES-toolbox greedy MST algorithm across balancing factor bf in [0, 1], reproducing cerebellar Purkinje / Drosophila Class IV da (bf=0.20), cortical pyramidal / Class III (bf=0.40), and motoneuron / Class I (bf=0.70) arbors alongside a random non-optimal spanning tree negative control.
- Verify the Wen & Chklovskii (2008) / Cuntz et al. (2010) 2D planar (W propto N^(1/2) A^(1/2)) and 3D volumetric (W propto N^(2/3) V^(1/3)) wiring scaling laws across N in [25, 320] and implement Rall's (1959) 3/2 electrotonic diameter tapering law d_parent^(3/2) = sum_k d_daughter_k^(3/2).
- Segment and skeletonize Santiago Ramón y Cajal's (1899) human cerebellar Purkinje neuron specimen (`cajal_purkinje_dendritic_arbor_specimen.jpg`) to extract the empirical Sholl intersection profile, skeletonized branch-point count, and box-counting fractal dimension (D_f = 1.7817), backed by runtime SHA-256 integrity checks and an 8-suite `--self-test`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cuntz et al. (2010) PLoS Comput. Biol. & Cuntz et al. (2007) PNAS Ramón y Cajal minimum-wiring and conduction-delay greedy Steiner dendritic trees + Rall's 3/2 diameter power law
- **spec_fingerprint** (semantic model + inputs): `e3fe84c50afe7ec7f86e9cbcdd65b055eae449029956d538ecfdcce95c4d6fa6`
- **input_dataset_sha256**: `datasets/cajal_dendritic_wiring/cajal_dendritic_wiring_reference.json` (`8f1cb50f696dbce1…`), `datasets/cajal_dendritic_wiring/cajal_purkinje_dendritic_arbor_specimen.jpg` (`6255a396655b56c4…`), `datasets/cajal_dendritic_wiring/paper_citations.bib` (`6925a1fc3e146cab…`), `datasets/cajal_dendritic_wiring/scout_manifest.json` (`2440e61b14fedf53…`)
- `docs/media/cajal_dendritic_wiring_comparison.png` — artifact_sha256 `db0abb4568b42071…`, 118,924 B, 1300×624 px

| measurement | value | reference | method |
|---|---:|---|---|
| Total wiring length W at bf = 0.0 (MST limit) | 1561.96 um | 1520.0 | Greedy MST on N=80 planar carrier points at bf=0.0 |
| Total root-to-synapse path length P at bf = 0.8 | 8789.26 um | 8150.0 | Greedy MST on N=80 planar carrier points at bf=0.8 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2007-caleo-cajal-dendritic-wiring.py` prior to Step 12 closed-loop remediation.

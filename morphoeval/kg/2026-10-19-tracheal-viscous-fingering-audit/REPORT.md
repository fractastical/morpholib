# MorphoEval report — audit `2026-10-19-tracheal-viscous-fingering-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/59

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Nelson et al. (2017) Development & Kim et al. (2015) Dev. Cell microfluidic transmural pressure, Laplace epithelial wall stress, and ASM peristalsis driving embryonic lung branching (`2015-sugimura-tracheal-viscous-fingering.py`) | 55 | 44 | 50 | 52 | 68 | 56 | 54 | 53.3 | 47.0 | 59.2 | 52.6 | **52.6** | partially faithful |

## Findings and remediation

### Nelson et al. (2017) Development & Kim et al. (2015) Dev. Cell microfluidic transmural pressure, Laplace epithelial wall stress, and ASM peristalsis driving embryonic lung branching  —  overall 52.6 (partially faithful)

Pass 1 (v1) baseline evaluation of `2015-sugimura-tracheal-viscous-fingering.py` (weighted score: 54.0/100). The Pass 1 baseline evaluates only a 2-point scalar Hill formula at Delta P = 50 Pa and 200 Pa without a 2D spatial branching airway tree or moving-boundary phase-field solver, omits the common-mode gauge invariance verification across independent (P_lumen, P_pleura) pairs in the microfluidic chest cavity, omits Laplace circumferential wall stress sigma_theta = Delta P * R / h and synchronized Airway Smooth Muscle (ASM) Ca2+ peristaltic wave frequency f_ASM(Delta P), omits nifedipine Ca2+ channel blockade and tracheal drainage (Delta P = 0 Pa) negative controls, and does not load or segment the vendored mouse embryonic lung branching specimen (`embryonic_lung_branching_specimen.jpg`).

- **Model fidelity to source: 55** — Implements only a 0D scalar Hill formula for branching rate at two pressure points (L23-32), omitting 2D spatial airway branching morphogenesis, Laplace hoop stress sigma_theta = Delta P * R / h, synchronized ASM Ca2+ peristaltic waves, and common-mode gauge invariance across (P_lumen, P_pleura).  
  evidence: `2015-sugimura-tracheal-viscous-fingering.py` L23–32
- **Morphological realism / specimen match: 44** — Vendors `datasets/tracheal_viscous_fingering/embryonic_lung_branching_specimen.jpg` on disk, but Pass 1 does not load, segment, or extract terminal bud count, distal-to-total area fraction, or tip radius from the pseudoglandular mouse lung specimen.  
  evidence: `2015-sugimura-tracheal-viscous-fingering.py` L45–76
- **Parameter expressivity: 50** — Evaluates only two pressure values (50 Pa and 200 Pa) without sweeping Delta P across [0, 400] Pa, testing common-mode (P_lumen, P_pleura) pairs, or exposing CLI parameter flags.  
  evidence: `2015-sugimura-tracheal-viscous-fingering.py` L35–51
- **Morphometric validation: 52** — `measure_morphoeval()` returns only 3 scalar values without Pearson correlations, RMSE, p-values, common-mode gauge invariance verification, nifedipine ASM blockade and tracheal drainage negative controls, or empirical specimen CV morphometrics.  
  evidence: `2015-sugimura-tracheal-viscous-fingering.py` L35–43
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `tracheal_viscous_fingering_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/tracheal_viscous_fingering/tracheal_viscous_fingering_reference.json` L1–185
- **Code-to-model traceability: 56** — Module docstring cites Nelson et al. (2017) and Kim et al. (2015), but lacks explicit equation-to-code and Lean 4 theorem mappings.  
  evidence: `2015-sugimura-tracheal-viscous-fingering.py` L1–6
- **Reproducibility package: 54** — Includes a minimal 3-line `--self-test` flag, but lacks SHA-256 dataset integrity checks, common-mode gauge invariance checks, Laplace hoop stress monotonicity, and nifedipine / tracheal drainage negative controls.  
  evidence: `2015-sugimura-tracheal-viscous-fingering.py` L79–82

Remediation:
- Implement a coupled 2D spatial branching airway tree & luminal pressure / Laplace epithelial hoop stress solver across Delta P in [0, 400] Pa, coupling sigma_theta = Delta P * R / h, synchronized ASM Ca2+ peristaltic wave frequency f_ASM(Delta P), and FGF10/ERK1/2/Etv4 tip bifurcation.
- Verify exact common-mode gauge invariance across independent microfluidic chest pressure pairs (P_lumen, P_pleura) = (200, 0), (0, -200), (500, 300), and (200, 200) Pa, alongside nifedipine L-type Ca2+ channel ASM blockade and tracheal drainage (Delta P = 0 Pa) negative controls.
- Segment the vendored Mus musculus pseudoglandular embryonic lung branching specimen (`embryonic_lung_branching_specimen.jpg`) to extract distal-to-total area fraction, terminal bud tip count (17 tips), effective bifurcation generations (4.09), and mean distal tip radius (3.85 px), backed by runtime SHA-256 verification and an 8-suite `--self-test`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Nelson et al. (2017) Development & Kim et al. (2015) Dev. Cell microfluidic transmural pressure, Laplace epithelial wall stress, and ASM peristalsis driving embryonic lung branching
- **spec_fingerprint** (semantic model + inputs): `3ff5590a2e90cba39f3787578e96d03eba17c0eb535c7b4f860a74e9898f90de`
- **input_dataset_sha256**: `datasets/tracheal_viscous_fingering/embryonic_lung_branching_specimen.jpg` (`8086ebb8b44cbdff…`), `datasets/tracheal_viscous_fingering/paper_citations.bib` (`147c24db66fc3582…`), `datasets/tracheal_viscous_fingering/scout_manifest.json` (`5ca9d0b7e3b9f2d2…`), `datasets/tracheal_viscous_fingering/tracheal_viscous_fingering_reference.json` (`da4d1143ccb80d60…`)
- `docs/media/tracheal_viscous_fingering_comparison.png` — artifact_sha256 `c3caf793f5996b27…`, 20,665 B, 720×480 px

| measurement | value | reference | method |
|---|---:|---|---|
| Terminal buds at 48h under 50 Pa hypoplastic pressure | 6.0 buds | 6.2 | Pass 1 scalar Hill formula at Delta P = 50 Pa |
| Terminal buds at 48h under 200 Pa physiological pressure | 28.2655 buds | 28.5 | Pass 1 scalar Hill formula at Delta P = 200 Pa |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2015-sugimura-tracheal-viscous-fingering.py` prior to Step 12 closed-loop remediation.

# MorphoEval report — audit `2026-10-19-corticothalamic-seizure-spirals-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Huang et al. (2004, 2010) & Pinto & Ermentrout (2001) neocortical spiral wave dynamics & phase singularity topology (`2013-ermentrout-corticothalamic-seizure-spirals.py`) | 56 | 50 | 55 | 54 | 66 | 60 | 60 | 55.7 | 51.5 | 62.0 | 55.9 | **55.9** | partially faithful |

## Findings and remediation

### Huang et al. (2004, 2010) & Pinto & Ermentrout (2001) neocortical spiral wave dynamics & phase singularity topology  —  overall 55.9 (partially faithful)

Pass 1 baseline assessment of `2013-ermentrout-corticothalamic-seizure-spirals.py` (56.9/100). The baseline script evaluates a static kinematic Archimedes spiral wave without solving the 2D excitatory-adaptation Wilson-Cowan / Pinto-Ermentrout neural field PDE tau_u du/dt = -u + D_u nabla^2 u + F(w_ee K_e * u - g_a a + I_ext), tau_a da/dt = -a + u, omits automatic 2x2 plaquette topological phase winding number oint nabla phi . dl singularity detection, omits counter-rotating figure-eight spiral pair (W = +1, -1) dipole conservation and CNQX/AMPA glutamatergic blockade quiescent controls, and displays the tangential rat neocortical cytochrome oxidase specimen without automated CV column segmentation.

- **Model fidelity to source: 56** — Evaluates only a kinematic Archimedes phase spiral (L43-64), omitting time-stepping of the 2D Wilson-Cowan / Pinto-Ermentrout excitatory-adaptation neural field PDE with 2D FFT Gaussian axonal footprint convolution K_e * u and complex state-space / Hilbert phase extraction Z(x,t) = (u - u_bar) + i(a - a_bar).  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L43–64
- **Morphological realism / specimen match: 50** — Vendors `datasets/corticothalamic_seizure_spirals/rat_neocortical_tangential_sheet_specimen.jpg` (tangential cytochrome oxidase histology section of Rattus norvegicus somatosensory neocortex Layer IV barrel sheet), but Pass 1 only displays the raw image without automated CV segmentation of cortical columns/barrels or radial intensity profile fitting.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L67–81; `docs/media/corticothalamic_seizure_spirals_comparison.png` L0–0
- **Parameter expressivity: 55** — Evaluates only a single counterclockwise kinematic spiral, omitting CW mirror chirality (W = -1), broken planar wave figure-eight counter-rotating spiral dipole (W_+ = +1, W_- = -1, sum W_i = 0), Pinto-Ermentrout expanded axonal footprint (sigma_e = 0.63 mm), planar wave control (W = 0), CNQX/AMPA blockade quiescent control (w_ee = 0.8), and a 2D (w_ee, g_a) Hopf spiral morphospace.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L43–64
- **Morphometric validation: 54** — Reports only 2 baseline metrics in `measure_morphoeval()`, lacking plaquette topological winding number verification, radial core amplitude deficit fit r against A_inf tanh(r/r_core), wave speed dispersion c = lambda * f_rot, CNQX blockade amplitude suppression, and specimen CV morphometrics.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L98–121
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `corticothalamic_seizure_spirals_reference.json`, `paper_citations.bib`, and `rat_neocortical_tangential_sheet_specimen.jpg` with SHA-256 verification linking Huang et al. (2004, 2010), Pinto & Ermentrout (2001), and Townsend et al. (2015).  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L32–41
- **Code-to-model traceability: 60** — Links baseline functions to Huang et al. (2010) and Pinto & Ermentrout (2001), requiring full alignment with all 3 Lean 4 theorems in `Loop_089_corticothalamic_seizure_spirals.lean`.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L1–7
- **Reproducibility package: 60** — Provides `--self-test` and deterministic execution, requiring expansion to an 8-check verification suite covering all 6 regimes, Lean 4 invariants, plaquette winding detection, specimen CV segmentation, and bitwise two-run determinism.  
  evidence: `2013-ermentrout-corticothalamic-seizure-spirals.py` L124–130

Remediation:
- Implement the 2D Wilson-Cowan / Pinto-Ermentrout excitatory-adaptation neural field PDE solver with 2D FFT Gaussian axonal footprint K_e * u, complex state-space phase map phi(x,y,t), and automatic 2x2 plaquette topological winding number oint nabla phi . dl singularity tracker.
- Simulate all 6 canonical regimes (CCW W=+1 spiral, CW W=-1 mirror spiral, figure-eight counter-rotating dipole W=+1/-1 with net charge 0, expanded axonal footprint fast spiral, planar wave W=0, and CNQX/AMPA glutamatergic blockade quiescent control), plus a 10x10 (w_ee, g_a) Hopf spiral morphospace.
- Add automated CV segmentation of the tangential Rattus norvegicus neocortical cytochrome oxidase specimen (barrel column centroids, inter-column spacing, and radial optical density profile), and render a 9-panel dark-mode synthesis figure and 24-frame animated GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Huang et al. (2004, 2010) & Pinto & Ermentrout (2001) neocortical spiral wave dynamics & phase singularity topology
- **spec_fingerprint** (semantic model + inputs): `8e19f210a705689fed055999b0cf4bc7b4186b66538d2729cc57db348f32af80`
- **input_dataset_sha256**: `datasets/corticothalamic_seizure_spirals/corticothalamic_seizure_spirals_reference.json` (`3570a9b466035f9e…`), `datasets/corticothalamic_seizure_spirals/paper_citations.bib` (`2ea747169ffdca2a…`), `datasets/corticothalamic_seizure_spirals/rat_neocortical_tangential_sheet_specimen.jpg` (`c4fdfb27cd39c842…`), `datasets/corticothalamic_seizure_spirals/scout_manifest.json` (`e31d28291faa189a…`)
- `docs/media/corticothalamic_seizure_spirals_comparison.png` — artifact_sha256 `168c38a1491d8d8b…`, 417,433 B, 1260×983 px

| measurement | value | reference | method |
|---|---:|---|---|
| Huang et al. (2010) neocortical spiral rotation frequency f_rot | 10.5 Hz | 10.5 | Baseline kinematic spiral frequency *(Pass 1 kinematic baseline frequency)* |
| Spiral phase singularity core minimum amplitude A_min | 0.1218 normalized amplitude | 0.08 | Minimum radial envelope amplitude at spiral center *(Core minimum amplitude = 0.1218)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2013-ermentrout-corticothalamic-seizure-spirals.py` in `2026-10-19-corticothalamic-seizure-spirals-loop` prior to Pass 2 closed-loop remediation.

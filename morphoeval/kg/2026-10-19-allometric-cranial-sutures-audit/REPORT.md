# MorphoEval report — audit `2026-10-19-allometric-cranial-sutures-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Miura et al. (2009) & Jaslow (1990) cranial suture fractal interdigitation and Kuramoto-Sivashinsky osteogenic instability (`2020-bailleul-allometric-cranial-sutures.py`) | 58 | 50 | 55 | 54 | 66 | 60 | 60 | 57.0 | 51.5 | 62.0 | 56.3 | **56.3** | partially faithful |

## Findings and remediation

### Miura et al. (2009) & Jaslow (1990) cranial suture fractal interdigitation and Kuramoto-Sivashinsky osteogenic instability  —  overall 56.3 (partially faithful)

Pass 1 baseline assessment of `2020-bailleul-allometric-cranial-sutures.py` (57.3/100). The baseline script implements a 1D single-valued linear spectral amplification of the Kuramoto-Sivashinsky dispersion relation omega(k) = alpha*k^2 - beta*k^4 - gamma for sagittal vs mechanically unloaded sutures, but omits the nonlinear KPZ normal-growth coupling (v_0/2)*(dh/dx)^2, 2D bidirectional overhanging phase-field / normal-front interdigitation (needed to reach SII > 2.5 and box-counting fractal dimension D_f ~ 1.35), infant neonatal, adult lambdoid, and FGFR2 Apert craniosynostosis regimes, Jaslow (1990) bending energy absorption quantification, and automated CV suture trace segmentation on `human_sagittal_suture_specimen.jpg`.

- **Model fidelity to source: 58** — Implements linear fourth-order dispersion omega(k) = alpha*k^2 - beta*k^4 - gamma on a single-valued 1D graph h(x, t) (L46-81), omitting pseudo-spectral semi-implicit / ETDRK4 integration of the nonlinear KPZ term (v_0/2)*(dh/dx)^2 + stochastic osteogenic noise eta(x,t) and 2D overhanging bone-finger interdigitation from Miura et al. (2009).  
  evidence: `2020-bailleul-allometric-cranial-sutures.py` L46–81
- **Morphological realism / specimen match: 50** — Vendors `datasets/allometric_cranial_sutures/human_sagittal_suture_specimen.jpg` (CC BY-SA 3.0 human calvaria showing sagittal, coronal, and lambdoid sutures), but Pass 1 only renders the raw photograph without automated CV segmentation of the interdigitated sagittal suture trace, box-counting fractal dimension D_f, or SII.  
  evidence: `2020-bailleul-allometric-cranial-sutures.py` L88–114; `docs/media/allometric_cranial_sutures_comparison.png` L0–0
- **Parameter expressivity: 55** — Evaluates only 2 regimes (sagittal vs mechanically unloaded), omitting neonatal infant flat suture, adult hyper-interdigitated lambdoid suture, FGFR2 Apert craniosynostosis fusion, and a 2D (alpha, beta) morphospace sweep.  
  evidence: `2020-bailleul-allometric-cranial-sutures.py` L85–112
- **Morphometric validation: 54** — Reports only 2 single-valued SII scalars in `measure_morphoeval()`, omitting log-log box-counting fractal dimension D_f, power spectral density scaling slope, Jaslow (1990) bending energy absorption, and empirical specimen suture D_f and SII.  
  evidence: `2020-bailleul-allometric-cranial-sutures.py` L137–161
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `allometric_cranial_sutures_reference.json`, `paper_citations.bib`, and `human_sagittal_suture_specimen.jpg` with SHA-256 verification linking Miura et al. (2009) and Jaslow (1990).  
  evidence: `2020-bailleul-allometric-cranial-sutures.py` L34–43
- **Code-to-model traceability: 60** — Links baseline dispersion to Miura et al. (2009), requiring full alignment with all 3 Lean 4 theorems in `Loop_020_allometric_cranial_sutures.lean`.  
  evidence: `2020-bailleul-allometric-cranial-sutures.py` L1–8
- **Reproducibility package: 60** — Provides `--self-test` and deterministic seed execution, requiring expansion to an 8-check verification suite covering all 5 developmental/pathological regimes, box-counting fractal dimension D_f, Jaslow energy scaling, and specimen CV extraction.  
  evidence: `2020-bailleul-allometric-cranial-sutures.py` L164–173

Remediation:
- Implement pseudo-spectral semi-implicit / ETDRK4 Kuramoto-Sivashinsky + KPZ nonlinear normal-growth cascade and 2D bidirectional overhanging phase-field / normal-front interdigitation producing multi-valued suture fingers with D_f ~ 1.35 and SII > 2.5.
- Simulate all 5 canonical regimes (infant neonatal flat suture, adult sagittal fractal suture, adult lambdoid hyper-interdigitated suture, mechanical unloading null control, and FGFR2 Apert craniosynostosis fusion) plus a 15x15 (alpha, beta) morphospace and Jaslow (1990) bending energy curve.
- Add automated CV extraction of the sagittal suture trace on `human_sagittal_suture_specimen.jpg`, measure empirical box-counting fractal dimension D_f and SII, and render a 9-panel dark-mode comparison figure and 24-frame animated GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Miura et al. (2009) & Jaslow (1990) cranial suture fractal interdigitation and Kuramoto-Sivashinsky osteogenic instability
- **spec_fingerprint** (semantic model + inputs): `53dc68ef4189bb186f820ba54b5a160dfe42da44326e82b66ab860cc7d2ad00e`
- **input_dataset_sha256**: `datasets/allometric_cranial_sutures/allometric_cranial_sutures_reference.json` (`dbb634957fb6941d…`), `datasets/allometric_cranial_sutures/human_sagittal_suture_specimen.jpg` (`b3c411a6a4713770…`), `datasets/allometric_cranial_sutures/paper_citations.bib` (`bffc86bbbd2dd3d6…`), `datasets/allometric_cranial_sutures/scout_manifest.json` (`226c072307597439…`)
- `docs/media/allometric_cranial_sutures_comparison.png` — artifact_sha256 `bffa061ad0a44e38…`, 422,533 B, 1380×936 px

| measurement | value | reference | method |
|---|---:|---|---|
| Baseline single-valued sagittal suture interdigitation index SII | 1.1116 L_arc / L_chord | 2.75 | 1D single-valued linear spectral evolution without KPZ cascade or 2D overhang *(Single-valued baseline SII = 1.1116 vs unloaded SII = 1.0000)* |
| Unloaded mechanical control suture interdigitation index SII | 1.0 L_arc / L_chord | 1.0 | 1D single-valued evolution with alpha = -0.35 *(Unloaded suture collapses to flat line SII = 1.0000)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2020-bailleul-allometric-cranial-sutures.py` in `2026-10-19-allometric-cranial-sutures-loop` prior to Pass 2 closed-loop remediation.

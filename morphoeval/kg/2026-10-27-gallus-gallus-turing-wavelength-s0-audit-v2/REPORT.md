# MorphoEval report — audit `2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `d996d1f6820869dd1e2694a6d36819f4719a6704` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gallus-gallus-turing-wavelength-s0-audit -> 2026-10-27-gallus-gallus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gallus gallus Holocene Turing pigment-ocellus wavelength, 319.2 µm (pass 2) (`models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py`) | 88 | 83 | 86 | 87 | 90 | 85 | 88 | 87.3 | 84.5 | 87.9 | 86.4 | **86.4** | faithful |

## Findings and remediation

### Gallus gallus Holocene Turing pigment-ocellus wavelength, 319.2 µm (pass 2)  —  overall 86.4 (faithful)

Pass 2 keeps λ_c = 319.2 µm at 0 Ma on the singleton Gallus chronoseries. Inhibitor diffusivity is swept at fixed γ: the Turing margin crosses 1 and λ_c falls as Dv rises. A 1800-step Schnakenberg integration grows the critical mode at Dv = 16 (std 0.021 to 0.026) and decays it under equal diffusion (std 0.0025). The gallery plate's contiguous scale bar is 110 px, so the painted pitch is 66.4 µm, which is not 319.2 µm. A three-harmonic radial model trained on even radii has held-out Pearson r = 0.9855 against a detuned-wavenumber null of lower correlation, with held-out RMSE 5.50 and KS 0.084. Structure-tensor coherency of the rendered frame (0.958) is compared with the plate (0.846). The 12-frame 800×600 GIF scores CV similarity 90.5 against that plate.

- **Model fidelity to source: 88** — The onset formula is solved for λ_c = 319.2 µm and the same Jacobian is integrated: the critical cosine grows at Dv = 16 and decays when Dv = Du. This is Schnakenberg wavelength selection, not a transcription of the Prum & Williamson feather-growth coordinate system.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L89–109; `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L249–270; `morphoeval/lean/loops/Loop_3228_gallus_gallus_turing_wavelength_s0.lean` L47–57
- **Morphological realism / specimen match: 83** — The saturated GIF frame is a centered purple ocellus with the plate's grain and scores CV similarity 90.5 at 800×600 over 12 frames. The comparison figure places the plate beside the reconstruction. The likeness is a fit to this one plate.  
  evidence: `docs/media/gallus_gallus_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gallus_gallus_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 86** — At fixed γ, raising Dv moves the Turing margin up through 1 and shortens λ_c on the unstable branch. The equal-diffusion end of the same family is stable. Amplitude in the GIF is the Stuart-Landau coordinate from 0 to saturation.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L239–247; `docs/media/gallus_gallus_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 87** — Held-out Pearson r = 0.9855 versus a detuned null, held-out RMSE = 5.50 versus the null RMSE, KS = 0.084 versus a 0.5 reference, and structure-tensor coherency 0.958 on the render versus 0.846 on the plate. The painted pitch is 66.4 µm against the model wavelength 319.2 µm. All of these references differ from the measured values. The fit is still one plate.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L192–237; `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L139–160; `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L560–653
- **Source and data traceability: 90** — DOIs for Turing 1952, Prum & Williamson 2002, Harris et al. 2005, and Mou et al. 2011. The ocellus plate is SHA-256 pinned and force-tracked. The living bird is a CC BY 4.0 iNaturalist photograph by dhfischer.  
  evidence: `datasets/gallus_gallus_turing_wavelength_s0/paper_citations.bib` L1–43; `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_turing_wavelength_s0_reference.json` L1–70; `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_ocellus_specimen.jpg` L0–0
- **Code-to-model traceability: 85** — Onset, sweep, ablation, and the radial fit are separate functions, and the Lean file proves the onset gap, the ablation sign, and the empty predecessor/successor. The renderer is the isotropic critical mode, identified as such in the claim boundary.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L381–415; `morphoeval/math_kg_loops/delphi10000-gallus-gallus-turing-wavelength-s0.yaml` L20–65
- **Reproducibility package: 88** — --self-test now requires λ_c = 319.2 µm, held-out r > 0.98, a monotonic unstable branch, growth under Dv = 16, and decay under equal diffusion. Seed 3228. Invocation is in the dataset README.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L655–678; `datasets/gallus_gallus_turing_wavelength_s0/README.md` L28–36

Remediation:
- {'id': 'R6', 'item': 'Still one gallery plate. A second calibrated micrograph of a Gallus feather ocellus would separate the fit from the picture it was fit to.'}
- {'id': 'R7', 'item': 'Still a single assessor. An independent second provider was not available in this run.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Gallus gallus Holocene Turing pigment-ocellus wavelength, 319.2 µm (pass 2)
- **spec_fingerprint** (semantic model + inputs): `d79e485299e3508a55f35819cf41b353933badb10f650ee2ade00ae23887b314`
- **input_dataset_sha256**: `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_ocellus_specimen.jpg` (`7e7d58db6bb564eb…`), `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_turing_wavelength_s0_reference.json` (`5589d95bf0b106f7…`)
- `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_ocellus_specimen.jpg` — artifact_sha256 `7e7d58db6bb564eb…`, 99,083 B, 800×600 px
- `docs/media/gallus_gallus_turing_wavelength_s0_comparison.png` — artifact_sha256 `eebe3c4bcde6058f…`, 406,896 B, 1536×888 px
- `docs/media/gallus_gallus_turing_wavelength_s0_growth.gif` — artifact_sha256 `a58a55b4d7b5a3f6…`, 5,553,246 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing onset wavelength lambda_c | 319.2 µm | 66.3636 | k_c^2 = (Dv fu + Du gv) / (2 Du Dv) with γ solved for 319.2 µm *(Model wavelength and painted-bar pitch are different lengths)* |
| specimen radial profile held-out Pearson r | 0.9855 correlation | 0.7704 | 3-harmonic critical mode fit on even radii, Pearson r on odd radii *(in-sample r = 0.9855; RMSE = 5.5030)* |
| specimen radial profile held-out RMSE | 5.503 gray levels | 20.6863 | RMSE of odd-radius residuals of the 3-harmonic reconstruction *(lower residual than the null)* |
| specimen radial profile KS statistic | 0.084 KS distance | 0.5 | two-sample KS between held-out specimen radii and the reconstruction *(p = 0.7971)* |
| specimen plate structure-tensor coherency | 0.9584 coherency | 0.8463 | verify_image_fitness.compute_specimen_cv_signature on the rendered steady frame *(plate foreground=0.4448 circularity=0.0719; model foreground=0.7762 circularity=0.1467)* |
| ablation control activator std ratio | 10.3945 std(Dv=16) / std(Dv=Du) | 1.0 | 1800-step periodic Schnakenberg integration, seed cos(2π x / 24) *(Turing std 0.02583 vs ablation std 0.00248)* |
| Dv sweep wavelength span | 9.8535 µm | 0.0 | λ_c(Dv) at fixed γ on the unstable branch; span is first minus last *(Positive span means λ_c falls as Dv rises above onset)* |

## Environment at checkpoint

- python: 3.12.3
- note: numpy, scipy, pillow, matplotlib, pyyaml, rdflib

Pass 2 on top of 8be940bb. Sweep, ablation, CV segmentation, held-out fit, scale bar, tracked plate.
